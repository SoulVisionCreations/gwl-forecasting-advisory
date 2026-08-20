"""TFTModelRunner — load the trained TFT (+Prithvi) once and run batched forward.

REUSES, no duplication:
  - tft_model.GWLForecastTFT            (the exact architecture trained)
  - prithvi_finetune.build_projector    (rebuilds LoRA encoder + projector skeleton)
  - prithvi_finetune.TileStore          (per-request tile loader, swapped in at predict)

Load recipe (mirrors train.py:2620-2690 exactly, incl. the max(.,1) embedding
guards), but reads the dims from the checkpoint's `config` DICT instead of a live
DataConfig object:

    ckpt = torch.load(run_dir/best_model.pt, weights_only=False)
    cfg  = ckpt["config"]                       # <-- a dict, use cfg["key"]
    proj = build_projector(manifest, model_dir=cfg["prithvi_model_dir"], ...)
    model = GWLForecastTFT(**shared(cfg), d_model=cfg["tft_d_model"], ...,
                           use_prithvi=cfg["use_prithvi"], prithvi_projector=proj)
    model.load_state_dict(ckpt["model_state_dict"])   # restores TFT + LoRA + projector

The projector built here wraps a TileStore over the TRAINING manifest (485k tiles).
At inference the K neighbours have their OWN small composite set with tile_idx
indexing into it, so predict() swaps `model.prithvi.store` for a fresh per-request
TileStore (reusing the loaded store's band-stats + lat/lon, so no re-read) and
resets `model.prithvi.zero_idx`. This is why parse_composite_filename's basename
requirement matters: the swapped store gets basenames + an explicit composite_dir.

predict() returns RAW model outputs (scaled delta + trend) aligned to the input
order. Inverse-scaling the delta and assembling StationPrediction is the engine's
job (it owns the FeatureScaler and current_gwl), keeping this class purely "run the
net".
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import torch


# Shared embedding/dim kwargs whose checkpoint-config key == GWLForecastTFT param
# name. The four wrapped in max(.,1) below are guarded exactly as train.py does.
_SHARED_KEYS = (
    "num_timesteps",
    "features_per_timestep",
    "static_embedding_dim",
    "fusion_hidden_dim",
    "num_lithology_types",
    "num_well_types",
    "num_aquifer_types",
    "num_lulc_types",
    "lithology_embedding_dim",
    "well_type_embedding_dim",
    "aquifer_embedding_dim",
    "aquifer_0_aquifer_embedding_dim",
    "litho_supergroup_embedding_dim",
    "lulc_embedding_dim",
    "state_embedding_dim",
    "district_embedding_dim",
    "num_forecast_features",
    "dropout_rate",
    "use_static_features",
    "use_linear_residual",
)

# Model forward()/predict() kwargs that come straight from a scaled sample.
_FLOAT_FIELDS = ("sequence", "forecast_features", "static_continuous")
_INT_FIELDS = (
    "lithology_idx", "well_type_idx", "aquifer_idx", "aquifer_0_aquifer_idx",
    "litho_supergroup_idx", "state_idx", "district_idx",
    "forecast_lulc_idx", "historical_lulc_indices",
)


def _shared_kwargs(cfg: dict) -> dict:
    """Reconstruct train.py's shared_model_kwargs from the checkpoint config dict,
    replicating its max(.,1) guards for the optional categorical cardinalities."""
    kw = {k: cfg[k] for k in _SHARED_KEYS}
    kw["num_aquifer_0_aquifer_types"] = max(cfg["num_aquifer_0_aquifer_types"], 1)
    kw["num_litho_supergroup_types"] = max(cfg["num_litho_supergroup_types"], 1)
    kw["num_state_types"] = max(cfg["num_state_types"], 1)
    kw["num_district_types"] = max(cfg["num_district_types"], 1)
    return kw


class TFTModelRunner:
    """Holds the loaded model + (optional) Prithvi projector for repeated predicts."""

    def __init__(self, model, cfg: dict, manifest: Optional[dict], device: str):
        self.model = model
        self.cfg = cfg
        self.manifest = manifest
        self.device = device
        self.use_prithvi = bool(cfg.get("use_prithvi", False))
        # RevIN: training applies per-instance instance-norm to the GWL channel
        # OUTSIDE the model (train.py Trainer), so we must replicate it here.
        self.use_revin = bool(cfg.get("use_revin", False))
        self.revin_std_floor = float(cfg.get("revin_std_floor", 0.1))
        # The projector is attached on the TFT as `.prithvi` (tft_model.py:401).
        self.projector = getattr(model, "prithvi", None) if self.use_prithvi else None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        run_dir: str,
        device: Optional[str] = None,
        model_dir: Optional[str] = None,
        station_csv: Optional[str] = None,
    ) -> "TFTModelRunner":
        """Rebuild the exact trained model from `run_dir/best_model.pt` and load weights.

        model_dir / station_csv override the Prithvi paths baked into the checkpoint
        config (use when the run was trained on a host with different absolute paths).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_path = os.path.join(run_dir, "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]                      # DICT (not an object) — dict access
        if not isinstance(cfg, dict):
            raise TypeError(
                f"Expected ckpt['config'] to be a dict; got {type(cfg).__name__}."
            )

        # Build the live Prithvi projector exactly as train.py does (only for TFT+Prithvi).
        manifest = None
        projector = None
        if cfg.get("use_prithvi", False):
            from gwlcore.prithvi_finetune import build_projector

            manifest_path = os.path.join(run_dir, "data", "tile_manifest.pkl")
            with open(manifest_path, "rb") as mf:
                manifest = pickle.load(mf)
            projector = build_projector(
                manifest,
                model_dir=model_dir or cfg["prithvi_model_dir"],
                station_csv=station_csv or cfg["station_index_csv"],
                proj_dim=cfg["prithvi_proj_dim"],
                lora_r=cfg["lora_r"],
                lora_alpha=cfg["lora_alpha"],
                device=device,
                # FAT bundle (not slim): the backbone is baked into best_model.pt and the
                # strict load below fills it, so skip the base *.pt load — the standalone
                # weights/ then needs only prithvi_mae.py + config.json, no 1.3G base .pt.
                exclude_ckpt=not ckpt.get("slim", False),
            )

        from gwlcore.tft_model import GWLForecastTFT

        model = GWLForecastTFT(
            **_shared_kwargs(cfg),
            d_model=cfg["tft_d_model"],
            n_heads=cfg["tft_n_heads"],
            lstm_layers=cfg["tft_lstm_layers"],
            tft_dropout=cfg["tft_dropout"],
            use_prithvi=cfg.get("use_prithvi", False),
            prithvi_proj_dim=cfg.get("prithvi_proj_dim", 32),
            prithvi_projector=projector,
        )
        state_dict = ckpt["model_state_dict"]
        if ckpt.get("slim"):
            # SLIM ckpt: the frozen Prithvi base was stripped at package time (it is
            # public pretrained weights, byte-identical to what build_projector just
            # loaded from the Prithvi dir). Load the trained delta (TFT + LoRA +
            # projector) non-strictly; the base stays as reconstructed from the dir.
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"slim ckpt has unexpected keys (not in the model): "
                    f"{list(incompatible.unexpected_keys)[:5]}")
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            leaked = trainable.intersection(incompatible.missing_keys)
            if leaked:  # a TRAINED param wasn't shipped — the slim strip was wrong
                raise RuntimeError(
                    f"slim ckpt is missing trained params: {sorted(leaked)[:5]}")
        else:
            # strict=True: full ckpt reconstructs identically — every key (TFT, frozen
            # Prithvi base, LoRA adapters, projector) must match.
            model.load_state_dict(state_dict)
        # Symmetry with the Prithvi/scaler load lines — make the TFT head load visible too.
        # `.prithvi` is the projector (wraps the frozen encoder + LoRA), so params NOT under
        # `prithvi.*` are the pure TFT forecaster head.
        _n_tft = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("prithvi"))
        _n_total = sum(p.numel() for p in model.parameters())
        print(f"[ft] loaded TFT forecaster head (d_model={cfg.get('tft_d_model')}, "
              f"heads={cfg.get('tft_n_heads')}, lstm_layers={cfg.get('tft_lstm_layers')}) "
              f"+ projector from best_model.pt | TFT-head params {_n_tft:,} / {_n_total:,} total")
        model.to(device).eval()

        return cls(model, cfg, manifest, device)

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(
        self,
        scaled_samples: "list[dict]",
        ordered_composite_paths: "Optional[list[str]]" = None,
        zero_idx: Optional[int] = None,
    ) -> dict:
        """Run the net over already-scaled samples.

        ordered_composite_paths / zero_idx are required when use_prithvi: they define
        the per-request TileStore that tile_idx (set by SampleBuilder) indexes into.

        Returns numpy arrays aligned to scaled_samples order:
            { "delta_scaled": (N,), "trend_prob": (N,), "trend_class": (N,) }
        For an empty input, returns empty arrays.
        """
        n = len(scaled_samples)
        if n == 0:
            return {k: np.empty(0, np.float32) for k in ("delta_scaled", "trend_prob", "trend_class")}

        if self.use_prithvi:
            if ordered_composite_paths is None or zero_idx is None:
                raise ValueError(
                    "use_prithvi model: predict() needs ordered_composite_paths and zero_idx "
                    "(the per-request tile set that tile_idx indexes into)."
                )
            self._bind_tile_store(ordered_composite_paths, int(zero_idx))

        batch = self._collate(scaled_samples)

        # RevIN (train.py validate(): norm GWL channel pre-forward, de-norm post).
        mean_s = std_s = None
        if self.use_revin:
            batch["sequence"], mean_s, std_s = self._apply_revin(batch["sequence"])

        out = self.model.predict(**batch)        # {gwl_pred (N,1), ...}
        gwl_pred = out["gwl_pred"].squeeze(-1)   # (N,)
        if self.use_revin:
            gwl_pred = gwl_pred * std_s + mean_s  # de-norm → scaler-scaled delta

        # Trend from the DE-NORMED scaled delta, matching train.py
        # _compute_trend_logit_and_target (delta mode: trend_logit = pred_global).
        # model.predict's own sigmoid is of the PRE-de-norm output, so under RevIN it
        # would disagree with training; recompute here so trend is exact-parity.
        # (RevIN off → gwl_pred is unchanged, so this equals model.predict's trend.)
        trend_prob = torch.sigmoid(gwl_pred)
        trend_class = (trend_prob >= 0.5).long()

        return {
            "delta_scaled": gwl_pred.cpu().numpy().astype(np.float32),
            "trend_prob": trend_prob.cpu().numpy().astype(np.float32),
            "trend_class": trend_class.cpu().numpy().astype(np.int64),
        }

    def _apply_revin(self, sequence):
        """Per-instance instance-norm of the GWL channel (idx 0) over present
        timesteps (idx 1 = is_present); mirrors train.py Trainer._apply_revin.
        Returns (sequence_normed, mean_s (N,), std_s (N,))."""
        gwl = sequence[..., 0:1]
        is_present = sequence[..., 1:2]
        n_present = is_present.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_s = (gwl * is_present).sum(dim=1, keepdim=True) / n_present
        var_s = (((gwl - mean_s) * is_present) ** 2).sum(dim=1, keepdim=True) / n_present
        std_s = (var_s + 1e-5).sqrt().clamp(min=self.revin_std_floor)
        gwl_normed = ((gwl - mean_s) / std_s) * is_present
        seq_normed = torch.cat([gwl_normed, sequence[..., 1:]], dim=-1)
        return seq_normed, mean_s.squeeze(-1).squeeze(-1), std_s.squeeze(-1).squeeze(-1)

    # ----------------------------------------------------------- internals
    def _collate(self, scaled_samples: "list[dict]") -> dict:
        """Stack scaled samples into model kwargs, mirroring LSTMTensorDataset dtypes
        (train.py:320-417). tile_idx is included only for use_prithvi (the other
        models' forward() doesn't accept it)."""
        dev = self.device
        batch = {}
        for f in _FLOAT_FIELDS:
            arr = np.asarray([s[f] for s in scaled_samples], dtype=np.float32)
            batch[f] = torch.from_numpy(arr).to(dev)
        for f in _INT_FIELDS:
            arr = np.asarray([s[f] for s in scaled_samples], dtype=np.int64)
            batch[f] = torch.from_numpy(arr).to(dev)
        if self.use_prithvi:
            tile = np.asarray([int(s.get("tile_idx", -1)) for s in scaled_samples], dtype=np.int64)
            batch["tile_idx"] = torch.from_numpy(tile).to(dev)
        return batch

    def _bind_tile_store(self, paths: "list[str]", zero_idx: int) -> None:
        """Swap the projector's TileStore for one over THIS request's composites.

        Reuses the loaded store's band mean/std and lat/lon (so no config.json /
        station-csv re-read), and the manifest's period/period_doy. parse_composite_
        filename needs basenames (anchored regex), so we pass basenames + an explicit
        composite_dir; CompositeFetcher writes every tile into one cache_dir, so a
        single common dir is guaranteed (asserted)."""
        from gwlcore.prithvi_finetune import TileStore

        # Invariant: only reached for use_prithvi, where load() set both of these.
        assert self.projector is not None and self.manifest is not None, (
            "_bind_tile_store requires a loaded Prithvi projector and manifest"
        )
        old = self.projector.store
        basenames = [os.path.basename(p) for p in paths]
        dirs = sorted({os.path.dirname(p) for p in paths})
        if len(dirs) > 1:
            raise NotImplementedError(
                f"Composites span multiple dirs {dirs}; per-request TileStore assumes a "
                "single cache_dir. Point CompositeFetcher at one cache_dir."
            )
        composite_dir = dirs[0] if dirs else ""   # empty paths -> store never queried

        new_store = TileStore(
            ordered_files=basenames,
            composite_dir=composite_dir,
            period=self.manifest["period"],
            period_doy=self.manifest["period_doy"],
            latlon=old.latlon,                        # full training set (covers neighbours)
            band_mean=np.asarray(old.mean).reshape(-1),   # (6,1,1) -> (6,)
            band_std=np.asarray(old.std).reshape(-1),
        )
        self.projector.store = new_store
        self.projector.zero_idx = int(zero_idx)
