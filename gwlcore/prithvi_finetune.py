"""
prithvi_finetune.py  (GWL joint fine-tune components)
=====================================================
Building blocks for jointly fine-tuning the Prithvi-EO-2.0 satellite encoder
(LoRA) + a projection head together with the GWL TFT. The Prithvi encoder is IN
the training graph: each sample carries an integer ``tile_idx`` (annotated in
data_preparation.py) that points at one HLS composite tile; that tile is loaded,
encoded through Prithvi+LoRA, projected to ``proj_dim``, and concatenated into the
TFT's static features. Gradients flow GWL-loss -> TFT -> projector -> Prithvi LoRA.

What differs from the NDVI port (deliberately)
----------------------------------------------
* **No offline (mu, sd) standardizer.** The NDVI pipeline warm-started from a
  frozen ``embeddings.parquet`` and standardized pooled vectors with the (mu, sd)
  measured on that parquet. Here the TFT is trained FROM SCRATCH, so there is no
  warm-start to match. We replace the offline (mu, sd) with an inline
  ``LayerNorm(1024)`` applied to Prithvi's pooled output before the projection
  Linear -- it learns the same centering/scaling jointly.
* **Period-agnostic tiles.** Composites are half-yearly (H1/H2) or quarterly
  (Q1-Q4). A tile row's (safe_id, year, period_label) is parsed from the manifest
  basename; its day-of-year comes from ``period_doy``. Switch half<->quarter via
  the manifest only -- no code change here.
* **Band normalization** is read from the Prithvi ``config.json`` so it always
  matches the checkpoint's pretraining statistics.

LoRA scope
----------
timm/Prithvi ViT blocks use a FUSED ``attn.qkv`` Linear, so LoRA on ``qkv`` adapts
q, k, v together. We target ONLY the encoder blocks (the decoder is never run by
``forward_features``). LoRA initializes to zero -> at step 0 the encoder reproduces
the frozen base vectors; LoRA then adapts from there.

Compute
-------
``LivePrithviProjector.forward`` de-duplicates tile indices within the batch and
encodes each unique tile once. Paired with ``TileGroupedBatchSampler`` (few unique
tiles per batch), each tile is encoded ~once per epoch instead of once per sample.
"""

import os
import sys
import glob
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

# Single source of truth for the composite filename convention + doy map.
from .data_preparation import parse_composite_filename, PERIOD_DOY  # noqa: F401


def amp_ctx(device, enabled=True):
    """bf16 autocast on CUDA; no-op elsewhere."""
    if enabled and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------------------------------------------------------------------------
# Prithvi base + LoRA on encoder attention (fused qkv)
# ---------------------------------------------------------------------------
def prithvi_band_stats(model_dir):
    """Return (mean[6], std[6]) HLS band statistics from the checkpoint config."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)["pretrained_cfg"]
    return (np.asarray(cfg["mean"], np.float32),
            np.asarray(cfg["std"], np.float32))


def load_prithvi_lora(model_dir, r=16, alpha=32, dropout=0.05, num_frames=1,
                      device="cuda", target="qkv", exclude_ckpt=False):
    """Instantiate PrithviMAE, load .pt weights, freeze the base, attach LoRA to the
    encoder attention (``encoder.blocks.*.attn.<target>``). Returns (peft_model, cfg).

    ``num_frames=1`` makes this a single-snapshot spatial encoder; the pretrained
    ``pos_embed`` (sized for the checkpoint's num_frames) is dropped and recomputed.

    ``exclude_ckpt``: skip loading the base ``*.pt`` (build the arch skeleton only).
    Use for a FAT self-contained bundle where the frozen backbone is already baked into
    best_model.pt and a strict load_state_dict fills it — so the ~1.3G base .pt need not
    ship; only prithvi_mae.py + config.json are required in ``model_dir``. Parity-identical.
    """
    sys.path.insert(0, model_dir)
    from prithvi_mae import PrithviMAE
    from peft import LoraConfig, get_peft_model

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)["pretrained_cfg"]

    args = dict(cfg)
    args["num_frames"] = num_frames                 # T=1 spatial encoder
    model = PrithviMAE(**args)

    if exclude_ckpt:
        print("[ft] exclude_ckpt: skipped base *.pt load (backbone comes from the fat bundle)")
    else:
        ckpt = glob.glob(os.path.join(model_dir, "*.pt"))
        if not ckpt:
            raise SystemExit(f"No .pt checkpoint in {model_dir}")
        if os.path.getsize(ckpt[0]) < 10_000_000:
            raise SystemExit(
                f"{ckpt[0]} looks like a Git LFS pointer, not the model. Run:\n"
                f"  cd {model_dir} && git lfs install && git lfs pull")
        state = torch.load(ckpt[0], map_location="cpu", weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        for k in [k for k in state if "pos_embed" in k]:   # recomputed for T=1
            del state[k]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ft] loaded {os.path.basename(ckpt[0])} | missing={len(missing)} "
              f"unexpected={len(unexpected)} (pos_embed expected)")

    # LoRA only on encoder attention; decoder is never used by forward_features.
    n_blocks = len(model.encoder.blocks)
    targets = [f"encoder.blocks.{i}.attn.{target}" for i in range(n_blocks)]
    lora_cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                          target_modules=targets, bias="none")
    model = get_peft_model(model, lora_cfg)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[ft] LoRA r={r} alpha={alpha} drop={dropout} on {n_blocks} encoder "
          f"blocks ({target}) | trainable {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.3f}%)")
    return model.to(device), cfg


def encode_tokens(peft_model, img, tcoords, loc, pool="mean"):
    """forward_features -> last (LayerNorm'd) tokens -> pooled (B, embed_dim).
    Grad flows when called outside no_grad (training)."""
    feats = peft_model.forward_features(img, tcoords, loc)
    tokens = feats[-1]                                  # (B, 1+num_patches, D)
    if pool == "cls":
        return tokens[:, 0, :]
    return tokens[:, 1:, :].mean(dim=1)


# ---------------------------------------------------------------------------
# Tile store: tile_idx -> normalized (img, temporal_coords, location_coords)
# ---------------------------------------------------------------------------
class TileStore:
    """Maps a tile index (row in the manifest's ``ordered_files``) to a normalized
    tensor triple for Prithvi-TL: (img[6,224,224], temporal_coords[1,2]=(year,doy),
    location_coords[2]=(lat,lon)). Threaded batch loads + a bounded warm cache.

    Built directly from the tile_manifest written by data_preparation.save_datasets:
      ordered_files : list[basename]; row index == tile_idx
      period        : 'halfyear' | 'quarter'
      period_doy    : {period_label: day_of_year}
    plus a ``latlon`` dict {safe_id: (lat, lon)} (from gwl_stations.csv) and the
    Prithvi band ``mean``/``std`` (from config.json).
    """

    def __init__(self, ordered_files, composite_dir, period, period_doy, latlon,
                 band_mean, band_std, cache_size=20000, n_threads=8):
        self.dir = composite_dir
        self.files = list(ordered_files)
        self.latlon = latlon
        self.mean = np.asarray(band_mean, np.float32)[:, None, None]
        self.std = np.asarray(band_std, np.float32)[:, None, None]
        self.cache = {}
        self.cache_size = cache_size
        self.pool = ThreadPoolExecutor(max_workers=n_threads)

        # Pre-parse every row -> (safe_id, year, doy). Done once; rows are static.
        self.meta = []
        n_bad = 0
        for f in self.files:
            parsed = parse_composite_filename(f, period)
            if parsed is None:
                self.meta.append(None)
                n_bad += 1
                continue
            sid, year, label = parsed
            doy = float(period_doy.get(label, 182))
            self.meta.append((sid, float(year), doy))
        if n_bad:
            print(f"[ft][TileStore] WARNING: {n_bad} manifest rows did not match "
                  f"the {period!r} composite naming and will load as zeros.")

    def _path(self, idx):
        return os.path.join(self.dir, self.files[idx])

    def _load_one(self, idx):
        hit = self.cache.get(idx)
        if hit is not None:
            return hit
        import rasterio
        meta = self.meta[idx]
        if meta is None:                                # unparseable row -> neutral
            arr = np.zeros((self.mean.shape[0], 224, 224), np.float32)
            rec = (arr, np.array([[0.0, 182.0]], np.float32),
                   np.array([0.0, 0.0], np.float32))
            return rec
        sid, year, doy = meta
        with rasterio.open(self._path(idx)) as src:
            arr = src.read().astype(np.float32)         # (6, 224, 224)
            nd = src.nodata
        nodata = (arr == 0)
        if nd is not None:
            nodata |= (arr == nd)
        arr = (arr - self.mean) / self.std
        arr[nodata] = 0.0                               # neutral after norm
        lat, lon = self.latlon.get(sid, (0.0, 0.0))
        rec = (arr,
               np.array([[year, doy]], np.float32),     # (1,2): (year, doy)
               np.array([float(lat), float(lon)], np.float32))   # (2,)
        if len(self.cache) < self.cache_size:
            self.cache[idx] = rec
        return rec

    def load_batch(self, idxs, device):
        recs = list(self.pool.map(self._load_one, idxs))
        img = torch.from_numpy(np.stack([r[0] for r in recs])).to(device)
        tc = torch.from_numpy(np.stack([r[1] for r in recs])).to(device)
        lc = torch.from_numpy(np.stack([r[2] for r in recs])).to(device)
        return img, tc, lc


# ---------------------------------------------------------------------------
# Live projector — tile_idx -> (B, proj_dim), Prithvi+LoRA in the graph
# ---------------------------------------------------------------------------
class LivePrithviProjector(nn.Module):
    """forward(tile_idx) -> (B, proj_dim). Encodes each UNIQUE tile in the batch
    once (LoRA in graph), inline-LayerNorms the pooled 1024-vector, and projects
    1024 -> proj_dim. Missing tiles (tile_idx == zero_idx) -> zeros: those rows
    still train the TFT on their other features but contribute no Prithvi gradient.
    """

    def __init__(self, lora_model, tile_store, zero_idx, in_dim=1024,
                 proj_dim=32, dropout=0.2, pool="mean", encode_chunk=256,
                 amp=True, device="cuda"):
        super().__init__()
        self.encoder = lora_model                       # PEFT-wrapped PrithviMAE
        self.store = tile_store
        self.zero_idx = int(zero_idx)
        self.pool = pool
        self.encode_chunk = int(encode_chunk)
        self.amp = amp
        self.dev = device
        self.out_dim = proj_dim
        # Inline standardization (replaces the NDVI offline (mu, sd)).
        self.pre_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _encode_unique(self, uniq_idxs):
        outs = []
        for s in range(0, len(uniq_idxs), self.encode_chunk):
            chunk = uniq_idxs[s:s + self.encode_chunk]
            img, tc, lc = self.store.load_batch(chunk, self.dev)
            with amp_ctx(self.dev, self.amp):
                emb = encode_tokens(self.encoder, img, tc, lc, self.pool)
            outs.append(emb.float())
        return torch.cat(outs, 0)                        # (U, in_dim)

    def forward(self, tile_idx):
        w = self.proj[0].weight
        idx = tile_idx.detach().to("cpu").numpy().astype(np.int64)
        B = len(idx)
        out = torch.zeros(B, self.out_dim, device=w.device, dtype=w.dtype)

        present = idx != self.zero_idx
        if not present.any():
            return out
        pos = np.nonzero(present)[0]
        uniq, inv = np.unique(idx[pos], return_inverse=True)
        emb_u = self._encode_unique([int(x) for x in uniq])      # (U, in_dim)
        emb_u = self.pre_norm(emb_u)                              # inline standardize
        proj_u = self.proj(emb_u)                                # (U, proj_dim)
        scattered = proj_u[torch.as_tensor(inv, device=proj_u.device)]
        out[torch.as_tensor(pos, device=out.device)] = scattered.to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Tile-grouped batch sampler — few unique tiles per batch (fast encoding)
# ---------------------------------------------------------------------------
class TileGroupedBatchSampler(torch.utils.data.Sampler):
    """Each batch = n_tiles tiles x samples_per_tile samples, so a batch of
    n_tiles*samples_per_tile rows touches only n_tiles unique tiles. Reshuffles
    every epoch. Makes joint training ~ samples_per_tile x cheaper than per-sample
    random batching, at the cost of exact epoch composition.

    Missing-tile samples (tile_idx == zero_idx) are grouped under the zero key and
    still trained (they contribute a zero Prithvi block).
    """

    def __init__(self, tile_idx, n_tiles=256, samples_per_tile=32, seed=42):
        self.by_tile = {}
        for i, t in enumerate(np.asarray(tile_idx).astype(np.int64)):
            self.by_tile.setdefault(int(t), []).append(i)
        self.tiles = list(self.by_tile.keys())
        self.n_tiles = int(n_tiles)
        self.spt = int(samples_per_tile)
        self.seed = int(seed)
        self.epoch = 0
        self._len = max(1, len(tile_idx) // (self.n_tiles * self.spt))

    def __len__(self):
        return self._len

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        tiles = self.tiles.copy()
        rng.shuffle(tiles)
        for b in range(self._len):
            start = (b * self.n_tiles) % len(tiles)
            chosen = tiles[start:start + self.n_tiles]
            if len(chosen) < self.n_tiles:
                chosen = chosen + tiles[:self.n_tiles - len(chosen)]
            batch = []
            for t in chosen:
                pool = self.by_tile[t]
                sel = rng.choice(pool, size=self.spt, replace=len(pool) < self.spt)
                batch.extend(int(x) for x in sel)
            rng.shuffle(batch)
            yield batch


# ---------------------------------------------------------------------------
# Helpers for the training scripts
# ---------------------------------------------------------------------------
def load_latlon(station_csv):
    """Read gwl_stations.csv|.tsv -> {safe_id: (lat, lon)} for location coords."""
    import pandas as pd
    df = pd.read_csv(station_csv, sep=None, engine="python")   # sniff comma or tab (.csv / .tsv)
    need = {"safe_id", "lat", "lon"}
    if not need.issubset(df.columns):
        raise ValueError(f"{station_csv} must have columns {need}; got {list(df.columns)}")
    return {str(r.safe_id): (float(r.lat), float(r.lon))
            for r in df.itertuples()}


def build_projector(manifest, model_dir, station_csv, proj_dim=32, lora_r=16,
                    lora_alpha=32, dropout=0.2, device="cuda", amp=True,
                    encode_chunk=256, cache_size=20000, n_threads=8, exclude_ckpt=False):
    """Convenience: tile_manifest + Prithvi dir + station csv -> LivePrithviProjector.

    Wires load_prithvi_lora -> TileStore (band stats from config.json, latlon from
    station csv) -> LivePrithviProjector(zero_idx from manifest). ``in_dim`` is the
    encoder embed_dim read from the checkpoint config.
    """
    lora_model, cfg = load_prithvi_lora(
        model_dir, r=lora_r, alpha=lora_alpha, dropout=dropout,
        num_frames=1, device=device, exclude_ckpt=exclude_ckpt)
    band_mean, band_std = prithvi_band_stats(model_dir)
    store = TileStore(
        manifest["ordered_files"], manifest["composite_dir"],
        manifest["period"], manifest["period_doy"],
        load_latlon(station_csv), band_mean, band_std,
        cache_size=cache_size, n_threads=n_threads)
    projector = LivePrithviProjector(
        lora_model, store, manifest["zero_idx"],
        in_dim=int(cfg["embed_dim"]), proj_dim=proj_dim, dropout=dropout,
        pool="mean", encode_chunk=encode_chunk, amp=amp, device=device)
    return projector.to(device)


def split_param_groups(model, ft_lr, forecaster_lr_scale):
    """Two LR groups: Prithvi (LoRA + projector) at ft_lr; the rest of the model
    (the from-scratch TFT) at ft_lr * forecaster_lr_scale.

    Assumes the projector is attached to the TFT under attribute name ``prithvi``
    (so its params are named ``prithvi.*``), matching model.py Step 7.
    """
    prithvi, forecaster = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (prithvi if n.startswith("prithvi.") else forecaster).append(p)
    groups = [{"params": prithvi, "lr": ft_lr},
              {"params": forecaster, "lr": ft_lr * forecaster_lr_scale}]
    n_p = sum(p.numel() for p in prithvi)
    n_f = sum(p.numel() for p in forecaster)
    print(f"[ft] trainable: prithvi(LoRA+proj) {n_p:,} @ lr {ft_lr:g} | "
          f"forecaster {n_f:,} @ lr {ft_lr * forecaster_lr_scale:g}")
    return groups
