"""
GWL Forecasting Temporal Fusion Transformer (TFT)

Drop-in replacement for GWLForecastLSTM — same forward() signature and output shape.

Architecture (adapted from Lim et al. 2021):
1. Variable Selection Networks: Learn per-feature importance for observed inputs
2. Static Covariate Encoders: Process static features into context vectors
3. LSTM Encoder: Local temporal processing with static enrichment
4. Interpretable Multi-Head Attention: Temporal self-attention over enriched states
5. Gated Residual Connections: Throughout for gradient flow
6. Output: Single scalar (GWL delta or absolute)

Key differences from our plain LSTM/Transformer:
- Learns WHICH features matter (variable selection) instead of treating all equally
- Static features enrich temporal processing via gated skip connections (not just concat)
- Attention is interpretable — can inspect which timesteps the model focuses on
- Gated residual networks prevent gradient vanishing in deep architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .model import LinearResidualHead


class GatedLinearUnit(nn.Module):
    """GLU activation: splits input in half, applies sigmoid gate to one half."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """
    GRN: Core building block of TFT.

    Applies: LayerNorm(x + GLU(ELU(W1·x + W2·context + b)))

    Optional context vector conditions the transformation (static enrichment).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim > 0 else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if dims differ
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = self.skip_proj(x) if self.skip_proj is not None else x

        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = F.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)

        return self.layer_norm(residual + hidden)


class VariableSelectionNetwork(nn.Module):
    """
    VSN: Learns feature importance weights via softmax over per-feature GRNs.

    For each feature, a GRN produces a transformed representation.
    A separate GRN produces softmax weights across features.
    Output = weighted sum of transformed features.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        hidden_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.feature_dim = input_dim // num_features

        # Per-feature GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(self.feature_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_features)
        ])

        # Weight GRN: takes flattened input + optional context → softmax weights
        self.weight_grn = GatedResidualNetwork(
            input_dim, hidden_dim, num_features,
            context_dim=context_dim, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim] where input_dim = num_features * feature_dim
            context: [..., context_dim] optional static context

        Returns:
            [..., hidden_dim] weighted combination of transformed features
        """
        # Split input into per-feature chunks
        features = x.split(self.feature_dim, dim=-1)  # list of [..., feature_dim]

        # Transform each feature
        transformed = [grn(f) for grn, f in zip(self.feature_grns, features)]
        transformed = torch.stack(transformed, dim=-2)  # [..., num_features, hidden_dim]

        # Compute importance weights
        weights = self.weight_grn(x, context)  # [..., num_features]
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)  # [..., num_features, 1]

        # Weighted sum
        return (transformed * weights).sum(dim=-2)  # [..., hidden_dim]


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable multi-head attention from TFT paper.

    Key difference from standard MHA: uses shared value weights across heads
    so attention weights are directly interpretable as feature importance.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)  # shared values
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, T, _ = query.shape

        Q = self.W_q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, heads, T, d_k]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.W_out(attn_output)


class GWLForecastTFT(nn.Module):
    """
    Temporal Fusion Transformer for GWL forecasting.

    Same forward() signature and output shape as GWLForecastLSTM.

    TFT-specific hyperparams:
        d_model: Internal dimension for GRNs and attention (default: 64)
        n_heads: Number of attention heads (default: 4)
        lstm_layers: LSTM layers in temporal encoder (default: 1)
        tft_dropout: Dropout throughout TFT (default: 0.1)
    """

    def __init__(
        self,
        # Shared kwargs (same as other models)
        num_timesteps: int = 10,
        features_per_timestep: int = 12,
        static_embedding_dim: int = 32,
        fusion_hidden_dim: int = 64,
        num_lithology_types: int = 8,
        num_well_types: int = 5,
        num_aquifer_types: int = 3,
        num_aquifer_0_aquifer_types: int = 5,
        num_litho_supergroup_types: int = 8,
        num_lulc_types: int = 10,
        num_state_types: int = 30,
        num_district_types: int = 700,
        lithology_embedding_dim: int = 4,
        well_type_embedding_dim: int = 3,
        aquifer_embedding_dim: int = 2,
        aquifer_0_aquifer_embedding_dim: int = 3,
        litho_supergroup_embedding_dim: int = 3,
        lulc_embedding_dim: int = 3,
        state_embedding_dim: int = 4,
        district_embedding_dim: int = 5,
        num_forecast_features: int = 6,
        dropout_rate: float = 0.2,
        use_static_features: bool = True,
        # TFT-specific
        d_model: int = 64,
        n_heads: int = 4,
        lstm_layers: int = 1,
        tft_dropout: float = 0.1,
        use_linear_residual: bool = False,
        # Prithvi-EO satellite features (Step 7). The projector is built in
        # train.py (prithvi_finetune.build_projector) and passed in; its output is
        # concatenated onto static_continuous so it enriches all 4 static contexts.
        use_prithvi: bool = False,
        prithvi_proj_dim: int = 32,
        prithvi_projector: nn.Module = None,
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        self.d_model = d_model
        self.use_static_features = use_static_features
        self.lulc_embedding_dim = lulc_embedding_dim
        self.num_forecast_features = num_forecast_features
        self.use_linear_residual = use_linear_residual
        self.use_prithvi = use_prithvi
        self.prithvi_proj_dim = prithvi_proj_dim if use_prithvi else 0
        if use_prithvi and not use_static_features:
            raise ValueError(
                "use_prithvi=True requires use_static_features=True "
                "(the Prithvi vector is concatenated into the static block)."
            )

        # Optional additive linear baseline (off by default — backward compatible).
        self.linear_residual_head = (
            LinearResidualHead() if use_linear_residual else None
        )

        # ====================================================================
        # 1. CATEGORICAL EMBEDDINGS
        # ====================================================================
        self.lulc_embedding = nn.Embedding(num_lulc_types, lulc_embedding_dim)

        if use_static_features:
            self.lithology_embedding = nn.Embedding(num_lithology_types, lithology_embedding_dim)
            self.well_type_embedding = nn.Embedding(num_well_types, well_type_embedding_dim)
            self.aquifer_embedding = nn.Embedding(num_aquifer_types, aquifer_embedding_dim)
            self.aquifer_0_aquifer_embedding = nn.Embedding(num_aquifer_0_aquifer_types, aquifer_0_aquifer_embedding_dim)
            self.litho_supergroup_embedding = nn.Embedding(num_litho_supergroup_types, litho_supergroup_embedding_dim)
            self.state_embedding = nn.Embedding(num_state_types, state_embedding_dim)
            self.district_embedding = nn.Embedding(num_district_types, district_embedding_dim)

            total_embedding_dim = (
                lithology_embedding_dim +
                well_type_embedding_dim +
                aquifer_embedding_dim +
                aquifer_0_aquifer_embedding_dim +
                litho_supergroup_embedding_dim +
                state_embedding_dim +
                district_embedding_dim
            )
            # 5 base (well_depth, elevation, stream_order, lat, lon)
            # + 7 derived (gwl_anomaly, mean_annual_rainfall, annual_rainfall_std,
            # station_mean_gwl, station_gwl_amplitude, station_delta_mean,
            # station_delta_std) = 12
            static_continuous_dim = 12
            # Prithvi proj vector is appended at the END of static_continuous
            # (indices 12..12+proj_dim-1), so existing indices are untouched.
            static_continuous_dim += self.prithvi_proj_dim
        else:
            total_embedding_dim = 0
            static_continuous_dim = 1  # only well_depth

        static_input_dim = static_continuous_dim + total_embedding_dim

        # ====================================================================
        # 2. STATIC COVARIATE ENCODERS (4 context vectors from TFT paper)
        # ====================================================================
        # TFT uses 4 static context vectors for different purposes:
        # c_s: context for temporal variable selection
        # c_e: context for static enrichment
        # c_h: initial hidden state for LSTM
        # c_c: initial cell state for LSTM
        self.static_context_vsn = GatedResidualNetwork(
            static_input_dim, d_model, d_model, dropout=tft_dropout
        )
        self.static_context_enrichment = GatedResidualNetwork(
            static_input_dim, d_model, d_model, dropout=tft_dropout
        )
        self.static_context_hidden = GatedResidualNetwork(
            static_input_dim, d_model, d_model, dropout=tft_dropout
        )
        self.static_context_cell = GatedResidualNetwork(
            static_input_dim, d_model, d_model, dropout=tft_dropout
        )

        # ====================================================================
        # 3. TEMPORAL INPUT PROCESSING
        # ====================================================================
        # Per-timestep input: sequence features + LULC embedding
        temporal_input_dim = features_per_timestep + lulc_embedding_dim

        # Project each observed feature to d_model individually for VSN
        # We treat each of the temporal_input_dim features as a separate "variable"
        self.num_temporal_features = temporal_input_dim
        self.temporal_feature_proj = nn.Linear(temporal_input_dim, d_model)

        # Variable selection for temporal features
        self.temporal_vsn = VariableSelectionNetwork(
            input_dim=temporal_input_dim,
            num_features=temporal_input_dim,
            hidden_dim=d_model,
            context_dim=d_model,  # static context
            dropout=tft_dropout,
        )

        # ====================================================================
        # 4. LSTM ENCODER (local temporal processing)
        # ====================================================================
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=tft_dropout if lstm_layers > 1 else 0,
        )

        # Gate + LayerNorm after LSTM (skip connection from VSN output)
        self.post_lstm_glu = GatedLinearUnit(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # ====================================================================
        # 5. STATIC ENRICHMENT
        # ====================================================================
        self.static_enrichment = GatedResidualNetwork(
            d_model, d_model, d_model,
            context_dim=d_model,  # c_e
            dropout=tft_dropout,
        )

        # ====================================================================
        # 6. INTERPRETABLE MULTI-HEAD ATTENTION
        # ====================================================================
        self.attention = InterpretableMultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=tft_dropout,
        )
        self.post_attn_glu = GatedLinearUnit(d_model, d_model)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # ====================================================================
        # 7. POSITIONWISE FEEDFORWARD
        # ====================================================================
        self.positionwise_grn = GatedResidualNetwork(
            d_model, d_model, d_model, dropout=tft_dropout,
        )

        # ====================================================================
        # 8. FORECAST FUSION + OUTPUT HEAD
        # ====================================================================
        # Combine temporal output with forecast features
        forecast_input_dim = d_model + lulc_embedding_dim + num_forecast_features
        self.forecast_fusion = nn.Sequential(
            nn.Linear(forecast_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(tft_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(tft_dropout),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(tft_dropout),
            nn.Linear(32, 1),
        )

        self._initialize_weights()

        # Attach the Prithvi projector AFTER _initialize_weights so its pretrained
        # encoder / zero-init LoRA / projection head are never Xavier-clobbered.
        # Named `prithvi` so its params are `prithvi.*` — the contract that
        # prithvi_finetune.split_param_groups uses to give it its own LR group.
        self.prithvi = prithvi_projector if use_prithvi else None

    def _initialize_weights(self):
        for name, module in self.named_modules():
            # Never touch the Prithvi subtree (pretrained encoder + LoRA + proj);
            # belt-and-suspenders even though it's attached after this runs.
            if name.startswith("prithvi"):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Restore zero init for the linear residual head (Xavier above clobbered it).
        if getattr(self, "linear_residual_head", None) is not None:
            self.linear_residual_head.reset_to_zero()

    def _encode_static(
        self,
        static_continuous: torch.Tensor,
        lithology_idx: torch.Tensor,
        well_type_idx: torch.Tensor,
        aquifer_idx: torch.Tensor,
        aquifer_0_aquifer_idx: torch.Tensor,
        litho_supergroup_idx: torch.Tensor,
        state_idx: torch.Tensor,
        district_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Encode static features into a flat vector."""
        if self.use_static_features:
            parts = [
                static_continuous,
                self.lithology_embedding(lithology_idx),
                self.well_type_embedding(well_type_idx),
                self.aquifer_embedding(aquifer_idx),
                self.aquifer_0_aquifer_embedding(aquifer_0_aquifer_idx),
                self.litho_supergroup_embedding(litho_supergroup_idx),
                self.state_embedding(state_idx),
                self.district_embedding(district_idx),
            ]
            return torch.cat(parts, dim=1)
        else:
            return static_continuous[:, 0:1]  # well_depth only

    def forward(
        self,
        sequence: torch.Tensor,
        static_continuous: torch.Tensor,
        forecast_features: torch.Tensor,
        lithology_idx: torch.Tensor,
        well_type_idx: torch.Tensor,
        aquifer_idx: torch.Tensor,
        aquifer_0_aquifer_idx: torch.Tensor,
        litho_supergroup_idx: torch.Tensor,
        state_idx: torch.Tensor,
        district_idx: torch.Tensor,
        historical_lulc_indices: torch.Tensor,
        forecast_lulc_idx: torch.Tensor,
        tile_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass — same signature and return shape as GWLForecastLSTM, plus an
        optional `tile_idx` (only when use_prithvi=True) selecting each sample's
        HLS composite tile for the live Prithvi encoder.

        Returns:
            gwl_pred: shape (batch_size, 1)
        """
        B, T, _ = sequence.shape

        # ── Step 0: Prithvi satellite features ────────────────────────────
        # Encode each sample's tile (LoRA in graph) → [B, proj_dim] and append to
        # static_continuous, so it feeds all 4 static context vectors below.
        # zero_idx tiles emit zeros inside the projector (no Prithvi gradient).
        if self.use_prithvi and self.prithvi is not None:
            if tile_idx is None:
                raise ValueError(
                    "use_prithvi=True but tile_idx was not passed to forward()."
                )
            prithvi_feat = self.prithvi(tile_idx)              # [B, proj_dim]
            static_continuous = torch.cat(
                [static_continuous, prithvi_feat.to(static_continuous.dtype)], dim=1
            )                                                  # [B, 12 + proj_dim]

        # ── Step 1: Static covariate encoding ─────────────────────────────
        static_input = self._encode_static(
            static_continuous, lithology_idx, well_type_idx, aquifer_idx,
            aquifer_0_aquifer_idx, litho_supergroup_idx, state_idx, district_idx,
        )

        # 4 context vectors from static features
        c_vsn = self.static_context_vsn(static_input)         # [B, d_model]
        c_enrich = self.static_context_enrichment(static_input)  # [B, d_model]
        c_h = self.static_context_hidden(static_input)         # [B, d_model]
        c_c = self.static_context_cell(static_input)           # [B, d_model]

        # ── Step 2: Temporal input with LULC ──────────────────────────────
        historical_lulc_emb = self.lulc_embedding(historical_lulc_indices)  # [B, T, lulc_dim]
        temporal_input = torch.cat([sequence, historical_lulc_emb], dim=2)  # [B, T, feat+lulc]

        # ── Step 3: Variable selection (per timestep) ─────────────────────
        # Expand static context for each timestep
        c_vsn_expanded = c_vsn.unsqueeze(1).expand(-1, T, -1)  # [B, T, d_model]

        # Reshape for VSN: process each timestep independently
        flat_input = temporal_input.reshape(B * T, -1)
        flat_context = c_vsn_expanded.reshape(B * T, -1)
        selected = self.temporal_vsn(flat_input, flat_context)  # [B*T, d_model]
        selected = selected.view(B, T, self.d_model)  # [B, T, d_model]

        # ── Step 4: LSTM encoder ──────────────────────────────────────────
        # Use static context as initial hidden/cell states
        h0 = c_h.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = c_c.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()

        lstm_out, _ = self.lstm(selected, (h0, c0))  # [B, T, d_model]

        # Gated skip connection: LSTM output + VSN output
        gated_lstm = self.post_lstm_glu(lstm_out)
        temporal_features = self.post_lstm_norm(selected + gated_lstm)  # [B, T, d_model]

        # ── Step 5: Static enrichment ─────────────────────────────────────
        c_enrich_expanded = c_enrich.unsqueeze(1).expand(-1, T, -1)
        flat_temporal = temporal_features.reshape(B * T, -1)
        flat_enrich = c_enrich_expanded.reshape(B * T, -1)
        enriched = self.static_enrichment(flat_temporal, flat_enrich)
        enriched = enriched.view(B, T, self.d_model)  # [B, T, d_model]

        # ── Step 6: Interpretable multi-head attention ────────────────────
        attn_out = self.attention(enriched, enriched, enriched)  # [B, T, d_model]

        # Gated skip connection: attention + enriched
        gated_attn = self.post_attn_glu(attn_out)
        attn_features = self.post_attn_norm(enriched + gated_attn)  # [B, T, d_model]

        # ── Step 7: Positionwise feedforward ──────────────────────────────
        flat_attn = attn_features.reshape(B * T, -1)
        output_features = self.positionwise_grn(flat_attn)
        output_features = output_features.view(B, T, self.d_model)  # [B, T, d_model]

        # Use last timestep (most recent) for prediction
        temporal_repr = output_features[:, -1, :]  # [B, d_model]

        # ── Step 8: Forecast fusion + output ──────────────────────────────
        forecast_lulc_emb = self.lulc_embedding(forecast_lulc_idx)  # [B, lulc_dim]
        forecast_input = torch.cat([
            temporal_repr,
            forecast_lulc_emb,
            forecast_features,
        ], dim=1)
        fused = self.forecast_fusion(forecast_input)
        gwl_pred = self.regression_head(fused)

        # Optional: add linear-baseline residual path.
        if self.linear_residual_head is not None:
            gwl_pred = gwl_pred + self.linear_residual_head(
                sequence, static_continuous, forecast_features,
            )

        return gwl_pred

    def predict(
        self,
        sequence: torch.Tensor,
        static_continuous: torch.Tensor,
        forecast_features: torch.Tensor,
        lithology_idx: torch.Tensor,
        well_type_idx: torch.Tensor,
        aquifer_idx: torch.Tensor,
        aquifer_0_aquifer_idx: torch.Tensor,
        litho_supergroup_idx: torch.Tensor,
        state_idx: torch.Tensor,
        district_idx: torch.Tensor,
        historical_lulc_indices: torch.Tensor,
        forecast_lulc_idx: torch.Tensor,
        tile_idx: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Same predict interface as GWLForecastLSTM."""
        self.eval()
        with torch.no_grad():
            gwl_pred = self.forward(
                sequence, static_continuous, forecast_features,
                lithology_idx, well_type_idx, aquifer_idx,
                aquifer_0_aquifer_idx, litho_supergroup_idx,
                state_idx, district_idx,
                historical_lulc_indices, forecast_lulc_idx,
                tile_idx,
            )
            trend_prob = torch.sigmoid(gwl_pred)
            trend_class = (trend_prob.squeeze(-1) >= 0.5).long()
            return {
                "gwl_pred": gwl_pred,
                "trend_prob": trend_prob,
                "trend_class": trend_class,
            }
