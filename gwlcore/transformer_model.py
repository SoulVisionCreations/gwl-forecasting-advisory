"""
GWL Forecasting Transformer Model

Drop-in replacement for GWLForecastLSTM — same forward() signature and output shape.

Architecture:
- Transformer Encoder: Processes sequences [num_timesteps, features] with self-attention
- Static Encoder: Same as LSTM version (embeddings + forecast conditioning)
- Fusion Layer: Combines temporal and static representations
- Regression Head: Single output (GWL delta or absolute)

No positional encoding — relies on existing sin/cos month features in the input.
Can be added later via learnable embeddings if needed.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .model import LinearResidualHead


class GWLForecastTransformer(nn.Module):
    """
    Transformer encoder for GWL forecasting.

    Same interface as GWLForecastLSTM: identical forward() args and return shape.

    Args:
        num_timesteps: Number of timesteps in sequence (default: 10)
        features_per_timestep: Features per timestep (default: 10)
        d_model: Transformer internal dimension (default: 64)
        nhead: Number of attention heads (default: 4)
        num_encoder_layers: Number of TransformerEncoder layers (default: 2)
        dim_feedforward: Feedforward dimension in transformer (default: 128)
        transformer_dropout: Dropout in transformer layers (default: 0.2)
        static_embedding_dim: Output dim for static encoder (default: 32)
        fusion_hidden_dim: Hidden dim after fusion (default: 64)
        num_lithology_types, num_well_types, etc.: Same as LSTM version
        num_forecast_features: Number of forecast features (default: 6)
        dropout_rate: Dropout for FC layers (default: 0.2)
    """

    def __init__(
        self,
        num_timesteps: int = 10,
        features_per_timestep: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        transformer_dropout: float = 0.2,
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
        use_linear_residual: bool = False,
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        self.d_model = d_model
        self.num_forecast_features = num_forecast_features
        self.dropout_rate = dropout_rate
        self.use_static_features = use_static_features
        self.use_linear_residual = use_linear_residual

        # Optional additive linear baseline (off by default — backward compatible).
        self.linear_residual_head = (
            LinearResidualHead() if use_linear_residual else None
        )

        # ====================================================================
        # 1. INPUT PROJECTION + TRANSFORMER ENCODER
        # ====================================================================
        # Project per-timestep features to d_model
        input_dim = features_per_timestep + lulc_embedding_dim
        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # ====================================================================
        # 2. STATIC + FORECAST ENCODER
        # ====================================================================
        # LULC always created — used in input sequence
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
                lithology_embedding_dim
                + well_type_embedding_dim
                + aquifer_embedding_dim
                + aquifer_0_aquifer_embedding_dim
                + litho_supergroup_embedding_dim
                + state_embedding_dim
                + district_embedding_dim
                + lulc_embedding_dim  # forecast LULC also goes through static encoder
            )
            # 5 base (well_depth, elevation, stream_order, lat, lon)
            # + 7 derived (gwl_anomaly, mean_annual_rainfall, annual_rainfall_std,
            # station_mean_gwl, station_gwl_amplitude, station_delta_mean,
            # station_delta_std) = 12
            static_continuous_dim = 12
        else:
            total_embedding_dim = lulc_embedding_dim  # only forecast LULC
            static_continuous_dim = 1  # only well_depth

        conditioning_input_dim = (
            static_continuous_dim
            + total_embedding_dim
            + num_forecast_features
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(conditioning_input_dim, static_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ====================================================================
        # 3. FUSION LAYER
        # ====================================================================
        # Transformer output: mean-pool over timesteps -> d_model
        fusion_input_dim = d_model + static_embedding_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ====================================================================
        # 4. REGRESSION HEAD
        # ====================================================================
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Restore zero init for the linear residual head (Xavier above clobbered it).
        if getattr(self, "linear_residual_head", None) is not None:
            self.linear_residual_head.reset_to_zero()

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
    ) -> torch.Tensor:
        """
        Forward pass — same signature and return shape as GWLForecastLSTM.

        Returns:
            gwl_pred: shape (batch_size, 1)
        """
        # Step 1: Embed historical LULC and concat with sequence
        historical_lulc_emb = self.lulc_embedding(historical_lulc_indices)  # [B, T, lulc_dim]
        transformer_input = torch.cat([sequence, historical_lulc_emb], dim=2)  # [B, T, input_dim]

        # Project to d_model and run through transformer
        transformer_input = self.input_projection(transformer_input)  # [B, T, d_model]
        transformer_out = self.transformer_encoder(transformer_input)  # [B, T, d_model]

        # Mean-pool over timesteps
        temporal_repr = transformer_out.mean(dim=1)  # [B, d_model]

        # Step 2: Static + forecast conditioning
        forecast_lulc_emb = self.lulc_embedding(forecast_lulc_idx)

        if self.use_static_features:
            lithology_emb = self.lithology_embedding(lithology_idx)
            well_type_emb = self.well_type_embedding(well_type_idx)
            aquifer_emb = self.aquifer_embedding(aquifer_idx)
            aquifer_0_aquifer_emb = self.aquifer_0_aquifer_embedding(aquifer_0_aquifer_idx)
            litho_supergroup_emb = self.litho_supergroup_embedding(litho_supergroup_idx)
            state_emb = self.state_embedding(state_idx)
            district_emb = self.district_embedding(district_idx)

            conditioning_features = torch.cat([
                static_continuous,        # [B, 5]
                lithology_emb,
                well_type_emb,
                aquifer_emb,
                aquifer_0_aquifer_emb,
                litho_supergroup_emb,
                state_emb,
                district_emb,
                forecast_lulc_emb,
                forecast_features,
            ], dim=1)
        else:
            # Stripped: only well_depth (index 0) + forecast LULC + forecast features
            well_depth = static_continuous[:, 0:1]
            conditioning_features = torch.cat([
                well_depth,
                forecast_lulc_emb,
                forecast_features,
            ], dim=1)

        static_repr = self.static_encoder(conditioning_features)

        # Step 3: Fuse
        combined = torch.cat([temporal_repr, static_repr], dim=1)
        fused = self.fusion(combined)

        # Step 4: Predict
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
            )
            trend_prob = torch.sigmoid(gwl_pred)
            trend_class = (trend_prob.squeeze(-1) >= 0.5).long()
            return {
                "gwl_pred": gwl_pred,
                "trend_prob": trend_prob,
                "trend_class": trend_class,
            }


class GWLForecastConditionedTransformer(nn.Module):
    """
    Conditioned Transformer for GWL forecasting — static features injected into Transformer.

    Reversed flow compared to GWLForecastTransformer:
    1. Static Encoder: continuous + categorical embeddings → MLP → static_encoding
    2. Conditioned Transformer: cat(sequence, lulc_emb, static_encoding) at each timestep
       → input_projection → TransformerEncoder → mean-pool → dynamic_encoding
    3. Forecast Fusion: cat(dynamic_encoding, forecast_lulc, forecast_features) → MLP → output

    Same forward() signature and output shape as GWLForecastLSTM / GWLForecastTransformer.
    """

    def __init__(
        self,
        num_timesteps: int = 10,
        features_per_timestep: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        transformer_dropout: float = 0.2,
        static_embedding_dim: int = 8,
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
        use_linear_residual: bool = False,
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        self.d_model = d_model
        self.num_forecast_features = num_forecast_features
        self.dropout_rate = dropout_rate
        self.use_static_features = use_static_features
        self.use_linear_residual = use_linear_residual

        # Optional additive linear baseline (off by default — backward compatible).
        self.linear_residual_head = (
            LinearResidualHead() if use_linear_residual else None
        )
        self.static_embedding_dim = static_embedding_dim
        self.lulc_embedding_dim = lulc_embedding_dim

        # ====================================================================
        # 1. STATIC ENCODER (processed BEFORE Transformer — no forecast features)
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
            # 5 base + 7 derived station stats (incl. delta_mean/std) = 12
            static_continuous_dim = 12
        else:
            total_embedding_dim = 0
            static_continuous_dim = 1

        static_input_dim = static_continuous_dim + total_embedding_dim
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, static_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # ====================================================================
        # 2. CONDITIONED TRANSFORMER (static_encoding injected at every timestep)
        # ====================================================================
        input_dim = features_per_timestep + lulc_embedding_dim + static_embedding_dim
        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # ====================================================================
        # 3. FORECAST FUSION (forecast features join AFTER Transformer)
        # ====================================================================
        forecast_input_dim = d_model + lulc_embedding_dim + num_forecast_features
        self.forecast_fusion = nn.Sequential(
            nn.Linear(forecast_input_dim, fusion_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # ====================================================================
        # 4. REGRESSION HEAD
        # ====================================================================
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Restore zero init for the linear residual head (Xavier above clobbered it).
        if getattr(self, "linear_residual_head", None) is not None:
            self.linear_residual_head.reset_to_zero()

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
    ) -> torch.Tensor:
        """
        Forward pass — same signature and return shape as GWLForecastLSTM.

        Returns:
            gwl_pred: shape (batch_size, 1)
        """
        # Step 1: Encode static features (BEFORE Transformer)
        if self.use_static_features:
            lithology_emb = self.lithology_embedding(lithology_idx)
            well_type_emb = self.well_type_embedding(well_type_idx)
            aquifer_emb = self.aquifer_embedding(aquifer_idx)
            aquifer_0_aquifer_emb = self.aquifer_0_aquifer_embedding(aquifer_0_aquifer_idx)
            litho_supergroup_emb = self.litho_supergroup_embedding(litho_supergroup_idx)
            state_emb = self.state_embedding(state_idx)
            district_emb = self.district_embedding(district_idx)

            static_input = torch.cat([
                static_continuous,
                lithology_emb,
                well_type_emb,
                aquifer_emb,
                aquifer_0_aquifer_emb,
                litho_supergroup_emb,
                state_emb,
                district_emb,
            ], dim=1)
        else:
            well_depth = static_continuous[:, 0:1]
            static_input = well_depth

        static_encoding = self.static_encoder(static_input)  # [batch, static_embedding_dim]

        # Step 2: Conditioned Transformer (static injected at every timestep)
        historical_lulc_emb = self.lulc_embedding(historical_lulc_indices)  # [B, T, lulc_dim]
        T = sequence.size(1)
        static_expanded = static_encoding.unsqueeze(1).expand(-1, T, -1)  # [B, T, S]
        transformer_input = torch.cat([sequence, historical_lulc_emb, static_expanded], dim=2)

        transformer_input = self.input_projection(transformer_input)  # [B, T, d_model]
        transformer_out = self.transformer_encoder(transformer_input)  # [B, T, d_model]
        dynamic_encoding = transformer_out.mean(dim=1)  # [B, d_model]

        # Step 3: Forecast fusion (forecast features join AFTER Transformer)
        forecast_lulc_emb = self.lulc_embedding(forecast_lulc_idx)  # [B, lulc_dim]
        forecast_input = torch.cat([
            dynamic_encoding,
            forecast_lulc_emb,
            forecast_features
        ], dim=1)
        fused = self.forecast_fusion(forecast_input)

        # Step 4: Regression head
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
            )
            trend_prob = torch.sigmoid(gwl_pred)
            trend_class = (trend_prob.squeeze(-1) >= 0.5).long()
            return {
                "gwl_pred": gwl_pred,
                "trend_prob": trend_prob,
                "trend_class": trend_class,
            }
