"""
GWL Forecasting LSTM Model

Implementation of the LSTM architecture described in lstm_design.md.

Architecture:
- LSTM Encoder: Processes sequences [num_timesteps, 10] (historical patterns only)
- Static Encoder: Embeds static features + forecast features with nn.Embedding for categoricals
- Fusion Layer: Combines temporal and static representations
- Regression Head: Single output — predicts GWL delta (or absolute value)

Single-output, dual-loss design:
  The model has ONE output (predicted delta or absolute GWL). Two losses share it:
  - MSE loss: penalizes magnitude error (how far off is the prediction?)
  - BCE loss: penalizes sign error (did the model predict the right trend direction?)
  This eliminates the redundant trend head — the regression output's sign IS the trend.
  sigmoid(output) gives P(increase), and (output >= 0) gives the trend class.

GWL Modes:
- Delta mode (default): Output is predicted (target_gwl - current_gwl).
  Reconstruction: predicted_gwl = current_gwl + model_output.
- Absolute mode: Output is predicted raw GWL value.

Per-timestep features: [gwl_encoded, is_reliable, sin, cos, 6 historical] = 10
- gwl_encoded: delta from current_gwl (delta mode) or raw value (absolute mode)
- is_reliable: 1.0 if present AND not outlier, 0.0 otherwise

Key Features:
1. Unidirectional LSTM (causal - can't see future)
2. nn.Embedding for categorical features (lithology, well_type, aquifer)
3. Forecast features used as conditioning (NOT fed into LSTM)
4. Dual-loss on single output (regression + trend via BCE)
5. Dropout regularization

Reference: lstm_design.md (Model Architecture section)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class LinearResidualHead(nn.Module):
    """
    Optional linear baseline path: predicts a per-well baseline GWL delta from a
    fixed 14-dim slice of static + state + LOCAL trend + forecast features.
    Trains end-to-end alongside the main LSTM/TFT path; the additive structure
    causes the LSTM to implicitly learn the residual.

    Inputs (all already scaled by feature_scaler upstream):
      - static_continuous: [B, 12]   — uses 6 indices
          1:  elevation                 *
          2:  stream_order              *
          3:  latitude                  *
          5:  gwl_anomaly               *
          6:  mean_annual_rainfall      *
          10: station_delta_mean        *  (KEY: per-well long-term 3mo delta)
          11: station_delta_std         *  (gating / scaling signal)
      - sequence: [B, T, F_t] — uses
          - last timestep's sin_month, cos_month (indices 4-5)
          - last K=4 timesteps' gwl_diff (index 2) for local trend stats
          - last K=4 timesteps' gwl_diff_reliable (index 3) as quality mask
      - forecast_features: [B, F_f] — uses indices 0-1 (rain, temp)

    Total: 7 static (incl mean_annual_rainfall) + 2 month + 3 local trend +
           2 forecast = 14 features.

    Local trend features (computed on the fly from the sequence with a
    reliability mask; all fall back to 0 when insufficient reliable data):
      - last_observed_delta:  most recent gwl_diff value if reliable, else 0
      - recent_delta_mean:    weighted mean of gwl_diff over last K timesteps
      - recent_delta_slope:   weighted OLS slope of gwl_diff over last K timesteps

    Fallback rules (so the linear pred remains meaningful for sparse wells):
      - last_observed_delta: 0 if last timestep unreliable
      - recent_delta_mean:   0 if 0 reliable points in window
      - recent_delta_slope:  0 if < 2 reliable points in window
    When a local feature falls back to 0, the regression's weight on
    station_delta_mean takes over as the per-well baseline.

    Output: [B, 1] scalar prediction of scaled GWL delta. Summed with the
            LSTM/TFT output and trained jointly with MSE/Huber on the sum.

    Init: zero weights and bias so the linear path contributes nothing at
    step 0 and "earns" its weights during training.
    """

    INPUT_DIM = 14
    K_LOCAL = 4   # window size for recent_delta_mean / slope (last K timesteps)

    # Indices into static_continuous for the 7 long-term fields used by the head.
    # Order matters: this also defines the position of those features in the
    # 14-dim concatenated input. station_delta_mean (the key baseline) is at
    # the end of the static block so its weight is easy to find: index 6.
    _STATIC_IDX = [1, 2, 3, 5, 6, 11, 10]   # last is station_delta_mean
    _STATIC_LABELS = [
        "elevation", "stream_order", "lat",
        "gwl_anomaly", "mean_annual_rainfall",
        "station_delta_std", "station_delta_mean",
    ]

    # Per-timestep sequence indices.
    _SEQ_SIN_MONTH_IDX = 4
    _SEQ_COS_MONTH_IDX = 5
    _SEQ_GWL_DIFF_IDX = 2
    _SEQ_GWL_DIFF_RELIABLE_IDX = 3

    # Forecast indices (rain, temp).
    _FORECAST_IDX = [0, 1]

    # Full 14-dim input layout (for inspection / labelling):
    FEATURE_LABELS = (
        _STATIC_LABELS
        + ["sin_month", "cos_month"]
        + ["last_observed_delta", "recent_delta_mean", "recent_delta_slope"]
        + ["forecast_rainfall", "forecast_temp"]
    )

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(self.INPUT_DIM, 1)
        self.reset_to_zero()

    def reset_to_zero(self):
        """Re-apply zero init to the linear weights and bias.

        The parent model's _initialize_weights() iterates all nn.Linear modules
        and applies Xavier init — clobbering our zero init. Each model calls
        this method at the end of _initialize_weights() to restore zero init.
        """
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @classmethod
    def _compute_local_trend(
        cls,
        sequence: torch.Tensor,   # [B, T, F_t]
    ) -> torch.Tensor:
        """Compute (last_observed_delta, recent_delta_mean, recent_delta_slope) per sample.

        All three fall back to 0 when reliability is insufficient:
          - last_observed_delta: 0 if last timestep is unreliable
          - recent_delta_mean:   0 if no reliable timesteps in last K
          - recent_delta_slope:  0 if < 2 reliable timesteps in last K
        """
        K = cls.K_LOCAL
        deltas = sequence[:, -K:, cls._SEQ_GWL_DIFF_IDX]            # [B, K]
        reliable = sequence[:, -K:, cls._SEQ_GWL_DIFF_RELIABLE_IDX]  # [B, K], 0/1

        n_reliable = reliable.sum(dim=-1)  # [B]

        # last_observed_delta: most recent reliable delta, else 0
        last_delta = deltas[:, -1] * reliable[:, -1]   # [B]

        # recent_delta_mean: weighted mean (denominator = n_reliable)
        w_sum = n_reliable.clamp(min=1.0)              # avoid /0; result will be masked below
        recent_mean = (deltas * reliable).sum(dim=-1) / w_sum
        recent_mean = torch.where(
            n_reliable >= 1, recent_mean, torch.zeros_like(recent_mean)
        )

        # recent_delta_slope: weighted OLS slope on last K reliable points
        x = torch.arange(K, dtype=deltas.dtype, device=deltas.device)  # [K]
        x = x.unsqueeze(0).expand_as(deltas)                            # [B, K]
        mean_x = (x * reliable).sum(dim=-1) / w_sum                     # [B]
        cx = x - mean_x.unsqueeze(-1)                                   # [B, K]
        cy = deltas - recent_mean.unsqueeze(-1)                         # [B, K]
        num = (cx * cy * reliable).sum(dim=-1)                          # [B]
        denom = (cx * cx * reliable).sum(dim=-1).clamp(min=1e-6)        # [B]
        slope = num / denom                                             # [B]
        slope = torch.where(
            n_reliable >= 2, slope, torch.zeros_like(slope)
        )

        return last_delta, recent_mean, slope

    def forward(
        self,
        sequence: torch.Tensor,
        static_continuous: torch.Tensor,
        forecast_features: torch.Tensor,
    ) -> torch.Tensor:
        # 7 long-term static features (last is station_delta_mean)
        static_part = static_continuous[:, self._STATIC_IDX]            # [B, 7]
        # sin/cos month from last timestep of the sequence
        seq_month = sequence[:, -1, [self._SEQ_SIN_MONTH_IDX, self._SEQ_COS_MONTH_IDX]]  # [B, 2]
        # Local recent dynamics with reliability-aware fallback
        last_delta, recent_mean, slope = self._compute_local_trend(sequence)
        local_part = torch.stack([last_delta, recent_mean, slope], dim=-1)  # [B, 3]
        # Forecast (rain, temp)
        fc_part = forecast_features[:, self._FORECAST_IDX[0]:self._FORECAST_IDX[-1] + 1]  # [B, 2]

        feats = torch.cat([static_part, seq_month, local_part, fc_part], dim=-1)  # [B, 14]
        return self.linear(feats)  # [B, 1]


class GWLForecastLSTM(nn.Module):
    """
    LSTM for GWL forecasting with embeddings for categorical features.

    Single output: predicted GWL value (delta or absolute). Trend direction is
    derived from the sign of the output — no separate trend head needed.

    The LSTM encoder processes ONLY historical patterns. Forecast features are used
    as conditioning along with static features, NOT fed into the LSTM. This allows
    the LSTM to learn pure temporal patterns from history, while forecast values
    condition the final prediction.

    Args:
        num_timesteps: Number of timesteps in sequence (default: 10)
        features_per_timestep: Features per timestep (default: 10)
            [gwl_encoded, is_reliable, sin, cos, 6 historical]
        lstm_hidden_dim: LSTM hidden state dimension (default: 128)
        lstm_num_layers: Number of stacked LSTM layers (default: 2)
        lstm_dropout: Dropout between LSTM layers (default: 0.2)
        static_embedding_dim: Output dimension for static encoder (default: 32)
        fusion_hidden_dim: Hidden dimension after fusion (default: 64)
        num_lithology_types: Number of unique lithology categories
        num_well_types: Number of unique well types
        num_aquifer_types: Number of unique aquifer types
        num_aquifer_0_aquifer_types: Number of unique aquifer_0_aquifer categories
        num_litho_supergroup_types: Number of unique litho_supergroup categories
        lithology_embedding_dim: Embedding dimension for lithology (default: 4)
        well_type_embedding_dim: Embedding dimension for well type (default: 3)
        aquifer_embedding_dim: Embedding dimension for aquifer (default: 2)
        aquifer_0_aquifer_embedding_dim: Embedding dimension for aquifer_0_aquifer (default: 3)
        litho_supergroup_embedding_dim: Embedding dimension for litho_supergroup (default: 3)
        num_forecast_features: Number of forecast features (default: 6)
        dropout_rate: Dropout probability for FC layers (default: 0.2)

    Reference: lstm_design.md (Model Architecture Details)
    """

    def __init__(
        self,
        num_timesteps: int = 10,
        features_per_timestep: int = 10,  # Changed to 10 for gwl_is_present flag
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
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
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_forecast_features = num_forecast_features
        self.dropout_rate = dropout_rate
        self.use_static_features = use_static_features
        self.use_linear_residual = use_linear_residual

        # Optional additive linear baseline path (residual learning style).
        # When enabled, predicts a per-well baseline from a 13-dim slice of
        # static + state + forecast features. The LSTM then implicitly learns
        # the residual via end-to-end joint training. Off by default for
        # backward compatibility.
        self.linear_residual_head = (
            LinearResidualHead() if use_linear_residual else None
        )

        # ====================================================================
        # 1. LSTM ENCODER
        # Reference: lstm_design.md (LSTM Encoder section)
        # ====================================================================
        lstm_input_dim = features_per_timestep + lulc_embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=False  # Causal - can't see future
        )

        # ====================================================================
        # 2. STATIC + FORECAST ENCODER
        # Reference: lstm_design.md (Static Encoder section)
        # ====================================================================
        # LULC embedding is ALWAYS created — it's used in the LSTM input sequence
        self.lulc_embedding = nn.Embedding(
            num_embeddings=num_lulc_types,
            embedding_dim=lulc_embedding_dim
        )

        # Other embeddings only created when use_static_features=True
        if use_static_features:
            self.lithology_embedding = nn.Embedding(
                num_embeddings=num_lithology_types,
                embedding_dim=lithology_embedding_dim
            )
            self.well_type_embedding = nn.Embedding(
                num_embeddings=num_well_types,
                embedding_dim=well_type_embedding_dim
            )
            self.aquifer_embedding = nn.Embedding(
                num_embeddings=num_aquifer_types,
                embedding_dim=aquifer_embedding_dim
            )
            self.aquifer_0_aquifer_embedding = nn.Embedding(
                num_embeddings=num_aquifer_0_aquifer_types,
                embedding_dim=aquifer_0_aquifer_embedding_dim
            )
            self.litho_supergroup_embedding = nn.Embedding(
                num_embeddings=num_litho_supergroup_types,
                embedding_dim=litho_supergroup_embedding_dim
            )
            self.state_embedding = nn.Embedding(
                num_embeddings=num_state_types,
                embedding_dim=state_embedding_dim
            )
            self.district_embedding = nn.Embedding(
                num_embeddings=num_district_types,
                embedding_dim=district_embedding_dim
            )

            # Total embedding dimensions for conditioning (excluding LULC which is in LSTM)
            total_embedding_dim = (
                lithology_embedding_dim +
                well_type_embedding_dim +
                aquifer_embedding_dim +
                aquifer_0_aquifer_embedding_dim +
                litho_supergroup_embedding_dim +
                state_embedding_dim +
                district_embedding_dim +
                lulc_embedding_dim  # forecast LULC also goes through static encoder
            )
            # Full static continuous: 5 base (well_depth, elevation, stream_order, lat, lon)
            # + 7 derived (gwl_anomaly, mean_annual_rainfall, annual_rainfall_std,
            # station_mean_gwl, station_gwl_amplitude, station_delta_mean,
            # station_delta_std) = 12
            static_continuous_dim = 12
        else:
            total_embedding_dim = lulc_embedding_dim  # only forecast LULC
            # Stripped: only well_depth (1)
            static_continuous_dim = 1

        # Total conditioning input: static continuous + embeddings + forecast features
        conditioning_input_dim = (
            static_continuous_dim +
            total_embedding_dim +
            num_forecast_features  # 6 forecast features for conditioning
        )

        # FC layer for static + forecast features (conditioning path)
        self.static_encoder = nn.Sequential(
            nn.Linear(conditioning_input_dim, static_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # ====================================================================
        # 3. FUSION LAYER
        # Reference: lstm_design.md (Fusion Layer section)
        # ====================================================================
        fusion_input_dim = lstm_hidden_dim + static_embedding_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # ====================================================================
        # 4. REGRESSION HEAD (single output — also used for trend via sign)
        # In delta mode: output is predicted Δ(GWL). Sign = trend direction.
        # Two losses share this output: MSE (magnitude) + BCE (sign/trend).
        # ====================================================================
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)  # Single value: GWL delta or absolute
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization for linear layers.
        LSTM weights are initialized by PyTorch with orthogonal initialization.
        Embeddings are initialized by PyTorch with N(0, 1).
        """
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
        forecast_lulc_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            sequence: Tensor of shape (batch_size, num_timesteps, features_per_timestep)
                Contains: [gwl_encoded, is_reliable, sin, cos, 6 historical] = 10 features
                gwl_encoded: delta from current_gwl (delta mode) or raw value (absolute mode)
                is_reliable: 1.0 if present AND not outlier, 0.0 otherwise

            static_continuous: Tensor of shape (batch_size, 5)
                Contains: [well_depth, elevation, stream_order, latitude, longitude]

            forecast_features: Tensor of shape (batch_size, 6)
                Contains: [rainfall, temp, et, runoff, ndvi, sm] climatological forecasts
                Used as conditioning for the prediction, NOT fed into LSTM

            lithology_idx: Tensor of shape (batch_size,) with lithology indices
            well_type_idx: Tensor of shape (batch_size,) with well type indices
            aquifer_idx: Tensor of shape (batch_size,) with aquifer type indices
            aquifer_0_aquifer_idx: Tensor of shape (batch_size,) with aquifer_0_aquifer indices
            litho_supergroup_idx: Tensor of shape (batch_size,) with litho_supergroup indices
            state_idx: Tensor of shape (batch_size,) with state indices
            district_idx: Tensor of shape (batch_size,) with district indices
            historical_lulc_indices: Tensor of shape (batch_size, num_timesteps) with LULC indices
            forecast_lulc_idx: Tensor of shape (batch_size,) with forecast LULC index

        Returns:
            gwl_pred: Predicted GWL delta or absolute value, shape (batch_size, 1)
                In delta mode, sign indicates trend: >0 = increase, <0 = decrease

        Reference: lstm_design.md (Architecture Flow)
        """
        batch_size = sequence.size(0)

        # ====================================================================
        # Step 1: Process sequence with LSTM (historical patterns ONLY)
        # Reference: lstm_design.md (LSTM Encoder)
        # ====================================================================
        # Embed historical LULC and concatenate with sequence
        historical_lulc_emb = self.lulc_embedding(historical_lulc_indices) # [batch, timesteps, lulc_emb_dim]
        lstm_input = torch.cat([sequence, historical_lulc_emb], dim=2)

        # lstm_out: [batch, num_timesteps, lstm_hidden_dim]
        # h_n: [num_layers, batch, lstm_hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        # Take the last hidden state from the last layer
        # Shape: [batch, lstm_hidden_dim]
        temporal_repr = h_n[-1]

        # ====================================================================
        # Step 2: Encode static + forecast features (conditioning path)
        # Reference: lstm_design.md (Static Encoder)
        # ====================================================================
        forecast_lulc_emb = self.lulc_embedding(forecast_lulc_idx)  # [batch, lulc_emb_dim]

        if self.use_static_features:
            # Full conditioning: all static features + all embeddings + forecast
            lithology_emb = self.lithology_embedding(lithology_idx)
            well_type_emb = self.well_type_embedding(well_type_idx)
            aquifer_emb = self.aquifer_embedding(aquifer_idx)
            aquifer_0_aquifer_emb = self.aquifer_0_aquifer_embedding(aquifer_0_aquifer_idx)
            litho_supergroup_emb = self.litho_supergroup_embedding(litho_supergroup_idx)
            state_emb = self.state_embedding(state_idx)
            district_emb = self.district_embedding(district_idx)

            conditioning_features = torch.cat([
                static_continuous,        # [batch, 5] — well_depth, elev, so, lat, lon
                lithology_emb,
                well_type_emb,
                aquifer_emb,
                aquifer_0_aquifer_emb,
                litho_supergroup_emb,
                state_emb,
                district_emb,
                forecast_lulc_emb,
                forecast_features         # [batch, 6]
            ], dim=1)
        else:
            # Stripped: only well_depth (index 0 of static_continuous) + forecast LULC + forecast features
            well_depth = static_continuous[:, 0:1]  # [batch, 1]
            conditioning_features = torch.cat([
                well_depth,
                forecast_lulc_emb,
                forecast_features
            ], dim=1)

        # Encode conditioning features
        static_repr = self.static_encoder(conditioning_features)  # [batch, 32]

        # ====================================================================
        # Step 3: Fuse temporal and conditioning representations
        # Reference: lstm_design.md (Fusion Layer)
        # ====================================================================
        combined = torch.cat([temporal_repr, static_repr], dim=1)  # [batch, 128 + 32]
        fused = self.fusion(combined)  # [batch, fusion_hidden_dim]

        # ====================================================================
        # Step 4: Single regression head (shared for value + trend)
        # Output is predicted delta (delta mode) or absolute GWL.
        # Sign of output = trend direction (used by BCE trend loss).
        # ====================================================================
        gwl_pred = self.regression_head(fused)     # [batch, 1]

        # Optional: add linear-baseline residual path (joint additive output).
        # Both heads predict in scaled space; sum is also scaled.
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
        forecast_lulc_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with post-processing for interpretability.

        Trend is derived from the regression output sign. This is only meaningful
        in delta mode (where the output IS a delta). In absolute mode, the caller
        should compute delta = gwl_pred - current_gwl externally for trend.

        Args:
            Same as forward()

        Returns:
            Dictionary containing:
                - gwl_pred: Predicted GWL delta or absolute value, shape (batch_size, 1)
                - trend_prob: P(increase) = sigmoid(gwl_pred), shape (batch_size, 1)
                    Only meaningful in delta mode. In absolute mode, compute
                    sigmoid(gwl_pred - current_gwl) externally instead.
                - trend_class: 1 if increase/stable, 0 if decrease, shape (batch_size,)
                    Only meaningful in delta mode.

        Note: In delta mode, gwl_pred is a delta. Reconstruct absolute GWL as:
              predicted_gwl = current_gwl + gwl_pred

        Reference: lstm_design.md (Model Output section)
        """
        self.eval()
        with torch.no_grad():
            gwl_pred = self.forward(
                sequence,
                static_continuous,
                forecast_features,
                lithology_idx,
                well_type_idx,
                aquifer_idx,
                aquifer_0_aquifer_idx,
                litho_supergroup_idx,
                state_idx,
                district_idx,
                historical_lulc_indices,
                forecast_lulc_idx
            )

            # Trend derived from sign of regression output
            # sigmoid(pred): >0.5 if pred>0 (increase), <0.5 if pred<0 (decrease)
            trend_prob = torch.sigmoid(gwl_pred)
            trend_class = (trend_prob.squeeze(-1) >= 0.5).long()

            return {
                'gwl_pred': gwl_pred,
                'trend_prob': trend_prob,
                'trend_class': trend_class
            }


class GWLForecastConditionedLSTM(nn.Module):
    """
    Conditioned LSTM for GWL forecasting — static features injected into LSTM.

    Reversed flow compared to GWLForecastLSTM:
    1. Static Encoder: continuous + categorical embeddings → MLP → static_encoding
    2. Conditioned LSTM: cat(sequence, lulc_emb, static_encoding) at each timestep → LSTM
    3. Forecast Fusion: cat(dynamic_encoding, forecast_lulc, forecast_features) → MLP → output

    This allows the LSTM to learn different temporal patterns conditioned on static
    context (geology, well type, location, etc.) rather than mixing them post-hoc.

    Same forward() signature and output shape as GWLForecastLSTM.
    """

    def __init__(
        self,
        num_timesteps: int = 10,
        features_per_timestep: int = 10,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
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
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_forecast_features = num_forecast_features
        self.dropout_rate = dropout_rate
        self.use_static_features = use_static_features
        self.static_embedding_dim = static_embedding_dim
        self.lulc_embedding_dim = lulc_embedding_dim
        self.use_linear_residual = use_linear_residual

        # Optional additive linear baseline (see LinearResidualHead docstring).
        # Off by default → identical to the previous architecture.
        self.linear_residual_head = (
            LinearResidualHead() if use_linear_residual else None
        )

        # ====================================================================
        # 1. STATIC ENCODER (processed BEFORE LSTM — no forecast features)
        # ====================================================================
        # LULC embedding — used in both LSTM input and forecast fusion
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
        else:
            total_embedding_dim = 0
            static_continuous_dim = 1  # only well_depth

        static_input_dim = static_continuous_dim + total_embedding_dim
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, static_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # ====================================================================
        # 2. CONDITIONED LSTM (static_encoding injected at every timestep)
        # ====================================================================
        lstm_input_dim = features_per_timestep + lulc_embedding_dim + static_embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )

        # ====================================================================
        # 3. FORECAST FUSION (forecast features join AFTER LSTM)
        # ====================================================================
        forecast_input_dim = lstm_hidden_dim + lulc_embedding_dim + num_forecast_features
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
        forecast_lulc_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass — same signature and return shape as GWLForecastLSTM.

        Returns:
            gwl_pred: shape (batch_size, 1)
        """
        # Step 1: Encode static features (BEFORE LSTM)
        if self.use_static_features:
            lithology_emb = self.lithology_embedding(lithology_idx)
            well_type_emb = self.well_type_embedding(well_type_idx)
            aquifer_emb = self.aquifer_embedding(aquifer_idx)
            aquifer_0_aquifer_emb = self.aquifer_0_aquifer_embedding(aquifer_0_aquifer_idx)
            litho_supergroup_emb = self.litho_supergroup_embedding(litho_supergroup_idx)
            state_emb = self.state_embedding(state_idx)
            district_emb = self.district_embedding(district_idx)

            static_input = torch.cat([
                static_continuous,        # [batch, 5]
                lithology_emb,
                well_type_emb,
                aquifer_emb,
                aquifer_0_aquifer_emb,
                litho_supergroup_emb,
                state_emb,
                district_emb,
            ], dim=1)
        else:
            well_depth = static_continuous[:, 0:1]  # [batch, 1]
            static_input = well_depth

        static_encoding = self.static_encoder(static_input)  # [batch, static_embedding_dim]

        # Step 2: Conditioned LSTM (static injected at every timestep)
        historical_lulc_emb = self.lulc_embedding(historical_lulc_indices)  # [batch, T, lulc_dim]
        T = sequence.size(1)
        static_expanded = static_encoding.unsqueeze(1).expand(-1, T, -1)  # [batch, T, S]
        lstm_input = torch.cat([sequence, historical_lulc_emb, static_expanded], dim=2)

        _, (h_n, _) = self.lstm(lstm_input)
        dynamic_encoding = h_n[-1]  # [batch, lstm_hidden_dim]

        # Step 3: Forecast fusion (forecast features join AFTER LSTM)
        forecast_lulc_emb = self.lulc_embedding(forecast_lulc_idx)  # [batch, lulc_dim]
        forecast_input = torch.cat([
            dynamic_encoding,
            forecast_lulc_emb,
            forecast_features
        ], dim=1)
        fused = self.forecast_fusion(forecast_input)

        # Step 4: Regression head
        gwl_pred = self.regression_head(fused)

        # Optional: add linear-baseline residual path (joint additive output).
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
        forecast_lulc_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Same predict interface as GWLForecastLSTM."""
        self.eval()
        with torch.no_grad():
            gwl_pred = self.forward(
                sequence, static_continuous, forecast_features,
                lithology_idx, well_type_idx, aquifer_idx,
                aquifer_0_aquifer_idx, litho_supergroup_idx,
                state_idx, district_idx,
                historical_lulc_indices, forecast_lulc_idx
            )
            trend_prob = torch.sigmoid(gwl_pred)
            trend_class = (trend_prob.squeeze(-1) >= 0.5).long()
            return {
                'gwl_pred': gwl_pred,
                'trend_prob': trend_prob,
                'trend_class': trend_class
            }


def _per_well_mean(
    per_sample: torch.Tensor,
    well_ids: torch.Tensor,
    trim_high: float = 0.0,
) -> torch.Tensor:
    """Aggregate per-sample loss as mean-per-well, then mean across wells.

    Each well contributes equally regardless of its sample count in the batch.
    Used for "perwell" loss mode to directly target per-well R² improvement.

    Args:
        per_sample: Shape (batch_size,) — per-sample loss values.
        well_ids: Shape (batch_size,) — integer well ID per sample (any range).
        trim_high: Fraction of highest-loss wells to drop from the gradient
            each batch (dynamic top-trim). 0.0 = no trimming (existing behavior).
            0.10 = drop worst 10% of wells per batch. Idea: high-loss wells are
            often unlearnable noise; ignoring them frees gradient for wells that
            follow a pattern.

    Returns:
        Scalar tensor: mean over wells of (mean over samples within well).
    """
    # Map arbitrary well IDs to dense [0..n_wells) indices for scatter
    unique_wells, inverse = torch.unique(well_ids, return_inverse=True)
    n_wells = unique_wells.numel()

    sums = torch.zeros(n_wells, device=per_sample.device, dtype=per_sample.dtype)
    counts = torch.zeros(n_wells, device=per_sample.device, dtype=per_sample.dtype)
    sums.scatter_add_(0, inverse, per_sample)
    counts.scatter_add_(0, inverse, torch.ones_like(per_sample))

    well_means = sums / counts.clamp(min=1.0)  # mean per well

    if trim_high > 0.0 and n_wells > 1:
        # Dynamic per-batch top-trim: drop the highest-loss wells from this
        # batch's gradient. sort() is differentiable through the kept slice.
        sorted_means, _ = well_means.sort()  # ascending
        keep_n = max(1, int(n_wells * (1.0 - trim_high)))
        well_means = sorted_means[:keep_n]

    return well_means.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss: regression (MSE) + trend (BCE) on a single model output.

    The model produces one value (predicted delta or absolute GWL). Two losses
    are computed:
    - Regression loss (MSE): on gwl_pred vs gwl_target (magnitude error)
    - Trend loss (BCE): on the predicted *delta* vs binary trend label (sign error)

    The BCE loss always operates on a delta (change from current), because only
    the sign of a delta indicates trend direction:
    - Delta mode: gwl_pred IS the delta → pass directly as trend_logit
    - Absolute mode: compute delta = gwl_pred - current_gwl → pass as trend_logit
    The caller (train.py) is responsible for computing the correct trend_logit.

    Args:
        alpha: Weight for regression loss (default: 0.7)
        beta: Weight for trend classification loss (default: 0.3)
        pos_weight: Optional weight for positive class (increase) in BCE loss.
                    Use for class imbalance, e.g. pos_weight=2.0 if increases are rare.
        regression_loss_fn: Loss function for regression (default: MSELoss)
        use_uncertainty_weighting: If True, use learnable uncertainty weights (default: False)

    Reference: lstm_design.md (Loss Function section)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        pos_weight: Optional[torch.Tensor] = None,
        regression_loss_fn: Optional[nn.Module] = None,
        use_uncertainty_weighting: bool = False,
        loss_trim_high: float = 0.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.loss_trim_high = loss_trim_high

        # Regression loss
        self.regression_loss_fn = regression_loss_fn or nn.MSELoss()

        # Trend loss: BCE on the predicted delta (logit)
        # sigmoid is applied internally by BCEWithLogitsLoss
        self.trend_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Uncertainty weighting (optional)
        if use_uncertainty_weighting:
            self.log_var_regression = nn.Parameter(torch.zeros(1))
            self.log_var_trend = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        gwl_pred: torch.Tensor,
        gwl_target: torch.Tensor,
        trend_logit: torch.Tensor,
        trend_target: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        well_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss from a single model output.

        Args:
            gwl_pred: Predicted GWL delta/absolute, shape (batch_size, 1)
            gwl_target: Target GWL delta/absolute, shape (batch_size, 1)
            trend_logit: Predicted delta used as BCE logit, shape (batch_size,)
            trend_target: Binary trend labels, shape (batch_size,)
            sample_weights: Optional per-sample weights, shape (batch_size,).
                For "std" mode: 1/std(target_delta) per station, normalized.
                For "perwell" mode: 1/variance(target_delta) per station, normalized.
                For "none" mode: None.
            well_ids: Optional well ID per sample, shape (batch_size,).
                When provided, the loss is computed PER WELL first, then averaged
                across wells (uniform well weighting). When None, samples are
                pooled (sample-weighted mean).

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        # Compute per-sample regression loss (no reduction)
        if isinstance(self.regression_loss_fn, nn.HuberLoss):
            per_sample_reg = nn.functional.huber_loss(
                gwl_pred, gwl_target, reduction='none',
                delta=self.regression_loss_fn.delta
            ).squeeze(-1)
        else:
            per_sample_reg = nn.functional.mse_loss(
                gwl_pred, gwl_target, reduction='none'
            ).squeeze(-1)

        # Per-sample trend loss (no reduction)
        per_sample_trend = nn.functional.binary_cross_entropy_with_logits(
            trend_logit, trend_target.float(), reduction='none'
        )

        # Apply per-sample weights if provided (1/std or 1/var depending on mode)
        if sample_weights is not None:
            per_sample_reg = per_sample_reg * sample_weights
            per_sample_trend = per_sample_trend * sample_weights

        # Aggregation: per-well mean then mean across wells, OR pooled mean
        if well_ids is not None:
            loss_regression = _per_well_mean(per_sample_reg, well_ids, self.loss_trim_high)
            loss_trend = _per_well_mean(per_sample_trend, well_ids, self.loss_trim_high)
        else:
            loss_regression = per_sample_reg.mean()
            loss_trend = per_sample_trend.mean()

        if self.use_uncertainty_weighting:
            # Uncertainty weighting (Kendall et al., 2018)
            precision_regression = torch.exp(-self.log_var_regression)
            precision_trend = torch.exp(-self.log_var_trend)

            total_loss = (
                precision_regression * loss_regression + self.log_var_regression +
                precision_trend * loss_trend + self.log_var_trend
            )
        else:
            # Fixed weighting
            total_loss = self.alpha * loss_regression + self.beta * loss_trend

        loss_dict = {
            'total_loss': total_loss.item(),
            'regression_loss': loss_regression.item(),
            'trend_loss': loss_trend.item()
        }

        if self.use_uncertainty_weighting:
            loss_dict['log_var_regression'] = self.log_var_regression.item()
            loss_dict['log_var_trend'] = self.log_var_trend.item()

        return total_loss, loss_dict


def classify_trend(
    current_gwl: torch.Tensor,
    target_gwl: torch.Tensor,
    threshold: float = 0.5,
    use_delta_gwl: bool = True
) -> torch.Tensor:
    """
    Convert GWL change to trend class labels (2-class: decrease vs increase).

    In delta mode: target_gwl IS the delta (target - current), use directly.
    In absolute mode: compute delta = target_gwl - current_gwl.

    Args:
        current_gwl: Current GWL values (raw absolute), shape (batch_size,)
        target_gwl: Target GWL — delta (delta mode) or absolute (absolute mode),
                    shape (batch_size,)
        threshold: Not used (kept for API compatibility)
        use_delta_gwl: If True, target_gwl is already a delta

    Returns:
        Trend class labels, shape (batch_size,) with values:
            0 = Decrease (Δ < 0)
            1 = Increase or stable (Δ >= 0)

    Reference: lstm_design.md (Trend Classification section)
    """
    if use_delta_gwl:
        delta = target_gwl  # Already a delta
    else:
        delta = target_gwl - current_gwl

    # 2-class: 0 = decrease, 1 = increase/stable
    trend_labels = torch.ones_like(delta, dtype=torch.long)  # Default to increase (1)
    trend_labels[delta < 0] = 0  # Decrease

    return trend_labels


if __name__ == '__main__':
    """
    Test the model with dummy data to verify shapes and forward pass.

    Usage:
        python gwl_lstm/model.py
        python gwl_lstm/model.py --device cuda
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Test GWL Forecasting LSTM model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, mps')
    args = parser.parse_args()

    # Resolve device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("Testing GWL Forecasting LSTM")
    print("=" * 70)
    print(f"Device: {device}")

    # Create model
    model = GWLForecastLSTM(
        num_timesteps=10,
        features_per_timestep=10,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        num_lithology_types=8,
        num_well_types=5,
        num_aquifer_types=3,
        num_aquifer_0_aquifer_types=5,
        num_litho_supergroup_types=8,
        num_lulc_types=10,
        num_state_types=30,
        num_district_types=700,
        num_forecast_features=6
    ).to(device)

    print(f"\nModel Architecture:")
    print(f"  Timesteps: {model.num_timesteps}")
    print(f"  Features per timestep: {model.features_per_timestep}")
    print(f"  Forecast features: {model.num_forecast_features} (for conditioning)")
    print(f"  LSTM hidden dim: {model.lstm_hidden_dim}")
    print(f"  LSTM layers: {model.lstm_num_layers}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dummy batch on device
    batch_size = 4
    sequence = torch.randn(batch_size, 10, 10, device=device)
    static_continuous = torch.randn(batch_size, 4, device=device)  # elevation, stream_order, lat, long
    forecast_features = torch.randn(batch_size, 6, device=device)
    lithology_idx = torch.randint(0, 8, (batch_size,), device=device)
    well_type_idx = torch.randint(0, 5, (batch_size,), device=device)
    aquifer_idx = torch.randint(0, 3, (batch_size,), device=device)
    aquifer_0_aquifer_idx = torch.randint(0, 5, (batch_size,), device=device)
    litho_supergroup_idx = torch.randint(0, 8, (batch_size,), device=device)
    state_idx = torch.randint(0, 30, (batch_size,), device=device)
    district_idx = torch.randint(0, 700, (batch_size,), device=device)
    historical_lulc_indices = torch.randint(0, 10, (batch_size, 10), device=device)
    forecast_lulc_idx = torch.randint(0, 10, (batch_size,), device=device)

    print(f"\nTest batch size: {batch_size}")
    print(f"Sequence shape: {sequence.shape}")
    print(f"Forecast features shape: {forecast_features.shape}")
    print(f"Static continuous shape: {static_continuous.shape}")

    # Forward pass
    gwl_pred = model(
        sequence,
        static_continuous,
        forecast_features,
        lithology_idx,
        well_type_idx,
        aquifer_idx,
        aquifer_0_aquifer_idx,
        litho_supergroup_idx,
        state_idx,
        district_idx,
        historical_lulc_indices,
        forecast_lulc_idx
    )

    print(f"\nForward pass results:")
    print(f"  GWL predictions shape: {gwl_pred.shape}")
    print(f"  GWL predictions (sample): {gwl_pred[:2, 0].detach().cpu().numpy()}")

    # Test prediction method
    predictions = model.predict(
        sequence,
        static_continuous,
        forecast_features,
        lithology_idx,
        well_type_idx,
        aquifer_idx,
        aquifer_0_aquifer_idx,
        litho_supergroup_idx,
        state_idx,
        district_idx,
        historical_lulc_indices,
        forecast_lulc_idx
    )

    print(f"\nPrediction method results:")
    print(f"  Keys: {list(predictions.keys())}")
    print(f"  Trend prob (P(increase)): {predictions['trend_prob'].squeeze().cpu().numpy()}")
    print(f"  Trend classes: {predictions['trend_class'].cpu().numpy()}")

    # Test classify_trend in both modes
    print("\n" + "=" * 70)
    print("Testing classify_trend (both modes)")
    print("=" * 70)

    current_gwl = torch.tensor([10.0, 15.0, 8.0, 20.0], device=device)

    # Delta mode: target_gwl is already a delta
    target_delta = torch.tensor([-2.0, 3.0, 0.0, -5.0], device=device)
    trend_delta = classify_trend(current_gwl, target_delta, use_delta_gwl=True)
    print(f"\nDelta mode:")
    print(f"  target (deltas):  {target_delta.cpu().numpy()}")
    print(f"  trend labels:     {trend_delta.cpu().numpy()}")
    print(f"  expected:         [0, 1, 1, 0]")  # decrease, increase, stable, decrease

    # Absolute mode: target_gwl is absolute, delta computed internally
    target_abs = current_gwl + target_delta  # [8, 18, 8, 15]
    trend_abs = classify_trend(current_gwl, target_abs, use_delta_gwl=False)
    print(f"\nAbsolute mode:")
    print(f"  current:          {current_gwl.cpu().numpy()}")
    print(f"  target (absolute):{target_abs.cpu().numpy()}")
    print(f"  trend labels:     {trend_abs.cpu().numpy()}")
    print(f"  expected:         [0, 1, 1, 0]")

    assert torch.equal(trend_delta, trend_abs), "Delta and absolute modes should produce same labels!"
    print("\n  ✓ Both modes produce identical trend labels")

    # Test loss function (single output, two losses)
    print("\n" + "=" * 70)
    print("Testing Multi-Task Loss (shared output: MSE + BCE)")
    print("=" * 70)

    gwl_target = torch.randn(batch_size, 1, device=device)
    trend_target = classify_trend(
        current_gwl, gwl_target.squeeze(), use_delta_gwl=True
    )

    print(f"\nDummy targets (delta mode):")
    print(f"  Target delta: {gwl_target.squeeze().cpu().numpy()}")
    print(f"  Trend classes: {trend_target.cpu().numpy()}")

    # Compute loss — 4 args: gwl_pred, gwl_target, trend_logit, trend_target
    # In delta mode: trend_logit = gwl_pred (it IS the delta)
    # In absolute mode: trend_logit = gwl_pred - current_gwl
    loss_fn = MultiTaskLoss(alpha=0.7, beta=0.3).to(device)
    trend_logit = gwl_pred.squeeze(-1)  # delta mode: pred is already a delta
    total_loss, loss_dict = loss_fn(gwl_pred, gwl_target, trend_logit, trend_target)

    print(f"\nLoss computation:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Test backward pass (verify gradients flow)
    total_loss.backward()
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    print(f"\nGradient check: {len(grad_norms)} parameters with gradients")
    print(f"  Max grad norm: {max(grad_norms.values()):.6f}")
    print(f"  Min grad norm: {min(grad_norms.values()):.6f}")

    # Verify trend derived from regression output matches
    with torch.no_grad():
        trend_from_pred = (torch.sigmoid(gwl_pred.squeeze()) >= 0.5).long()
        print(f"\nTrend from regression output sign:")
        print(f"  Predicted deltas: {gwl_pred.squeeze().cpu().numpy()}")
        print(f"  sigmoid(pred):    {torch.sigmoid(gwl_pred.squeeze()).cpu().numpy()}")
        print(f"  Trend classes:    {trend_from_pred.cpu().numpy()}")

    # Demonstrate absolute mode: must compute delta for BCE
    print("\n" + "=" * 70)
    print("Absolute mode BCE: must use (pred - current) as logit")
    print("=" * 70)
    abs_pred = torch.tensor([[62.0]], device=device)    # Predicted absolute GWL
    abs_current = torch.tensor([60.0], device=device)   # Current GWL
    abs_target_val = torch.tensor([[65.0]], device=device) # Target absolute GWL
    abs_trend = torch.tensor([1], device=device)         # Actual trend = increase

    # WRONG: passing absolute prediction directly as logit
    # sigmoid(62) ≈ 1.0 always → meaningless
    wrong_logit = abs_pred.squeeze(-1)
    print(f"  WRONG: sigmoid(abs_pred={wrong_logit.item():.1f}) = {torch.sigmoid(wrong_logit).item():.6f} (always ~1.0)")

    # CORRECT: passing delta (pred - current) as logit
    correct_logit = abs_pred.squeeze(-1) - abs_current  # 62 - 60 = +2
    print(f"  CORRECT: sigmoid(pred-current={correct_logit.item():.1f}) = {torch.sigmoid(correct_logit).item():.4f} (meaningful)")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)