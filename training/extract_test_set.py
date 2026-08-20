"""Reproduce the spatial split from Run_With_MWS_Split.py and save the test set.

Replicates lines 5-170 of Run_With_MWS_Split.py exactly:
  1. Load merged_data_with_spei_sensitivity.csv
  2. Feature engineering (same columns)
  3. Drop NaNs on critical_cols then dropna()
  4. perform_spatial_split (70/20/10 by MWS volume)
  5. Save test split as (mws_id, date, NDVI) to ndvi_test_set.csv
"""

import os

import pandas as pd
import numpy as np

# env-driven (no machine-specific paths): set GWL_NDVI_SRC / GWL_NDVI_DST before running.
SRC = os.environ.get("GWL_NDVI_SRC", "")
DST = os.environ.get("GWL_NDVI_DST", "")
if not SRC or not DST:
    raise SystemExit("set GWL_NDVI_SRC (merged_data_with_spei_sensitivity.csv) and GWL_NDVI_DST (output)")

# --- Replicate Run_With_MWS_Split.py lines 5-118 ---
df_complete = pd.read_csv(SRC)
print(f"Loaded: {len(df_complete):,} rows")

df_complete['date'] = pd.to_datetime(df_complete['date'])
df_complete = df_complete.sort_values(['mws_id', 'date']).reset_index(drop=True)

df_complete['year'] = df_complete['date'].dt.year
df_complete['month'] = df_complete['date'].dt.month
df_complete['day_of_year'] = df_complete['date'].dt.dayofyear

df_complete['season'] = df_complete['month'].apply(
    lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
)

df_complete['days_since_start'] = (df_complete['date'] - df_complete['date'].min()).dt.days
df_complete['month_sin'] = np.sin(2 * np.pi * df_complete['month'] / 12)
df_complete['month_cos'] = np.cos(2 * np.pi * df_complete['month'] / 12)
df_complete['doy_sin'] = np.sin(2 * np.pi * df_complete['day_of_year'] / 365)
df_complete['doy_cos'] = np.cos(2 * np.pi * df_complete['day_of_year'] / 365)
df_complete['precip_et_ratio'] = df_complete['precipitation'] / (df_complete['et'] + 0.001)
df_complete['precip_runoff_diff'] = df_complete['precipitation'] - df_complete['runoff']

df_complete['precipitation_prev'] = df_complete.groupby('mws_id')['precipitation'].shift(1)
df_complete['et_prev'] = df_complete.groupby('mws_id')['et'].shift(1)
df_complete['NDVI_prev'] = df_complete.groupby('mws_id')['NDVI'].shift(1)
df_complete['NDVI_prev2'] = df_complete.groupby('mws_id')['NDVI'].shift(2)

for col in ['NDVI']:
    df_complete[f'{col}_rolling_mean'] = df_complete.groupby('mws_id')[col].shift(1).transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df_complete[f'{col}_rolling_std'] = df_complete.groupby('mws_id')[col].shift(1).transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

df_complete['NDVI_change_prev'] = df_complete['NDVI_prev'] - df_complete['NDVI_prev2']

from sklearn.preprocessing import LabelEncoder
encoders = {}
for col in ['state', 'district', 'tehsil', 'mws_id']:
    le = LabelEncoder()
    df_complete[f'{col}_encoded'] = le.fit_transform(df_complete[col])
    encoders[col] = le

dynamic_features = [
    'NDVI', 'month_sin', 'month_cos',
    'et', 'runoff', 'precipitation',
    'dewpoint_temperature_2m', 'sm_rootzone', 'sm_surface',
    'surface_net_thermal_radiation_sum', 'surface_solar_radiation_downwards_sum',
    'temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
    'precip_et_ratio'
]

static_numerical_features = ['latitude', 'longitude', 'baseline_ndvi_mean', 'baseline_ndvi_std']
static_categorical_features = ['state_encoded', 'district_encoded', 'tehsil_encoded', 'mws_id_encoded']
static_features = static_numerical_features + static_categorical_features

critical_cols = list(set(dynamic_features + static_features + ['date', 'mws_id', 'NDVI']))

initial_count = len(df_complete)
df_complete = df_complete.dropna(subset=critical_cols)
rows_lost = initial_count - len(df_complete)
print(f"Targeted drop: {initial_count:,} -> {len(df_complete):,} ({rows_lost:,} lost)")

df_complete = df_complete.dropna()
print(f"After full dropna: {len(df_complete):,}")


# --- Replicate perform_spatial_split (lines 124-166) ---
def perform_spatial_split(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    station_counts = df.groupby(['district', 'mws_id']).size().reset_index(name='record_count')
    station_counts = station_counts.sort_values(by='record_count', ascending=False)

    total_records = station_counts['record_count'].sum()
    station_counts['cum_perc'] = station_counts['record_count'].cumsum() / total_records

    train_ids = station_counts[station_counts['cum_perc'] <= train_ratio]['mws_id'].unique()
    val_mask = (station_counts['cum_perc'] > train_ratio) & (station_counts['cum_perc'] <= (train_ratio + val_ratio))
    val_ids = station_counts[val_mask]['mws_id'].unique()
    test_ids = station_counts[station_counts['cum_perc'] > (train_ratio + val_ratio)]['mws_id'].unique()

    df_train = df[df['mws_id'].isin(train_ids)].copy()
    df_val = df[df['mws_id'].isin(val_ids)].copy()
    df_test = df[df['mws_id'].isin(test_ids)].copy()

    print(f"\nSET ALLOCATION SUMMARY")
    print(f"Total Rows: {total_records:,}")
    print(f"Train: {len(df_train):,} rows ({len(train_ids)} IDs) | {len(df_train)/total_records:.1%}")
    print(f"Val:   {len(df_val):,} rows ({len(val_ids)} IDs) | {len(df_val)/total_records:.1%}")
    print(f"Test:  {len(df_test):,} rows ({len(test_ids)} IDs) | {len(df_test)/total_records:.1%}")

    return df_train, df_val, df_test


df_train, df_val, df_test = perform_spatial_split(df_complete)

# --- Save test set: only (mws_id, date, NDVI) ---
test_out = df_test[['mws_id', 'date', 'NDVI']].sort_values(['mws_id', 'date']).reset_index(drop=True)
test_out.to_csv(DST, index=False)

n_mws = test_out['mws_id'].nunique()
print(f"\nSaved test set: {len(test_out):,} rows, {n_mws} MWS IDs -> {DST}")

# Quick stats
counts = test_out.groupby('mws_id').size()
print(f"Rows per MWS: min={counts.min()}, median={int(counts.median())}, max={counts.max()}")
