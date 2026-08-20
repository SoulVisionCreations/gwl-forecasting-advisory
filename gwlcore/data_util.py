def split_by_district(gwl_df, train_frac=0.6, val_frac=0.3):
    """Splits stations into train/val/test sets based on district and reading counts."""
    districts = gwl_df["district"].unique()
    train_stations, val_stations, test_stations = set(), set(), set()

    for district in districts:
        district_df = gwl_df[gwl_df["district"] == district]
        station_counts = district_df.groupby("station_code").size().sort_values(ascending=False)
        n_stations = len(station_counts)
        n_train = int(train_frac * n_stations)
        n_val   = int(val_frac * n_stations)

        stations = station_counts.index.tolist()
        train_stations.update(stations[:n_train])
        val_stations.update(stations[n_train:n_train + n_val])
        test_stations.update(stations[n_train + n_val:])

    return train_stations, val_stations, test_stations

def split_by_date(gwl_df, train_end_date, val_start_date, val_end_date, test_start_date, test_end_date):
    """Splits stations into train/val/test sets based on date ranges."""
    import pandas as pd
    dates = gwl_df["date"]
    train_stations = set(gwl_df[dates <= pd.Timestamp(train_end_date)]["station_code"].unique())
    val_stations   = set(gwl_df[(dates >= pd.Timestamp(val_start_date))  & (dates <= pd.Timestamp(val_end_date))]["station_code"].unique())
    test_stations  = set(gwl_df[(dates >= pd.Timestamp(test_start_date)) & (dates <= pd.Timestamp(test_end_date))]["station_code"].unique())
    return train_stations, val_stations, test_stations
