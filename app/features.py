import pandas as pd

def add_lag_features(df_zip: pd.DataFrame, lags=(1,2,3), target_col="price", prefix="value"):
    if "date" not in df_zip.columns or target_col not in df_zip.columns:
        raise ValueError(f"df_zip must contain ['date', '{target_col}']; got {list(df_zip.columns)}")

    df = df_zip[["date", target_col]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["date", target_col]).sort_values("date").reset_index(drop=True)

    lags = tuple(int(k) for k in lags)
    if any(k <= 0 for k in lags):
        raise ValueError(f"All lags must be positive integers; got {lags}")

    for k in lags:
        df[f"{prefix}_lag_{k}"] = df[target_col].shift(k)

    lag_cols = [f"{prefix}_lag_{k}" for k in lags]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No rows left after creating lags={lags}."
            f"Series may be too short (need > max (lags)).")

    return df