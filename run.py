from pathlib import Path
import pandas as pd

from app.ingest import run as ingest_run
from app.features import add_lag_features
from app.train import train_one_zip, save_artifacts
from app.evaluate import mae_by_zip


DATA_RAW = Path("./data/raw/Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv")
DATA_CLEAN = Path("./data/processed/colorado_home_values.csv")

LAGS = (1, 2, 3)
DESIRED_HOLDOUT = 12      # your train_one_zip can shrink this if series is short
MIN_ROWS_AFTER_LAG = 50   # tweak: “good enough” history for demo model


def _has_enough_history(df_zip: pd.DataFrame, lags=LAGS, min_rows=MIN_ROWS_AFTER_LAG) -> bool:
    """Check rows remaining after lagging & dropna."""
    try:
        df_feat = add_lag_features(df_zip[['date', 'price']], lags=lags, target_col='price', prefix='value')
        return len(df_feat) >= min_rows
    except Exception:
        return False


def _pick_sample_zip(df: pd.DataFrame, lags=LAGS, min_rows=MIN_ROWS_AFTER_LAG):
    """Return a ZIP with enough rows after lagging. Prefer the given sample if possible."""
    # First try the user’s preferred demo ZIP
    preferred = 80132
    if (df.RegionName == preferred).any():
        df_zip = df.loc[df.RegionName == preferred, ['date', 'price']].copy()
        df_zip['date'] = pd.to_datetime(df_zip['date'], errors='coerce')
        df_zip['price'] = pd.to_numeric(df_zip['price'], errors='coerce')
        df_zip = df_zip.dropna().sort_values('date')
        if _has_enough_history(df_zip, lags, min_rows):
            return preferred

    # Otherwise scan for a good candidate (max rows after lag)
    best_zip, best_len = None, -1
    for z, g in df.groupby('RegionName'):
        g = g[['date', 'price']].copy()
        g['date'] = pd.to_datetime(g['date'], errors='coerce')
        g['price'] = pd.to_numeric(g['price'], errors='coerce')
        g = g.dropna().sort_values('date')
        try:
            feat = add_lag_features(g, lags=lags, target_col='price', prefix='value')
            n = len(feat)
            if n >= min_rows and n > best_len:
                best_zip, best_len = z, n
        except Exception:
            continue

    return best_zip  # may be None


def main():
    # 1) Ingest cleaned dataset
    df = ingest_run(str(DATA_RAW), str(DATA_CLEAN))
    # Ensure expected columns exist
    required = {'RegionName', 'date', 'price'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ingest_run did not produce required columns {required}. Missing: {missing}")

    # 2) Pick a robust sample ZIP for demo training
    sample_zip = _pick_sample_zip(df, lags=LAGS, min_rows=MIN_ROWS_AFTER_LAG)
    if sample_zip is None:
        raise ValueError("No ZIP has enough history after lagging; reduce LAGS/holdout or aggregate.")
    print(f"[INFO] Using sample ZIP for demo training: {sample_zip}")

    df_zip = df.loc[df.RegionName == sample_zip, ['date', 'price']].copy()
    df_zip['date'] = pd.to_datetime(df_zip['date'], errors='coerce')
    df_zip['price'] = pd.to_numeric(df_zip['price'], errors='coerce')
    df_zip = df_zip.dropna().sort_values('date')

    # 3) Train demo models (don’t let a single failure kill the run)
    for model_name in ("rf", "xgb"):
        try:
            model, meta = train_one_zip(df_zip, lags=LAGS, model_name=model_name)
            save_artifacts(model, meta, tag=model_name)
            print(f"[INFO] Trained {model_name} on {sample_zip}: test_mae={meta.get('test_mae')}, n_train={meta.get('n_train')}")
        except Exception as e:
            print(f"[WARN] {model_name} demo training failed on {sample_zip}: {e}")

    # 4) Evaluate all ZIPs, skipping short/invalid ones inside mae_by_zip
    mae_by_zip(df, "rf").to_csv("artifacts/mae_by_zip_rf.csv", index=False)
    try:
        mae_by_zip(df, "xgb").to_csv("artifacts/mae_by_zip_xgb.csv", index=False)
    except Exception as e:
        print(f"[WARN] mae_by_zip for xgb failed: {e}")


if __name__ == "__main__":
    main()