import json, joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from app.features import add_lag_features
from app.splitting import holdout_last_n, expanding_window_splits

def _cv_select(df_feat, X_cols, y_col, model_name, initial=36, step=6):
    default_params = {"rf": {"n_estimators": 200},
                      "xgb": {"n_estimators": 400, "learning_rate": 0.1}}
    n_obs = len(df_feat)
    if n_obs < (initial + step):
        #Not enough data to do any folds-return defaults and NaN CV score
        return default_params[model_name], float("nan")

    param_grid = {
        "rf": [{"n_estimators": n} for n in (100,200,400)],
        "xgb": [{"n_estimators": n, "learning_rate": lr}
                for n in (200,400) for lr in (0.05, 0.1)]
    }[model_name]

    best, best_mae = None, float("inf")

    for p in param_grid:
        maes = []
        for tr_idx, va_idx in expanding_window_splits(n_obs, initial=initial, step=step):
            Xtr = df_feat.iloc[tr_idx][X_cols]
            ytr = df_feat.iloc[tr_idx][y_col]
            Xva = df_feat.iloc[va_idx][X_cols]
            yva = df_feat.iloc[va_idx][y_col]

            if Xtr.empty or Xva.empty:
                continue

            if model_name=="rf":
                m = RandomForestRegressor(random_state=42, **p)
            else:
                m = XGBRegressor(random_state=42, objective='reg:squarederror', **p)

            m.fit(Xtr, ytr)
            pred = m.predict(Xva)
            maes.append(mean_absolute_error(yva, pred))

        #skip configs that produce no folds
        if not maes:
            continue

        mean_mae = float(np.mean(maes))
        if mean_mae < best_mae:
            best_mae, best = mean_mae, p

    if best is None: # all configs failed to produce folds
        return default_params[model_name], float("nan")

    return best, best_mae

def _choose_lag_columns(df, lags):
    """
    Return the correct lag feature column names for the given lags.
    Tries common prefixes first ('price', 'value'), then falls back to any '*_lag_{k}' match.
    """
    # 1) Try common prefixes
    for prefix in ("price", "value", "y", "target"):
        cols = [f"{prefix}_lag_{k}" for k in lags]
        if all(c in df.columns for c in cols):
            return cols

    # 2) Fallback: any columns that end with _lag_{k}
    chosen = []
    for k in lags:
        matches = [c for c in df.columns if c.endswith(f"_lag_{k}")]
        if len(matches) == 1:
            chosen.append(matches[0])
        elif len(matches) > 1:
            # If multiple match, just take the first deterministically
            matches.sort()
            chosen.append(matches[0])
        else:
            raise KeyError(
                f"No column found for lag {k}. "
                f"Available columns: {list(df.columns)}"
            )

    return chosen

def train_one_zip(df_zip, lags=(1,2,3), model_name="rf"):
    #Build features and drop NaNs from lagging
    df_feat = add_lag_features(df_zip[["date", "price"]], lags = lags)
    df_feat = df_feat.dropna().reset_index(drop=True)

    #Hold out last 12 months
    train, test = holdout_last_n(df_feat, n=6)

    #Auto detect feature names
    X_cols = _choose_lag_columns(train, lags)
    y_col = "price"

    #If train is still too short for CV settings, _cv_select will fall back
    best_params, cv_mae = _cv_select(train, X_cols, y_col, model_name)
    if best_params is None:
        best_params = _default_params(model_name)

    #Fit chosen model
    if model_name == "rf":
        model = RandomForestRegressor(random_state = 42, **best_params)
    else:
        model = XGBRegressor(random_state = 42, objective = "reg:squarederror", **best_params)

    model.fit(train[X_cols], train[y_col])
    test_mae = mean_absolute_error(test[y_col], model.predict(test[X_cols]))

    return model, {
        "model": model_name,
        "lages": lags,
        "cv_mae": float(cv_mae) if not np.isnan(cv_mae) else None,
        "test_mae": float(test_mae),
        "params": best_params,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

def save_artifacts(model, meta, tag='rf'):
    import pathlib, json, joblib
    pathlib.Path('artifacts').mkdir(exist_ok=True)
    joblib.dump(model, f'artifacts/model_{tag}.joblib')
    with open("artifacts/featurespec.json", "w") as f:
        json.dump(meta, f, indent=2)