import json, joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from app.features import add_lag_features
from app.splitting import holdout_last_n, expanding_window_splits

def _cv_select(df_feat, X_cols, y_col, model_name):
    params_grid = {
        "rf" : [{"n_estimators" : n} for n in range(100, 200, 400)],
        "xgb": [{"n_estimators" : n, "learning_rate": "lr"} for n in (200,400) for lr in (0.05, 0.1)]
    }[model_name]

    best, best_mae = None, float("inf")
    for p in params_grid:
        maes = []
        for tr, va in expanding_window_splits(len(df_feat), initial=36, step=6):
            Xtr, ytr = df_feat.iloc[tr][X_cols], df_feat.iloc[tr][y_col]
            Xva, yva = df_feat.iloc[va][X_cols], df_feat.iloc[va][y_col]
            if model_name == "rf":
                m = RandomForestRegressor(random_state=42, **p)
            else:
                m = XGBRegressor(random_state=42, objective="reg:squarederror", **p)
            m.fit(Xtr, ytr)
            maes.append(mean_absolute_error(yva, m.predict(Xva)))
        if np.mean(maes) < best_mae:
            best_mae, best = np.mean(maes), p
    return best, best_mae

def train_one_zip(df_zip, lags=(1,2,3), model_name="rf"):
    df_feat = add_lag_features(df_zip[['date','price']], lags = lags)
    train, test = holdout_last_n(df_feat, n=12)
    X_cols = [f'value_lag_{l}' for l in lags]; y_col = 'price'

    best_params, cv_mae = _cv_select(train, X_cols, y_col, model_name)
    if model_name == "rf":
        model = RandomForestRegressor(random_state=42, **best_params)

    model.fit(train[X_cols], train[y_col])
    test_mae = mean_absolute_error(test[y_col], model.predict(test[X_cols]))
    return model, {"model": model_name, "lags": lags, "cv_mae": float(cv_mae), "test_mae": float(test_mae), "params": best_params}

def save_artifacts(model, meta, tag='rf'):
    import pathlib, json, joblib
    pathlib.Path('artifacts').mkdir(exist_ok=True)
    joblib.dump(model, f'artifacts/model_{tag}.joblib')
    with open("artifacts/featurespec.json", "w") as f:
        json.dump(meta, f, indent=2)