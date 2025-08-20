import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from app.features import add_lag_features
from app.splitting import holdout_last_n

def mae_by_zip(df_all, model_name="rf", lags(1,2,3)):
    rows = []
    for zip_code, d in df_all.groupby("RegionName"):
        d = add_lag_features(d[["date", "price"]], lags=lags)
        if len(d) < 24: continue
        train, test = holdout_last_n(d, n=12)
        X_cols = [f'value_lag_{l}' for l in lags]; y_col = 'price'
        if model_name == 'rf':
            m = RandomForestRegressor(n_estimators=200, random_state=42)
        else:
            m = XGBRegressor(n_estimators=400, learning_rate=0.1, random_state=42, objective='reg:squarederror')
            m.fit(train[X_cols], train[y_col])
            mae = mean_absolute_error(test[y_col], m.predict(test[X_cols]))
            rows.append({"RegionName":zip_code, "MAE":float(mae)})
    return pd.DataFrame(rows)