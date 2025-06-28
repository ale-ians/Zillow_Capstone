import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def forecast_home_values(df_all, target_zip, forecast_months=6):
    df_zip = df_all[df_all['RegionName'] == target_zip].sort_values('date').copy()

    for lag in [1,2,3]:
        df_zip[f'value_lag_{lag}'] = df_zip['price'].shift(lag)
    df_zip = df_zip.dropna()

    feature_cols = [f'value_lag_{lag}' for lag in [1,2,3]]
    X = df_zip[feature_cols]
    y = df_zip['price']

    model = RandomForestRegressor(n_estimators=100, max_depth=42)
    model.fit(X,y)

    df_history = df_zip[['date','price']].copy()
    last_known = df_zip.iloc[-1:].copy()
    future_dates = pd.date_range(start=last_known['date'].values[0], periods= forecast_months+1, freq='MS')[1:]

    forecasts = []
    for date in future_dates:
        lag_1 = last_known['price'].values[0]
        lag_2 = last_known['value_lag_1'].values[0]
        lag_3 = last_known['value_lag_2'].values[0]
        input_data = pd.DataFrame([[lag_1, lag_2, lag_3]], columns=feature_cols)
        pred = model.predict(input_data)[0]
        forecasts.append({'date': date, 'predicted_value': pred})

        last_known = pd.DataFrame([{
           'price': pred,
           'value_lag_1':lag_1,
           'value_lag_2':lag_2
        }])

    return df_history, pd.DataFrame(forecasts)
