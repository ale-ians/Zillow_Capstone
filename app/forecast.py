import pandas as pd
import joblib, json
from app.features import add_lag_features

def forecast(df_zip, model_path, spec_path, months=6):
    model = joblib.load(model_path)
    with open(spec_path) as f: spec = json.load(f)
    lags = tuple(spec['lags']); X_cols = [f'value_lag{l}' for l in lags]

    df = add_lag_features(df_zip[["date", "price"]], lags=lags)
    history = df[['date', 'price']].copy()
    last = df.iloc[-1].copy()

    preds = []
    for i in range(1, months+1):
        feats = last[X_cols].values.reshape(1,-1)
        yhat = model.predict(feats)[0]
        next_date = (pd.to_datetime(last['date']) + pd.DateOffset(months=1)).normalize().replace(day=1)
        preds.append({"date": next_date, "predicted": yhat})
        #roll lags
        for l in reversed(lags):
            if l == 1: last[f'value_lag_1'] = yhat
            else:
                last[f'value_lag_{l}'] = last[f'value_lag_{l-1}']
        last['date'] = next_date; last['price'] = yhat
    return history, pd.DataFrame(preds)