import pandas as pd

def add_lag_features(df_zip, lags=(1,2,3)):
    df_zip = df_zip.sore_values('date').copy()
    for l in lags:
        df_zip[f'value_lag_{1}'] = df_zip['price'].shift(l)
    return df_zip.dropna().reset_index(drop=True)