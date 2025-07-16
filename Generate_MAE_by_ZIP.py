import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def generate_mae_by_zip(home_values_path, geo_data_path, output_path):
    #Load Housing Data
    df_co = pd.read_csv(home_values_path, parse_dates=['date'])
    df_co['RegionName'] = df_co['RegionName'].astype(str).str.zfill(5)

    results = []

    for zip_code in df_co['RegionName'].unique():
        df_zip = df_co[df_co['RegionName'] == zip_code].sort_values(by='date').copy()

        for lag in range(1,4):
            df_zip[f'value_lag{lag}'] = df_zip['price'].shift(lag)

            df_zip.dropna(inplace=True)

        if len(df_zip) < 20:
            continue

        train = df_zip[:-12]
        test = df_zip[-12:]

        feature_cols = [f'value_lag{i}' for i in range(1,4)]
        X_train, y_train = train[feature_cols], train['price']
        X_test, y_test = test[feature_cols], test['price']

        if X_test.isnull().values.any() or X_train.empty:
            continue

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        results.append({'RegionName': zip_code, 'MAE': mae})

    #Create DateFrame of MAE results
    mae_df = pd.DataFrame(results)

    #Load Geographic Data
    df_geo = pd.read_csv(geo_data_path, dtype={"Zip Code Tabulation Area Code": str})
    df_geo = df_geo.rename(columns={
        "Zip Code Tabulation Area Code": "RegionName",
        "Internal Point Latitude": "lat",
        "Internal Point Longitude": 'lng'
    })

    #Merge and Save
    mae_geo = pd.merge(mae_df, df_geo[['RegionName', 'lat', 'lng']], on='RegionName', how='inner')
    mae_geo.to_csv(output_path, index=False)

    return mae_geo