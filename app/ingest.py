import pandas as pd
from pathlib import Path

def run(raw_path, out_path):
    df = pd.read_csv(raw_path)
    id_vars = ['RegionID', 'RegionName', 'SizeRank', 'RegionType', 'StateName']
    date_cols = df.columns.difference(id_vars)

    # wide to long
    df = df.melt(id_vars=id_vars, value_vars=date_cols,
                 var_name='date', value_name='price')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'price'], inplace = True)

    #Keep Colorado ZIP Codes (80xxx to 81xxx up to 816xx), zfill
    df['RegionName'] = df['RegionName'].astype(str).str.zfill(5)
    df = df[df[['RegionName'].str.match(r'(80\d{3}|81[0-6]\d)$')]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df[['RegionName', 'date', 'price']].to_csv(out_path, index=False)
    return pd.read_csv(out_path, parse_dates=['date'])
