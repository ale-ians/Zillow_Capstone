import numpy as np

def holdout_last_n(df, n=12):
    return df.iloc[:-n].copy(), df.iloc[-n:].copy()

def expanding_window_splits(n_obs, initial=36, step=6)
    start = initial
    while start + step <= n_obs:
        yield np.arange(0, start), np.arange(start, start, start + step)
        start += step