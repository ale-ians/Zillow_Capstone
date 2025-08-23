import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.metrics import mean_absolute_error

# import pipeline helpers
from app.features import add_lag_features
from app.train import train_one_zip  # uses same CV/holdout as pipeline


# Streamlit config

st.set_page_config(page_title="CO Home Value Forecast", layout="wide")

# Paths & constants

DATA_PATH = "data/processed/colorado_home_values.csv"
GEO_PATH = "data/external/us_zip_centroids.csv"
ART_MAE_RF = "artifacts/mae_by_zip_rf.csv"
ART_MAE_XGB = "artifacts/mae_by_zip_xgb.csv"

DEFAULT_LAGS = (1, 2, 3)
DEFAULT_FORECAST_MONTHS = 6


# Cached loaders

@st.cache_data
def load_home_values(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # normalize ZIP dtype and formatting
    df["RegionName"] = df["RegionName"].astype(str).str.zfill(5)
    # ensure numeric price and valid dates
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    return df.sort_values(["RegionName", "date"]).reset_index(drop=True)

@st.cache_data
def load_geo(path: str) -> pd.DataFrame:
    geo = pd.read_csv(path, dtype={"Zip Code Tabulation Area Code": str})
    geo.columns = geo.columns.str.strip()
    geo = geo.rename(columns={
        "Zip Code Tabulation Area Code": "RegionName",
        "Internal Point Latitude": "lat",
        "Internal Point Longitude": "lng",
    })
    # Keep only CO ZIPs (80xxx–81xxx)
    geo["RegionName"] = geo["RegionName"].str.zfill(5)
    geo = geo[geo["RegionName"].str.match(r"^(80|81)\d{3}$")].copy()
    return geo[["RegionName", "lat", "lng"]]

@st.cache_data
def load_mae_from_artifacts(preferred_model: str = "rf") -> pd.DataFrame:

    order = [preferred_model, "xgb" if preferred_model == "rf" else "rf"]
    for m in order:
        path = ART_MAE_RF if m == "rf" else ART_MAE_XGB
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"RegionName": str})
            df["RegionName"] = df["RegionName"].str.zfill(5)
            if "MAE" not in df.columns and "test_mae" in df.columns:
                df = df.rename(columns={"test_mae": "MAE"})
            df["model"] = m
            # keep only relevant columns
            keep = ["RegionName", "MAE", "model"]
            extra = [c for c in df.columns if c not in keep]
            return df[keep + extra] if extra else df[keep]
    return pd.DataFrame(columns=["RegionName", "MAE", "model"])


# Forecast (per ZIP) using pipeline code

def forecast_home_values(
    df_all: pd.DataFrame,
    target_zip: str,
    model_name: str = "rf",
    lags: tuple = DEFAULT_LAGS,
    forecast_months: int = DEFAULT_FORECAST_MONTHS,
):

    df_zip = df_all.loc[df_all["RegionName"] == target_zip, ["date", "price"]].copy()
    if df_zip.empty:
        return None, None, None  # history, forecast, mae_row

    df_zip["date"] = pd.to_datetime(df_zip["date"], errors="coerce")
    df_zip["price"] = pd.to_numeric(df_zip["price"], errors="coerce")
    df_zip = df_zip.dropna(subset=["date", "price"]).sort_values("date").reset_index(drop=True)

    # Build features
    try:
        df_feat = add_lag_features(df_zip, lags=lags, target_col="price", prefix="value")
    except Exception:
        return None, None, None

    if len(df_feat) < 15:
        # too short after lagging to get a stable split/fit
        return None, None, None

    # Train via pipeline
    model, meta = train_one_zip(df_zip, lags=lags, model_name=model_name)

    # History (for plotting)
    history = df_zip[["date", "price"]].rename(columns={"price": "value"}).copy()

    # Recursive forecast
    last = df_feat.iloc[-1]
    # keep three most recent values per lag scheme
    lag_vals = [last[f"value_lag_{k}"] for k in [1, 2, 3]]
    # current target at t (becomes next lag1)
    current = last["price"]

    feature_cols = [f"value_lag_{k}" for k in lags]
    future_dates = pd.date_range(
        start=last["date"] + pd.DateOffset(months=1),
        periods=forecast_months,
        freq="MS",
    )

    preds = []
    # structure lags as (lag1, lag2, lag3) rolling window
    lag1, lag2, lag3 = current, lag_vals[0], lag_vals[1]
    for dt in future_dates:
        x_next = pd.DataFrame([[lag1, lag2, lag3]], columns=feature_cols)
        y_next = float(model.predict(x_next)[0])
        preds.append({"date": dt, "predicted_value": y_next})
        # roll lags
        lag1, lag2, lag3 = y_next, lag1, lag2

    forecast_df = pd.DataFrame(preds)
    mae_val = meta.get("test_mae", np.nan)
    mae_row = {"RegionName": target_zip, "MAE": mae_val, "model": model_name}

    return history, forecast_df, mae_row


# Build heatmap DF

@st.cache_data
def build_mae_geo(mae_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    if mae_df.empty:
        return pd.DataFrame(columns=["RegionName", "MAE", "lat", "lng"])
    df = mae_df.copy()
    df["RegionName"] = df["RegionName"].astype(str).str.zfill(5)
    return df.merge(geo_df, on="RegionName", how="inner")


# UI

st.title("Colorado Home Value Forecast Dashboard")

st.sidebar.header("Controls")
st.sidebar.markdown(
    "**Disclaimer:** Forecasts are estimates based on historical trends. "
    "They do not guarantee future performance and should be used for informational purposes only."
)

# ensure required files exist
missing_files = [p for p in [DATA_PATH, GEO_PATH] if not os.path.exists(p)]
if missing_files:
    st.error(f"Missing data files: {missing_files}")
    st.stop()

# Load core data
df = load_home_values(DATA_PATH)
geo = load_geo(GEO_PATH)

# Choose model for overview metrics (uses pipeline artifacts)
overview_model = st.sidebar.selectbox("Overview model (artifacts)", ["rf", "xgb"], index=0)

with st.spinner("Loading ZIP-level performance..."):
    mae_all = load_mae_from_artifacts(overview_model)
    if mae_all.empty:
        st.info(
            "No pipeline MAE artifacts found. "
            "Run the pipeline to generate 'artifacts/mae_by_zip_*.csv'."
        )
    mae_geo = build_mae_geo(mae_all, geo) if not mae_all.empty else pd.DataFrame()

# Quick metrics
overall_mae = mae_all["MAE"].mean() if not mae_all.empty else np.nan
best_row = mae_all.loc[mae_all["MAE"].idxmin()] if not mae_all.empty else None
worst_row = mae_all.loc[mae_all["MAE"].idxmax()] if not mae_all.empty else None

col1, col2, col3 = st.columns(3)
col1.metric("Overall Mean MAE", f"${overall_mae:,.0f}" if pd.notna(overall_mae) else "N/A")
col2.metric(
    "Best ZIP (lowest MAE)",
    f"{best_row['RegionName']} • ${best_row['MAE']:,.0f}" if best_row is not None else "N/A",
)
col3.metric(
    "Worst ZIP (highest MAE)",
    f"{worst_row['RegionName']} • ${worst_row['MAE']:,.0f}" if worst_row is not None else "N/A",
)

# Tabs
tab_overview, tab_map, tab_metrics = st.tabs(["📈 ZIP Forecast", "🗺️ MAE Heatmap", "📊 Metrics Table"])

#Tab 1: ZIP Forecast
with tab_overview:
    zip_list = sorted(df["RegionName"].unique())
    default_index = zip_list.index("80134") if "80134" in zip_list else 0
    target_zip = st.selectbox("Select a Colorado ZIP code:", zip_list, index=default_index)

    perzip_model = st.radio("Model for this ZIP forecast", ["rf", "xgb"], horizontal=True, index=0)

    history, forecast, mae_row = forecast_home_values(
        df, target_zip, model_name=perzip_model, lags=DEFAULT_LAGS, forecast_months=DEFAULT_FORECAST_MONTHS
    )
    if history is None or forecast is None:
        st.warning("Not enough data for this ZIP (after lagging). Try another ZIP.")
    else:
        # Plot actual vs predicted
        combined = pd.concat(
            [
                history.rename(columns={"value": "price"}).tail(36).assign(type="Actual"),
                forecast.rename(columns={"predicted_value": "price"}).assign(type="Predicted"),
            ],
            ignore_index=True,
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        for label, subset in combined.groupby("type"):
            ax.plot(subset["date"], subset["price"], marker="o", label=label)
        ax.set_title(f"Home Value Forecast for ZIP {target_zip}")
        ax.set_ylabel("Home Value ($)")
        ax.set_xlabel("Date")
        ax.legend()
        plt.xticks(rotation=45, fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        # Forecast table
        st.subheader(f"Predicted Values ({DEFAULT_FORECAST_MONTHS}-Month Outlook)")
        table_df = forecast.copy()
        table_df["date"] = table_df["date"].dt.strftime("%Y-%m")
        table_df["predicted_value"] = table_df["predicted_value"].map(lambda x: f"${x:,.2f}")
        st.dataframe(table_df.rename(columns={"predicted_value": "price"}), use_container_width=True)

        # Per-ZIP MAE (from pipeline meta)
        if mae_row is not None and pd.notna(mae_row.get("MAE", np.nan)):
            st.caption(f"Holdout MAE ({perzip_model}): ${mae_row['MAE']:,.0f}")

#Tab 2: Heatmap
with tab_map:
    st.subheader("ZIP-Level MAE Heatmap (Lower = Better)")
    if mae_geo.empty:
        st.info("No MAE artifact data available to plot.")
    else:
        mae_clip = mae_geo["MAE"].clip(lower=0, upper=np.nanpercentile(mae_geo["MAE"], 95))
        norm = (mae_clip - mae_clip.min()) / (mae_clip.max() - mae_clip.min() + 1e-9)
        mae_geo_vis = mae_geo.copy()
        mae_geo_vis["r"] = (255 * norm).astype(int)
        mae_geo_vis["g"] = (180 * (1 - norm)).astype(int)
        mae_geo_vis["b"] = 0
        mae_geo_vis["a"] = 180

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=mae_geo_vis,
            get_position="[lng, lat]",
            get_fill_color="[r, g, b, a]",
            get_radius=3500,
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=39.55, longitude=-105.70, zoom=6.5, pitch=0)

        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "ZIP: {RegionName}\nMAE: ${MAE}"})
        )

#Tab 3: Metrics
with tab_metrics:
    st.subheader("MAE by ZIP (ascending)")
    if mae_all.empty:
        st.info("No MAE artifact data available.")
    else:
        sorted_mae = mae_all.sort_values("MAE").reset_index(drop=True)
        show = sorted_mae.copy()
        show["MAE"] = show["MAE"].map(lambda x: f"${x:,.0f}")
        st.dataframe(show, use_container_width=True)



        cap = st.sidebar.number_input("Histogram MAE cap ($)", min_value=1000, max_value=200000,
                                      value=100000, step=10000)

        # make sure MAE is numeric
        mae_numeric = pd.to_numeric(mae_all["MAE"], errors="coerce").dropna()

        # cap values at the chosen threshold (so extreme outliers don't stretch bins)
        mae_capped = np.clip(mae_numeric, a_min=0, a_max=cap)

        fig, ax = plt.subplots()
        ax.hist(mae_capped, bins=40)
        ax.set_title(f"MAE Histogram (capped at ${cap:,.0f})")
        ax.set_xlabel("MAE ($)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, cap)  # keep x-axis within the cap
        st.pyplot(fig)


        '''st.subheader("Distribution of MAE Across ZIPs")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.hist(mae_all["MAE"].dropna(), bins=30, edgecolor="white")
        ax2.set_title("MAE Histogram")
        ax2.set_xlabel("MAE ($)")
        ax2.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig2)'''


