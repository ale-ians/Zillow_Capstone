import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pydeck as pdk

# Config & helpers

st.set_page_config(page_title="CO Home Value Forecast", layout="wide")

DATA_PATH = "data/processed/colorado_home_values.csv"
GEO_PATH = "data/external/us_zip_centroids.csv"

# Data loaders (cached)


@st.cache_data
def load_home_values(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["RegionName"] = df["RegionName"].astype(str).str.zfill(5)
    df = df.sort_values(["RegionName", "date"])
    return df

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
    geo = geo[geo["RegionName"].str.match(r"^(80|81)\d{3}$")].copy()
    return geo[["RegionName", "lat", "lng"]]


# Forecast function (per ZIP)
#   - Lags 1–3
#   - Last 12 months = test for MAE
#   - 6-month rolling forecast


def forecast_home_values(df_all: pd.DataFrame, target_zip: str, forecast_months: int = 6):
    df_zip = df_all[df_all["RegionName"] == target_zip].sort_values("date").copy()
    if df_zip.empty:
        return None, None, None  # history, forecast, mae_row

    # Create lag features
    for lag in [1, 2, 3]:
        df_zip[f"value_lag_{lag}"] = df_zip["price"].shift(lag)
    df_zip = df_zip.dropna().reset_index(drop=True)

    if len(df_zip) < 24:
        return None, None, None

    # Train-test split (time-ordered)
    feature_cols = [f"value_lag_{lag}" for lag in [1, 2, 3]]
    train = df_zip.iloc[:-12]
    test = df_zip.iloc[-12:]

    X_train, y_train = train[feature_cols], train["price"]
    X_test, y_test = test[feature_cols], test["price"]

    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # In-sample test prediction for MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Build history for plotting
    history = df_zip[["date", "price"]].rename(columns={"price": "value"}).copy()

    # Rolling 6‑month forecast using last known lags
    last_row = df_zip.iloc[-1]
    lag1, lag2, lag3 = last_row["price"], last_row["value_lag_1"], last_row["value_lag_2"]

    future_dates = pd.date_range(
        start=last_row["date"] + pd.DateOffset(months=1),
        periods=forecast_months,
        freq="MS"
    )

    forecasts = []
    for dt in future_dates:
        X_next = pd.DataFrame([[lag1, lag2, lag3]], columns=feature_cols)
        y_next = float(model.predict(X_next)[0])
        forecasts.append({"date": dt, "predicted_value": y_next})
        # roll lags
        lag1, lag2, lag3 = y_next, lag1, lag2

    forecast_df = pd.DataFrame(forecasts)
    mae_row = {"RegionName": target_zip, "MAE": mae}
    return history, forecast_df, mae_row


# Compute MAE for ALL ZIPs (once, cached)


@st.cache_data
def compute_mae_by_zip(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for z in sorted(df_all["RegionName"].unique()):
        h, f, m = forecast_home_values(df_all, z, forecast_months=6)
        if m is not None:
            rows.append(m)
    mae_df = pd.DataFrame(rows)
    return mae_df


# Build heatmap DataFrame (merge MAE with lat/lng)


@st.cache_data
def build_mae_geo(mae_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    mae_df["RegionName"] = mae_df["RegionName"].astype(str).str.zfill(5)
    return mae_df.merge(geo_df, on="RegionName", how="inner")


# UI


st.title("Colorado Home Value Forecast Dashboard")

# Sidebar
st.sidebar.header("Controls")
st.sidebar.markdown(
    "**Disclaimer:** Forecasts are estimates based on historical trends. "
    "They do not guarantee future performance and should be used for informational purposes only."
)

# Load data
if not (os.path.exists(DATA_PATH) and os.path.exists(GEO_PATH)):
    st.error("Missing data files. Ensure `data/colorado_home_values.csv` and `data/us_zip_centroids.csv` exist.")
    st.stop()

df = load_home_values(DATA_PATH)
geo = load_geo(GEO_PATH)

# Precompute MAE across ZIPs for overview & map
with st.spinner("Computing ZIP‑level performance..."):
    mae_all = compute_mae_by_zip(df)
    mae_geo = build_mae_geo(mae_all, geo)

# Quick metrics
overall_mae = mae_all["MAE"].mean() if not mae_all.empty else np.nan
best_row = mae_all.loc[mae_all["MAE"].idxmin()] if not mae_all.empty else None
worst_row = mae_all.loc[mae_all["MAE"].idxmax()] if not mae_all.empty else None

col1, col2, col3 = st.columns(3)
col1.metric("Overall Mean MAE", f"${overall_mae:,.0f}" if pd.notna(overall_mae) else "N/A")
col2.metric("Best ZIP (lowest MAE)",
            f"{best_row['RegionName']} • ${best_row['MAE']:,.0f}" if best_row is not None else "N/A")
col3.metric("Worst ZIP (highest MAE)",
            f"{worst_row['RegionName']} • ${worst_row['MAE']:,.0f}" if worst_row is not None else "N/A")

# Tabs
tab_overview, tab_map, tab_metrics = st.tabs(["📈 ZIP Forecast", "🗺️ MAE Heatmap", "📊 Metrics Table"])


# Tab 1: ZIP Forecast


with tab_overview:
    zip_list = sorted(df["RegionName"].unique())
    target_zip = st.selectbox("Select a Colorado ZIP code:", zip_list, index=zip_list.index("80134") if "80134" in zip_list else 0)

    history, forecast, mae_row = forecast_home_values(df, target_zip, forecast_months=6)
    if history is None:
        st.warning("Not enough data for this ZIP. Try another.")
    else:
        # Plot actual vs. predicted
        combined = pd.concat([
            history.rename(columns={"value": "price"}).tail(36).assign(type="Actual"),
            forecast.rename(columns={"predicted_value": "price"}).assign(type="Predicted")
        ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        for label, subset in combined.groupby("type"):
            ax.plot(subset["date"], subset["price"], marker='o', label=label)
        ax.set_title(f"Home Value Forecast for ZIP {target_zip}")
        ax.set_ylabel("Home Value ($)")
        ax.set_xlabel("Date")
        ax.legend()
        plt.xticks(rotation=45, fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        # Forecast table
        st.subheader("Predicted Values (6‑Month Outlook)")
        table_df = forecast.copy()
        table_df["date"] = table_df["date"].dt.strftime("%Y-%m")
        table_df["predicted_value"] = table_df["predicted_value"].map(lambda x: f"${x:,.2f}")
        st.dataframe(table_df.rename(columns={"predicted_value": "price"}), use_container_width=True)


# Tab 2: Heatmap

with tab_map:
    st.subheader("ZIP‑Level MAE Heatmap (Lower = Better)")
    if mae_geo.empty:
        st.info("No MAE data available to plot.")
    else:
        # Color mapping: lower MAE -> greener, higher -> redder
        # normalize MAE for color scaling
        mae_clip = mae_geo["MAE"].clip(lower=0, upper=np.nanpercentile(mae_geo["MAE"], 95))
        norm = (mae_clip - mae_clip.min()) / (mae_clip.max() - mae_clip.min() + 1e-9)
        # Create color columns (R,G,B,alpha)
        mae_geo = mae_geo.copy()
        mae_geo["r"] = (255 * norm).astype(int)
        mae_geo["g"] = (180 * (1 - norm)).astype(int)
        mae_geo["b"] = 0
        mae_geo["a"] = 180

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=mae_geo,
            get_position="[lng, lat]",
            get_fill_color="[r, g, b, a]",
            get_radius=3500,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=39.55,
            longitude=-105.70,
            zoom=6.5,
            pitch=0,
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "ZIP: {RegionName}\nMAE: ${MAE}"}
        ))


# Tab 3: Metrics Table & Histogram


with tab_metrics:
    st.subheader("MAE by ZIP (ascending)")
    if mae_all.empty:
        st.info("No MAE data available.")
    else:
        sorted_mae = mae_all.sort_values("MAE").reset_index(drop=True)
        show = sorted_mae.copy()
        show["MAE"] = show["MAE"].map(lambda x: f"${x:,.0f}")
        st.dataframe(show, use_container_width=True)

        st.subheader("Distribution of MAE Across ZIPs")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.hist(mae_all["MAE"], bins=30, edgecolor="white")
        ax2.set_title("MAE Histogram")
        ax2.set_xlabel("MAE ($)")
        ax2.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig2)
