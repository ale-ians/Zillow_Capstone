import streamlit as st
import pandas as pd
import pydeck as pdk
from Generate_MAE_by_ZIP import generate_mae_by_zip

st.title("ZIP Code Model Accuracy Heatmap")
st.caption("This map visualizes model error by ZIP code. Lower MAE means more accurate predictions.")

# Generate MAE if not already saved
generate_mae_by_zip(
    home_values_path="data/colorado_home_values.csv",
    geo_data_path="data/us_zip_centroids.csv",
    output_path="data/mae_by_zip.csv"
)

# Load result
mae_df = pd.read_csv("data/mae_by_zip.csv")
mae_df['MAE_str'] = mae_df['MAE'].apply(lambda x: "%.2f" % x)
# Precompute color
mae_df['color'] = mae_df['MAE'].apply(lambda x: [255, max(0, 160 - int(x // 200)), 0, 160])

# Build heatmap layer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=mae_df,
    get_position='[lng, lat]',
    get_fill_color='color',
    get_radius=3000,
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=39.55,
    longitude=-105.70,
    zoom=6.5,
    pitch=0
)

# Show map
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "ZIP: {RegionName}\nMAE: ${MAE_str}"}
))