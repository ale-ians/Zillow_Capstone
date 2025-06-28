import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from forecast_model import forecast_home_values

#load data
@st.cache_data
def load_data():
    return pd.read_csv('data/colorado_home_values.csv', parse_dates=['date'])

st.title('Colorado Home Price Forecast Dashboard')

df = load_data()

#show ZIP options
zip_list = sorted(df['RegionName'].unique())
target_zip = st.sidebar.selectbox('Select a Colorado Zip Code:', zip_list)

#Forecast
history, forecast = forecast_home_values(df, target_zip, forecast_months=6)

#handle missing or empty forecast
if history is None or forecast is None or history.empty or forecast.empty:
    st.warning(f"No data available for ZIP code {target_zip}. Please try a different ZIP.")
    st.stop()

forecast = forecast.rename(columns={'predicted_value': 'price'})
forecast['type'] = 'Predicted'

history = history.rename(columns={'value': 'price'})
history['type'] = 'Actual'

combined = pd.concat([history.tail(12), forecast], ignore_index=True)

#Plot
fig, ax = plt.subplots()
for label, data in combined.groupby("type"):
    ax.plot(data['date'], data['price'], marker='o', label=label)

ax.set_title(f'Home Value Forecast for ZIP Code {target_zip}')
ax.set_ylabel('Home Value ($)')
ax.set_xlabel('Date')
ax.legend()
st.pyplot(fig)

#Display forecast table
display_df = combined.copy()
display_df['price'] = display_df['price'].map(lambda x: f"${x:,.2f}")
display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%m-%Y')

st.subheader('Predicted Values')
st.dataframe(display_df[display_df['type'] == 'Predicted'][['date', 'price']].reset_index(drop=True))