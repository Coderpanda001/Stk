import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os

# App title
st.title("📈 Stock Price Predictor App")

# Stock input
stock = st.text_input("Enter the Stock ID (e.g., AAPL, GOOG, MSFT)", "GOOG")

# Load the trained model
model_path = "stock.keras"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please make sure it’s in the same folder.")
    st.stop()

model = load_model(model_path)

# Download data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start, end)

if google_data.empty:
    st.error("No data found for this stock symbol. Try another one.")
    st.stop()

st.subheader("📊 Stock Data")
st.write(google_data)

splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Moving averages
st.subheader('📉 Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('📉 Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('📉 Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('📉 MA for 100 days vs MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Scaling and preparing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Prediction
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Results DataFrame
ploting_data = pd.DataFrame(
    {
        'Original Test Data': inv_y_test.reshape(-1),
        'Predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

st.subheader("📋 Original values vs Predicted values")
st.write(ploting_data)

# Plot original vs predicted
st.subheader('📉 Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data not used", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)
