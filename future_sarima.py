import streamlit as st
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

@st.cache_resource
def sarima_forecast_next_month(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    st.markdown("<h3 style='color: cyan;'>M4.1: Future Prediction using SARIMA</h3>", unsafe_allow_html=True)
    st.markdown("`FUTURE PREDICTION PLOT`", unsafe_allow_html=True)

    # Train model on full dataset
    model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    # Generate future dates for the next month
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')  # 'B' ensures business days

    # Forecast next 30 days
    forecast = model_fit.get_forecast(steps=30).predicted_mean

    # Prepare data for plotting
    historical_data = data['Close'].iloc[-100:]  # Last 100 actual stock prices

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data, mode='lines', name='Historical Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast (Next 1 Month)', line=dict(color='red', dash='dot')))

    fig.update_layout(
        title="Future Prediction using SARIMA",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(title="Stock Price", showline=True, linecolor="white", linewidth=1),
        legend_title='Reference'
    )

    st.plotly_chart(fig)
