import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta

def forecast_next_month_arima(data, order=(6,1,0)):
    data.index = pd.to_datetime(data.index)
    
    st.markdown("<h3 style='color: cyan;'>M3.1: Future prediction using ARIMA</h3>", unsafe_allow_html=True)
    st.markdown("`FUTURE PREDICTION PLOT`", unsafe_allow_html=True)
    
    # Train model on entire dataset
    model = ARIMA(data['Close'], order=order)
    model_fit = model.fit()
    
    # Generate future dates for the next month
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')  # 'B' ensures business days

    # Forecast for next 30 days
    forecast = model_fit.forecast(steps=30)

    # Prepare data for plotting
    historical_data = data['Close'].iloc[-100:]  # Last 100 data points
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data, mode='lines', name='Historical Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast (Next 1 Month)', line=dict(color='red', dash='dot')))

    fig.update_layout(
        title="Future prediction using ARIMA",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(title="Stock Price", showline=True, linecolor="white", linewidth=1),
        legend_title='Reference'
    )
    
    st.plotly_chart(fig)
