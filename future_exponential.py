import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.api import ExponentialSmoothing

@st.cache_resource
def forecast_stock_prices_expsmoothing(data):
    st.markdown("<h3 style='color: cyan;'>M2.1: Future prediction using Exponential Smoothing</h3>", unsafe_allow_html=True)
    st.markdown("`FUTURE PREDICTION PLOT`", unsafe_allow_html=True)
    
    # Ensure the data index is in datetime format
    data.index = pd.to_datetime(data.index)
    
    # Training on 100% data
    train = data.copy()
    
    # Forecast next 30 days
    future_steps = 30
    future_dates = pd.date_range(start=train.index[-1], periods=future_steps + 1, freq='D')[1:]
    
    # Triple Exponential Smoothing Model (Best for seasonality and trend handling)
    model = ExponentialSmoothing(train['Close'], seasonal='add', seasonal_periods=12, trend='add').fit()
    forecast_values = model.forecast(future_steps)
    
    # Prepare Data for Plotting
    last_100_train = train.tail(100)  # Last 100 entries of training data
    
    # Plot the actual vs predicted values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last_100_train.index, y=last_100_train['Close'], mode='lines', name='Last 100 Train Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_values, mode='lines', name='Predicted Next 30 Days', line=dict(color='red', dash='dot')))
    
    fig.update_layout(
        title="Future prediction using Exponential Smoothing",
        xaxis_title='Date',
        yaxis_title='Stock Price',
        xaxis=dict(rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(showline=True, linecolor="white", linewidth=1),
        legend_title='Reference',
    )
    
    st.plotly_chart(fig)
    
    return fig
