import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st

@st.cache_data
def train_and_forecast_prophet(data):
    st.markdown("<h3 style='color: cyan;'>M7.1:Future prediction using Prophet</h3>", unsafe_allow_html=True)
    st.markdown("`FUTURE PREDICTION PLOT`", unsafe_allow_html=True)

    # Prepare data for Prophet
    data['ds'] = data.index  # Date as 'ds'
    data['y'] = data['Close']  # Closing price as 'y'
    
    # Train the model on the full dataset
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    
    # Create future dataframe for the next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Extract relevant parts of the forecast
    future_dates = forecast['ds'][-30:]
    predicted_prices = forecast['yhat'][-30:]
    
    # Plot actual vs predicted
    fig = go.Figure()
    
    # Plot last 100 actual data points
    fig.add_trace(go.Scatter(x=data['ds'][-100:], y=data['y'][-100:],
                             mode='lines', name='Actual (Last 100)',
                             line=dict(color='blue')))
    
    # Plot forecasted future prices
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices,
                             mode='lines', name='Forecast (Next 30 Days)',
                             line=dict(color='orange', dash='dot')))
    
    # Update layout with range slider
    fig.update_layout(
        title="Future prediction using Prophet",
        xaxis_title="Date",
        yaxis_title="Close Price",
        xaxis=dict(rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(showline=True, linecolor="white", linewidth=1),
        template="plotly_dark",
        legend_title="Legend"
    )
    
    st.plotly_chart(fig)
    
    return forecast[-30:][['ds', 'yhat']]
