import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Function to preprocess data
@st.cache_data
def preprocess_data(xg_data):
    xg_data = xg_data.copy()
    xg_data.index = pd.to_datetime(xg_data.index)
    xg_data['day_of_week'] = xg_data.index.dayofweek
    xg_data['month'] = xg_data.index.month
    xg_data['year'] = xg_data.index.year
    xg_data['day'] = xg_data.index.day
    return xg_data

# Train XGBoost and Predict Next Month
@st.cache_resource
def train_xgboost_forecast(xg_data, target_column='Close'):
    xg_data = preprocess_data(xg_data)
    
    # Define features & target
    X, y = xg_data.drop(columns=[target_column]), xg_data[target_column]
    
    # Train XGBoost model on full data
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    
    # Generate future dates
    last_date = xg_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
    
    # Create future feature set with the same columns as training data
    future_data = pd.DataFrame(index=future_dates)
    future_data['day_of_week'] = future_data.index.dayofweek
    future_data['month'] = future_data.index.month
    future_data['year'] = future_data.index.year
    future_data['day'] = future_data.index.day
    
    # Copy last known values for missing features
    missing_features = [col for col in X.columns if col not in future_data.columns]
    for feature in missing_features:
        future_data[feature] = X[feature].iloc[-1]  # Use last known value
    
    # Predict future prices
    future_predictions = model.predict(future_data[X.columns])  # Ensure correct feature order
    
    return y, future_dates, future_predictions

# Plot actual vs predicted future prices
def plot_future_predictions(y, future_dates, future_predictions):
    fig = go.Figure()
    
    # Last 100 training data points
    y_last_100 = y[-100:]
    fig.add_trace(go.Scatter(x=y_last_100.index, y=y_last_100, mode='lines', name='Last 100 Actual', line=dict(color='blue')))
    
    # Future predictions
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Predicted Future', line=dict(color='red', dash='dash')))
    
    fig.update_layout(
        title='Future Prediction using XG Boost',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        xaxis=dict(rangeslider=dict(visible=True)),
        legend_title='Reference'
    )
    
    return fig

# Run Streamlit app
def run_xgboost_forecast(xg_data, target_column='Close'):
    st.markdown("<h3 style='color: cyan;'>M6.1: Future Prediction using XG Boost</h3>", unsafe_allow_html=True)
    st.markdown("`FUTURE PREDICTION PLOT`", unsafe_allow_html=True)
    
    if xg_data is None or xg_data.empty:
        st.error('Error: No data available for processing!')
        return
    
    y, future_dates, future_predictions = train_xgboost_forecast(xg_data, target_column)
    fig = plot_future_predictions(y, future_dates, future_predictions)
    
    # Plot results
    st.plotly_chart(fig)
    