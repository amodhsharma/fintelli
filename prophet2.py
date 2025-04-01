#works

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_resource
def train_and_forecast_prophet(prophet_data):
    # Ensure the dataset is not empty
    if prophet_data.empty:
        raise ValueError("Input DataFrame is empty. Please provide valid stock data.")

    # Reset index if it's a datetime index
    if isinstance(prophet_data.index, pd.DatetimeIndex):
        prophet_data = prophet_data.reset_index()  # Convert index to column

    # Identify potential datetime and numeric columns
    date_cols = prophet_data.select_dtypes(include=['datetime', 'object']).columns
    num_cols = prophet_data.select_dtypes(include=['number']).columns

    # Validate column existence
    if len(date_cols) == 0:
        raise ValueError("No datetime column found in the dataset (Check if the index is correctly set).")
    if len(num_cols) == 0:
        raise ValueError("No numerical target column found in the dataset.")

    # Use the first datetime and first numerical column
    date_col = date_cols[0]
    target_col = num_cols[0]

    # Create a copy with only necessary columns
    prophet_data = prophet_data[[date_col, target_col]].copy()
    prophet_data.columns = ['ds', 'y']  # Rename for Prophet

    # Convert date column to datetime format
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])

    # Ensure there are enough data points for training
    if len(prophet_data) < 10:
        raise ValueError("Not enough data points for model training. Provide more historical data.")

    # Splitting data
    train_size = int(len(prophet_data) * 0.85)
    train_df = prophet_data.iloc[:train_size]
    test_df = prophet_data.iloc[train_size:]

    # Train Prophet model
    model = Prophet()
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    model.fit(train_df)

    # Forecast future values
    future = test_df[['ds']]
    forecast = model.predict(future)

    # Extract predicted values
    predicted = forecast[['ds', 'yhat']]
    actual = test_df[['ds', 'y']]

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(actual['y'], predicted['yhat']))
    mae = mean_absolute_error(actual['y'], predicted['yhat'])
    mse = mean_squared_error(actual['y'], predicted['yhat'])
    mape = mean_absolute_percentage_error(actual['y'], predicted['yhat'])
    r2 = r2_score(actual['y'], predicted['yhat'])

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual['ds'], y=actual['y'], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predicted['ds'], y=predicted['yhat'], mode='lines', name='Predicted', line=dict(color='orange')))
    fig.update_layout(title='Prophet Model Forecast', xaxis_title='Date', yaxis_title='Stock Price', template='plotly_dark')

    # Streamlit UI
    st.title("Stock Price Prediction with Prophet")
    st.plotly_chart(fig)

    # Display evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**MAPE:** {mape:.2f}%")
    st.write(f"**R² Score:** {r2:.4f}")

    return fig, rmse, mae, mse, mape, r2
