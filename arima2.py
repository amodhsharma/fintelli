import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import streamlit as st

# Function to calculate MAPE
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Function to calculate evaluation metrics
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = calculate_mape(actual, predicted)
    
    return mse, rmse, mae, mape, r2

# Function to create ARIMA model, make predictions, and display results
def arima_model(data):
    # Ensure the data is numeric
    data = pd.to_numeric(data, errors='coerce')  # Convert to numeric and handle non-numeric gracefully
    data = data.dropna()  # Drop any NaN values that might appear after coercion

    # Ensure that the data is a 1D array or Series (not DataFrame)
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]  # If it's a DataFrame, take the first column
    
    # Step 1: Split the data into training and testing parts
    train_size = int(len(data) * 0.85)  # 85% data for training
    train, test = data[:train_size], data[train_size:]
    
    # Step 2: Fit ARIMA model
    model = ARIMA(train, order=(5, 1, 0))  # ARIMA(p,d,q) parameters can be adjusted
    model_fit = model.fit()

    # Step 3: Make predictions on the test data
    predicted = model_fit.forecast(steps=len(test))[0]  # Forecast next 'n' values, where 'n' is the size of test data
    
    # Step 4: Calculate evaluation metrics
    mse, rmse, mae, mape, r2 = calculate_metrics(test, predicted)
    
    # Step 5: Plot the results
    # Create traces for the plot
    trace_actual = go.Scatter(x=np.arange(len(train), len(train) + len(test)), y=test, mode='lines', name='Actual', line=dict(color='blue'))
    trace_predicted = go.Scatter(x=np.arange(len(train), len(train) + len(test)), y=predicted, mode='lines', name='Predicted', line=dict(color='orange'))

    # Create layout for the plot
    layout = go.Layout(
        title='ARIMA Model Prediction vs Actual',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Value'),
        showlegend=True
    )

    # Create the figure
    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Display the evaluation metrics in Streamlit
    st.subheader("Evaluation Metrics")
    st.write(f"**MSE:** {mse}")
    st.write(f"**RMSE:** {rmse}")
    st.write(f"**MAE:** {mae}")
    st.write(f"**MAPE:** {mape}%")
    st.write(f"**R²:** {r2}")
