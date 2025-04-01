import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_resource
def train_and_forecast_prophet(data):
    # Ensure dataset is valid
    if data.empty:
        raise ValueError("Dataset is empty. Please provide valid stock data.")

    # Reset index to access 'Date' column
    data = data.reset_index()

    # Ensure Date is in datetime format
    data["Date"] = pd.to_datetime(data["Date"])

    # Select required features
    data = data.rename(columns={"Date": "ds", "Close": "y"})

    # Split dataset (85% train, 15% test)
    train_size = int(len(data) * 0.85)
    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size:]

    # Train the model
    model = Prophet(daily_seasonality=True)
    model.fit(train_df)

    # Predict future values (last 15% test range)
    future = test_df[["ds"]]  # Use only test dates
    forecast = model.predict(future)

    # Extract actual vs. predicted
    predicted = forecast[["ds", "yhat"]]
    actual = test_df[["ds", "y"]]

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(actual["y"], predicted["yhat"]))
    mae = mean_absolute_error(actual["y"], predicted["yhat"])
    mse = mean_squared_error(actual["y"], predicted["yhat"])
    mape = mean_absolute_percentage_error(actual["y"], predicted["yhat"])
    r2 = r2_score(actual["y"], predicted["yhat"])

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual["ds"], y=actual["y"], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predicted["ds"], y=predicted["yhat"], mode='lines', name='Predicted', line=dict(color='orange')))
    fig.update_layout(title="Tesla Stock Price Prediction (Prophet)", xaxis_title="Date", yaxis_title="Close Price", template="plotly_dark")

    # Prophet components
    #trend_fig = model.plot_components(forecast)

    # Streamlit UI
    st.title("Tesla Stock Price Prediction using Prophet")
    st.plotly_chart(fig)
    #st.pyplot(trend_fig)

    # Display evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**MAPE:** {mape:.2f}%")
    st.write(f"**R² Score:** {r2:.4f}")

    return fig, rmse, mae, mse, mape, r2
    #return trend_fig
