import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def get_metric_color(value, thresholds, higher_is_better=True):
    """Returns color based on metric thresholds."""
    if higher_is_better:
        if value >= thresholds[0]:
            return "green"
        elif value >= thresholds[1]:
            return "yellow"
        else:
            return "red"
    else:
        if value <= thresholds[0]:
            return "green"
        elif value <= thresholds[1]:
            return "yellow"
        else:
            return "red"

def evaluate_forecast(actual, predicted):
    """Calculates evaluation metrics with universal thresholds."""
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]  
    predicted = predicted[:min_len]  

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    # Universal Thresholds
    avg_price = np.mean(actual)
    variance = np.var(actual)

    rmse_thresholds = [0.05 * avg_price, 0.10 * avg_price]  # 5% and 10% of avg stock price
    mae_thresholds = [0.05 * avg_price, 0.10 * avg_price]  
    mse_thresholds = [variance, 2 * variance]  # Variance-based scaling
    r2_thresholds = [0.75, 0.50]  # Higher is better

    # Get Colors
    rmse_color = get_metric_color(rmse, rmse_thresholds, higher_is_better=False)
    mae_color = get_metric_color(mae, mae_thresholds, higher_is_better=False)
    mse_color = get_metric_color(mse, mse_thresholds, higher_is_better=False)
    r2_color = get_metric_color(r2, r2_thresholds, higher_is_better=True)

    # Display Metrics in Right-Aligned Format with Colors
    st.markdown("""
    <style>
        .metric-box { 
            text-align: right; 
            font-size: 20px;
            font-weight: bold; 
        }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Evaluation Metrics")
    
    st.markdown(f'<p class="metric-box" style="color:{rmse_color};">RMSE: {rmse:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-box" style="color:{mae_color};">MAE: {mae:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-box" style="color:{mse_color};">MSE: {mse:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-box" style="color:{r2_color};">R² Score: {r2:.4f}</p>', unsafe_allow_html=True)

def forecast_stock_prices(data, period, n_years):
    """Trains a Prophet model and forecasts future stock prices."""

    st.subheader(f'Prophet Forecast for {n_years} Years')

    # Prepare Data for Prophet
    df_train = data[['Close']].rename(columns={"Close": "y"})
    df_train["ds"] = data.index  # Assign index as date

    # Train Prophet Model
    m = Prophet()
    m.fit(df_train)

    # Make Future Predictions
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display Forecast Data
    st.write(forecast.tail())

    # Plot Forecast
    st.write(f'Forecast plot for {n_years} years')
    fig_prophet = plot_plotly(m, forecast)
    st.plotly_chart(fig_prophet)

    # Forecast Components
    st.write("Forecast components for Prophet")
    fig_prophet2 = m.plot_components(forecast)
    st.write(fig_prophet2)

    # Extract Actual vs Predicted for Evaluation
    actual_prices = data["Close"].values  # Actual prices
    predicted_prices = forecast["yhat"].values[-len(actual_prices):]  # Align forecast

    # Evaluate Forecast
    evaluate_forecast(actual_prices, predicted_prices)
