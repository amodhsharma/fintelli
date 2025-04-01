#dependencies
#explanation of model 

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def forecast_stock_prices_linear(data, period, start_date, end_date):
    df_train = data[['Close']].rename(columns={"Close": "y"})
    df_train["ds"] = np.arange(len(df_train))  # Convert index to numerical values

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(df_train[["ds"]], df_train["y"])

    # Extend future dates from the last date in data to TODAY
    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    future_ds = np.arange(total_days).reshape(-1, 1)

    # Predict stock prices for all future dates
    forecast_y = model.predict(future_ds)

    # Generate proper date range from START to TODAY
    future_dates = pd.date_range(start=start_date, periods=len(future_ds), freq='D')

    # Plot Results
    fig = go.Figure()

    # Plot Actual Prices (using index directly)
    fig.add_trace(go.Scatter(
        x=data.index,  
        y=data["Close"], 
        mode='lines', 
        name="Actual Price", 
        line=dict(color="green")
    ))

    # Plot Forecasted Prices
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=forecast_y, 
        mode='lines', 
        name="Forecasted Price", 
        line=dict(color="cyan", dash="dot")
    ))

    st.subheader("Linear Regression Forecast")
    st.markdown("""
<div style="font-size:14px;">
    This model predicts future stock prices by fitting a straight line to historical data.  
    It minimizes the difference between actual and predicted values.  
    A lower <u>MSE</u>/<u>RMSE</u>/<u>MAE</u> indicates better accuracy, while the <u>R² Score</u> shows how well the model explains price variations.  
    Use this insight to gauge market trends and potential price movements.  
</div>
""", unsafe_allow_html=True)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis=dict(rangeslider=dict(visible=True)),
    )

    st.plotly_chart(fig)

    # **Fix: Match actual & predicted lengths before evaluation**
    actual_prices = data["Close"].values  # Get actual prices as a NumPy array
    evaluate_forecast(actual_prices, forecast_y)
