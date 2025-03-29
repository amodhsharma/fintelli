import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
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
    """Calculates evaluation metrics with color-coded display."""
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]  
    predicted = predicted[:min_len]  

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    avg_price = np.mean(actual)
    variance = np.var(actual)

    rmse_thresholds = [0.05 * avg_price, 0.10 * avg_price]
    mae_thresholds = [0.05 * avg_price, 0.10 * avg_price]  
    mse_thresholds = [variance, 2 * variance]
    r2_thresholds = [0.75, 0.50]

    rmse_color = get_metric_color(rmse, rmse_thresholds, higher_is_better=False)
    mae_color = get_metric_color(mae, mae_thresholds, higher_is_better=False)
    mse_color = get_metric_color(mse, mse_thresholds, higher_is_better=False)
    r2_color = get_metric_color(r2, r2_thresholds, higher_is_better=True)

    st.subheader("Evaluation Metrics")
    st.markdown(f'<p style="color:{rmse_color}; text-align:right;">RMSE: {rmse:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{mae_color}; text-align:right;">MAE: {mae:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{mse_color}; text-align:right;">MSE: {mse:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{r2_color}; text-align:right;">R² Score: {r2:.4f}</p>', unsafe_allow_html=True)

def forecast_stock_prices_arima(data, order=(6,1,0)):
    data.index = pd.to_datetime(data.index)
    
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    st.subheader("ARIMA - Autoregressive Integrated Moving Average")
    st.write("Decomposition plots for Arima")
    decomposition = seasonal_decompose(train['Close'], model='additive', period=30)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    decomposition.trend.plot(ax=axes[0], title='Trend')
    decomposition.seasonal.plot(ax=axes[1], title='Seasonality')
    decomposition.resid.plot(ax=axes[2], title='Residuals')
    st.pyplot(fig)
    
    st.write("ACF and PACF plots for Arima")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train['Close'], ax=axes[0])
    plot_pacf(train['Close'], ax=axes[1])
    st.pyplot(fig)
    
    model = ARIMA(train['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Test', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red', dash='dot')))
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Stock Price', xaxis=dict(rangeslider=dict(visible=True)))
    
    st.subheader('ARIMA Forecast')
    st.plotly_chart(fig)
    
    evaluate_forecast(test['Close'].values, forecast.values)