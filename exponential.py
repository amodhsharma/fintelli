import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_metric_color(value, thresholds, higher_is_better=True):
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

def forecast_stock_prices_expsmoothing(data):
    data.index = pd.to_datetime(data.index)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    model_single = SimpleExpSmoothing(train['Close']).fit(smoothing_level=0.2, optimized=True)
    forecast_single = model_single.forecast(len(test))
    
    model_double = Holt(train['Close']).fit(smoothing_level=0.2, smoothing_slope=0.1, optimized=True)
    forecast_double = model_double.forecast(len(test))
    
    model_triple = ExponentialSmoothing(train['Close'], seasonal='add', seasonal_periods=12, trend='add').fit()
    forecast_triple = model_triple.forecast(len(test))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Test', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast_single, mode='lines', name='Single Exp Smoothing', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast_double, mode='lines', name='Double Exp Smoothing', line=dict(color='purple', dash='dot')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast_triple, mode='lines', name='Triple Exp Smoothing', line=dict(color='orange', dash='dot')))
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Stock Price', xaxis=dict(rangeslider=dict(visible=True)))
    
    st.subheader('Exponential Smoothing Forecast')
    st.plotly_chart(fig)
    
    evaluate_forecast(test['Close'].values, forecast_triple.values)