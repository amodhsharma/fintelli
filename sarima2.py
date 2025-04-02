import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

@st.cache_data
def evaluate_forecast(actual, predicted):
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]  
    predicted = predicted[:min_len]  

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100  # in percentage
    r2 = r2_score(actual, predicted)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R^2": r2
    }

    blue_text = "color: #3498DB;"

    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)

    st.subheader("Evaluation Metrics")
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

@st.cache_resource
def sarima_forecast(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    st.markdown("<h3 style='color: cyan;'>M4: SARIMA - Seasonal Autoregressive Integrated Moving Average", unsafe_allow_html=True),
    #st.title("SARIMA - Seasonal Autoregressive Integrated Moving Average"),
    #title_font=dict(color="yellow"),
    st.write("SARIMA is an extension of ARIMA that supports seasonal differencing. It is particularly useful for time series data with seasonal patterns.")
    
    # Ensure stationarity with differencing if necessary
    if data['Close'].diff().dropna().var() > data['Close'].var() * 0.01:
        data['Close'] = data['Close'].diff().dropna()
    
    # 85-15 train-test split
    train_size = int(len(data) * 0.85)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    
    model = SARIMAX(train['Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=len(test)).predicted_mean
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Test Data', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red', dash='dot')))
    
    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)

    fig.update_layout(title="SARIMA",
        xaxis=dict(title="Date",rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(title="Stock Price", showline=True, linecolor="white", linewidth=1),
        legend_title="Reference",
    )
    st.plotly_chart(fig)
    
    # Call the evaluation function
    evaluate_forecast(test['Close'].values, forecast.values)
