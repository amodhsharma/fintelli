import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

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

    predicted_price = predicted.iloc[-1] if not predicted.empty else None
    st.markdown("`CLOSING PRICE PREDECTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="ARIMA", value=f"₹{predicted_price:.2f}" if predicted_price else "N/A")

def forecast_stock_prices_arima(data, order=(6,1,0)):
    data.index = pd.to_datetime(data.index)
    
    # 85-15 train-test split
    train_size = int(len(data) * 0.85)
    train, test = data[:train_size], data[train_size:]
    
    st.markdown("<h3 style='color: cyan;'>M3: ARIMA - Autoregressive Integrated Moving Average", unsafe_allow_html=True),
    #st.title("ARIMA - Autoregressive Integrated Moving Average"),
    #title_font=dict(color="yellow"),
    st.write("ARIMA is a popular statistical method for time series forecasting. It combines autoregression (AR), differencing (I), and moving average (MA) components.")
    st.write("ARIMA models are characterized by three parameters: p, d, and q.")
    st.write("p: The number of lag observations included in the model (lag order).")
    st.write("d: The number of times that the raw observations are differenced (degree of differencing).")
    st.write("q: The size of the moving average window (order of moving average).")
    #st.write("Decomposition plots for Arima")
    #decomposition = seasonal_decompose(train['Close'], model='additive', period=30)
    # fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    # decomposition.trend.plot(ax=axes[0], title='Trend')
    # decomposition.seasonal.plot(ax=axes[1], title='Seasonality')
    # decomposition.resid.plot(ax=axes[2], title='Residuals')
    # st.pyplot(fig)
    
    # st.write("ACF and PACF plots for Arima")
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # plot_acf(train['Close'], ax=axes[0])
    # plot_pacf(train['Close'], ax=axes[1])
    # st.pyplot(fig)
    
    model = ARIMA(train['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Test', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red', dash='dot')))
    
    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)

    fig.update_layout(title="ARIMA", xaxis_title='Date',
        xaxis=dict(title="Date",rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(title='Stock Price', showline=True, linecolor="white", linewidth=1),
        legend_title='Reference',
    )
    #st.subheader('ARIMA Forecast')
    st.plotly_chart(fig)
    
    # Call the evaluation function
    evaluate_forecast(test['Close'].values, forecast)
