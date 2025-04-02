import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cache the function so that it's not recomputed each time (for performance)
@st.cache_data
def evaluate_forecast(actual, predicted):
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    # Calculating MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "R^2": r2,
        "MAPE": mape
    }

    return metrics

@st.cache_resource
def forecast_stock_prices_expsmoothing(data):
    st.markdown("<h3 style='color: cyan;'>M2: Exponential Smoothing", unsafe_allow_html=True),
    #st.title('Exponential Smoothing')
    st.write("This section provides a detailed analysis of stock price forecasting using Exponential Smoothing methods.")
    # Ensure the data index is in datetime format
    data.index = pd.to_datetime(data.index)

    # Splitting data into train and test (85% train, 15% test)
    train_size = int(len(data) * 0.85)
    train, test = data[:train_size], data[train_size:]

    # Simple Exponential Smoothing (Single)
    model_single = SimpleExpSmoothing(train['Close']).fit(smoothing_level=0.2, optimized=True)
    forecast_single = model_single.forecast(len(test))

    # Double Exponential Smoothing (Holt)
    model_double = Holt(train['Close']).fit(smoothing_level=0.2, smoothing_slope=0.1, optimized=True)
    forecast_double = model_double.forecast(len(test))

    # Triple Exponential Smoothing (ExponentialSmoothing)
    model_triple = ExponentialSmoothing(train['Close'], seasonal='add', seasonal_periods=12, trend='add').fit()
    forecast_triple = model_triple.forecast(len(test))

    # Plot the actual vs predicted values for all three methods
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Test', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast_single, mode='lines', name='Single Exp Smoothing', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast_double, mode='lines', name='Double Exp Smoothing', line=dict(color='purple', dash='dot')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast_triple, mode='lines', name='Triple Exp Smoothing', line=dict(color='orange', dash='dot')))

    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)

    fig.update_layout(title="Exponential Smoothening",xaxis_title='Date', yaxis_title='Stock Price',
        xaxis=dict(rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(showline=True, linecolor="white", linewidth=1),
        legend_title='Reference',
    )

    st.plotly_chart(fig)

    # Evaluate all three forecasts
    metrics_single = evaluate_forecast(test['Close'].values, forecast_single)
    metrics_double = evaluate_forecast(test['Close'].values, forecast_double)
    metrics_triple = evaluate_forecast(test['Close'].values, forecast_triple)

    blue_text = "color: #3498DB;"

    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)

    # Display Evaluation Metrics for each model (without formatting)
    st.subheader("Evaluation Metrics for Single Exponential Smoothing")
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs. <span style='{blue_text}'>{metrics_single['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics_single['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics_single['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The R² value of <span style='{blue_text}'>{metrics_single['R^2']:.2f}</span> indicates that the model explains <span style='{blue_text}'>{metrics_single['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    st.subheader("Evaluation Metrics for Double Exponential Smoothing")
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs. <span style='{blue_text}'>{metrics_double['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics_double['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics_double['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The R² value of <span style='{blue_text}'>{metrics_double['R^2']:.2f}</span> indicates that the model explains <span style='{blue_text}'>{metrics_double['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    st.subheader("Evaluation Metrics for Triple Exponential Smoothing")
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs. <span style='{blue_text}'>{metrics_triple['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics_triple['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics_triple['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The R² value of <span style='{blue_text}'>{metrics_triple['R^2']:.2f}</span> indicates that the model explains <span style='{blue_text}'>{metrics_triple['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    return fig, metrics_single, metrics_double, metrics_triple
