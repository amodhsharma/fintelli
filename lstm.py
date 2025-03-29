import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define model path inside the project directory
model_path = "/Users/amodhsharma/Desktop/fintelli1/Stock_Predictions_Model.keras"

@st.cache_resource
def load_or_train_model(data_train_scaled):
    """
    Loads the existing LSTM model or trains a new one if not found.
    """
    if os.path.exists(model_path):
        st.write("Loading pre-trained LSTM model...")
        return load_model(model_path)
    else:
        st.write("Training new LSTM model...")

        # Create LSTM Model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(100, 1)),
            LSTM(50, activation='relu', return_sequences=False),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train Model
        model.fit(data_train_scaled[:-100], data_train_scaled[100:], epochs=20, batch_size=32)

        # Save Model
        model.save(model_path)
        st.write(f"Model saved at {model_path}")

        return model

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
    
    st.markdown(f'<p class="metric-box" style="color:{rmse_color}; text-align:right;">RMSE: {rmse:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-box" style="color:{mae_color}; text-align:right;">MAE: {mae:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-box" style="color:{mse_color}; text-align:right;">MSE: {mse:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-box" style="color:{r2_color}; text-align:right;">R² Score: {r2:.4f}</p>', unsafe_allow_html=True)

def forecast_lstm(data, ticker):
    st.subheader(f'LSTM Forecast for {ticker}')

    # Split Data
    data_train = data[['Close']].iloc[:int(len(data) * 0.80)]
    data_test = data[['Close']].iloc[int(len(data) * 0.80):]

    # Scaling Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scaled = scaler.fit_transform(data_train)
    
    # Prepare Test Data
    past_100_days = data_train.tail(100)
    data_test_full = pd.concat([past_100_days, data_test])
    data_test_scaled = scaler.transform(data_test_full)

    # Create Sequences
    x_test, y_test = [], []
    for i in range(100, data_test_scaled.shape[0]):
        x_test.append(data_test_scaled[i-100:i])
        y_test.append(data_test_scaled[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Load or Train Model
    model = load_or_train_model(data_train_scaled)

    # Predict Prices
    predictions = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    predictions = predictions * scale_factor
    y_test = y_test * scale_factor

    # Use actual dates from index
    prediction_dates = data.index[-len(predictions):]

    # Plot Results with Range Slider
    fig = go.Figure()

    # Add actual prices
    fig.add_trace(go.Scatter(
        x=prediction_dates, 
        y=y_test, 
        mode='lines', 
        name='Actual Price', 
        line=dict(color='green')
    ))

    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=prediction_dates, 
        y=predictions.flatten(), 
        mode='lines', 
        name='Predicted Price', 
        line=dict(color="cyan", dash="dot")
    ))

    # Configure Range Slider
    fig.update_layout(
        xaxis=dict(
            title="Date",
            type="date",
            rangeslider=dict(visible=True),  # Enable range slider
            showline=True,  
            linewidth=1  
        ),
        yaxis=dict(
            title="Stock Price",
            showline=True,  
            linewidth=1  
        )
    )

    # Show the interactive plot
    st.plotly_chart(fig, use_container_width=True)

    # **Fix: Match actual & predicted lengths before evaluation**
    actual_prices = data_test["Close"].values  # Get actual prices as a NumPy array
    evaluate_forecast(actual_prices, predictions.flatten())
