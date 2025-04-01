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

# Load data
data = pd.read_csv("stock_data.csv")

# ----------------------------- Creates a new LSTM dataset from the stock data, excluding unnecessary columns and normalizing it. ---------------------------- #

def create_lstm_data(data):
    lstm_data = data.copy()     # Copy the original data to avoid modifying it
    lstm_data = lstm_data.drop(columns=['Volatility_30', 'MA_30', 'MA_60', 'MA_90', 'Volatility_90'])   # Drop unnecessary columns for LSTM processing
    lstm_data.index = pd.to_datetime(lstm_data.index)   # Set the index as datetime

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    lstm_data_scaled = scaler.fit_transform(lstm_data[['Close']])
    
    return lstm_data, scaler

# ----------------------------- Loads the existing LSTM model or trains a new one if not found. ---------------------------- #

@st.cache_resource
def load_or_train_model(data_train_scaled):
    if os.path.exists(model_path):
        st.write("Loading pre-trained LSTM model...")
        return load_model(model_path)
    else:
        st.write("Training new LSTM model...")

        model = Sequential([    # Create LSTM Model
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
        return model

# ----------------------------- Calculates evaluation metrics without formatting. ---------------------------- #

def evaluate_forecast(actual, predicted):
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]  
    predicted = predicted[:min_len]  

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    # Display Metrics
    st.subheader("Evaluation Metrics")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"R² Score: {r2:.4f}")

# ----------------------------- Calculates forcasting based on previosu 85% data ---------------------------- #

def forecast_lstm(data, ticker):
    st.subheader(f'LSTM Forecast for {ticker}')

    # Create LSTM Data
    lstm_data, scaler = create_lstm_data(data)

    # Split Data
    data_train = lstm_data[['Close']].iloc[:int(len(lstm_data) * 0.85)]  # 85% training data
    data_test = lstm_data[['Close']].iloc[int(len(lstm_data) * 0.85):]   # 15% test data

    # Scaling Data
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
    fig.add_trace(go.Scatter(x=data_test.index, y=y_test, mode='lines', name='Actual Price (15%)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Predicted Price (from 85% model)', line=dict(color="orange", dash="dot")))
    fig.update_layout(
        xaxis=dict(title="Date",type="date",rangeslider=dict(visible=True),showline=True, linewidth=1),
        yaxis=dict(title="Stock Price",showline=True,linewidth=1)
    )

    # Show the interactive plot
    st.plotly_chart(fig)

    # **Fix: Match actual & predicted lengths before evaluation**
    actual_prices = y_test  # Get actual prices as a NumPy array
    evaluate_forecast(actual_prices, predictions.flatten())
