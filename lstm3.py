import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def prepare_data(lstm_data):\
    # Set the index to the Date column for plotting
    lstm_data.index = pd.to_datetime(lstm_data.index)
    
    # Define the feature and target columns
    target_column = 'Close'  # Assuming the target column is 'Close'
    
    # Extract features and target
    data = lstm_data[[target_column]].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Split the data into training (85%) and testing (15%)
    train_size = int(len(scaled_data) * 0.85)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    return train_data, test_data, scaler

@st.cache_data
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(1, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@st.cache_data
def train_lstm_model(train_data, model):
    st.title("LSTM Model")
    st.write("LSTM is a type of Recurrent Neural Network (RNN) designed to capture long-term dependencies in sequential data. It is particularly useful for tasks like time series forecasting, natural language processing, and stock market prediction.")

    X_train, y_train = [], []
    
    for i in range(1, len(train_data)):
        X_train.append(train_data[i-1:i, 0])
        y_train.append(train_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    return model

@st.cache_data
def predict_lstm(test_data, model, _scaler):
    X_test, y_test = [], []
    
    for i in range(1, len(test_data)):
        X_test.append(test_data[i-1:i, 0])
        y_test.append(test_data[i, 0])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted = model.predict(X_test)
    
    # Inverse transform both predicted and actual values
    predicted = _scaler.inverse_transform(predicted)
    y_test = _scaler.inverse_transform(y_test.reshape(-1, 1))  # FIX: Transform actual values
    
    return predicted, y_test.flatten()  # Ensure actual values have the right shape


def plot_results(actual, predicted, lstm_data, target_column='Close'):
    # Get the index (dates) for plotting
    dates = lstm_data.index[-len(actual):]
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add the actual data
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual', line=dict(color='blue')))
    
    # Add the predicted data
    fig.add_trace(go.Scatter(x=dates, y=predicted.flatten(), mode='lines', name='Predicted', line=dict(color='orange')))
    
    fig.update_layout(title='Forecasting Model for LSTM',
                      xaxis=dict(title="Date", rangeslider=dict(visible=True)),
                      yaxis_title=target_column,
                      legend_title='Reference')
    
    # Show the plot in Streamlit
    st.plotly_chart(fig)

@st.cache_data
def evaluate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R^2': r2
    }
    
    return metrics

def lstm(lstm_data):
    # Prepare data
    train_data, test_data, scaler = prepare_data(lstm_data)
    
    # Create and train the LSTM model
    model = create_lstm_model()
    model = train_lstm_model(train_data, model)
    
    # Make predictions on the test set
    predicted, actual = predict_lstm(test_data, model, scaler)
    
    # Plot the results
    plot_results(actual, predicted, lstm_data)
    
    # Evaluate the metrics
    metrics = evaluate_metrics(actual, predicted)
    
    # Display the metrics
    st.subheader("Evaluation Metrics")
    blue_text = "color: #3498DB;"
    #st.markdown(f'<p style="{blue_text}"><b>MSE:</b> {metrics["MSE"]:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    
    return metrics
