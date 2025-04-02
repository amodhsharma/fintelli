import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Preprocess dataset
def preprocess_data(df):
    df = df.copy()
    
    # Ensure the index is a datetime type
    df.index = pd.to_datetime(df.index)
    
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day'] = df.index.day
    
    return df

# ✅ Train XGBoost model
def train_xgboost(df, target_column):
    df = preprocess_data(df)
    
    # Debugging: Print sample data
    st.write("Processed Data Sample:", df.head())

    # Ensure the target column exists
    if target_column not in df.columns:
        st.error(f"Error: Target column '{target_column}' not found in dataset!")
        return None, None, None, None, None
    
    # Splitting data
    split_idx = int(len(df) * 0.85)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    # Selecting features and target
    features = ['day_of_week', 'month', 'year', 'day']
    X_train, y_train = train[features], train[target_column]
    X_test, y_test = test[features], test[target_column]

    # Train XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)
    
    # Plot results
    fig = plot_results(y_test, predictions, test.index)

    return y_test, predictions, model, metrics, fig

# ✅ Plot actual vs. predicted values
def plot_results(y_test, predictions, test_dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=predictions, mode='lines', name='Predicted', line=dict(color='orange')))

    fig.update_layout(title='XGBoost Forecast vs Actual', xaxis_title='Date', yaxis_title='Stock Price')
    return fig

# ✅ Compute evaluation metrics
def calculate_metrics(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R^2': r2
    }

# ✅ Run Streamlit app
def run_xgboost_forecast(df, target_column='Close'):  # Default target column as 'Close'
    st.title("Stock Price Forecasting using XGBoost")

    if df is None or df.empty:
        st.error("Error: No data available for processing!")
        return

    # ✅ Pass the correct target column
    y_test, predictions, _, metrics, fig = train_xgboost(df, target_column)
    
    if y_test is None:
        return  # Stop execution if error in training

    # Plot chart
    st.plotly_chart(fig)

    # Display evaluation metrics
    st.subheader("Evaluation Metrics")
    blue_text = "color: #3498DB;"
    st.markdown(f"**RMSE:** The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"**MAE:** On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"**MAPE:** The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"**R² Score:** <span style='{blue_text}'>{metrics['R^2']:.4f}</span> (Explains {metrics['R^2'] * 100:.2f}% of variance in stock prices).", unsafe_allow_html=True)



