import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Streamlit App Title
st.title("Stock Price Prediction using Random Forest")

def train_and_forecast_random_forest(rf_data):
    # Selecting only required columns
    rf_data = rf_data[["Close"]]
    rf_data["Day"] = rf_data.index.day
    rf_data["Month"] = rf_data.index.month
    rf_data["Year"] = rf_data.index.year

    # Splitting the data (85% train, 15% test)
    train_size = int(len(rf_data) * 0.85)
    train, test = rf_data.iloc[:train_size], rf_data.iloc[train_size:]
    
    # Preparing features and target
    features = ["Day", "Month", "Year"]
    X_train, X_test = train[features], test[features]
    y_train, y_test = train["Close"], test["Close"]
    
    # Training Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Making predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Display evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")
    
    # Plot actual vs predicted values using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=y_test.values, mode='lines', name='Actual Closing Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=y_pred, mode='lines', name='Predicted Closing Price', line=dict(color='red', dash='dash')))
    fig.update_layout(title="Actual vs Predicted Closing Price", template='plotly_dark',
        xaxis=dict(title="Date",rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(title="Closing Price",showline=True, linecolor="white", linewidth=1)
    )
    
    st.plotly_chart(fig)
    
    return model, mae, mse, rmse, r2