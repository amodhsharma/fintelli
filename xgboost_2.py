import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Function to preprocess data
@st.cache_data
def preprocess_data(xg_data):
    xg_data = xg_data.copy()
    xg_data.index = pd.to_datetime(xg_data.index)
    xg_data['day_of_week'] = xg_data.index.dayofweek
    xg_data['month'] = xg_data.index.month
    xg_data['year'] = xg_data.index.year
    xg_data['day'] = xg_data.index.day
    return xg_data

# Train XGBoost model
@st.cache_resource
def train_xgboost(xg_data, target_column='Close'):
    xg_data = preprocess_data(xg_data)
    
    # Split data (85% train, 15% test)
    train_size = int(len(xg_data) * 0.85)
    train_df, test_df = xg_data.iloc[:train_size], xg_data.iloc[train_size:]
    
    # Define features & target
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]
    
    # Train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)
    
    # Plot results
    fig = plot_results(y_test, predictions, test_df.index)
    
    return y_test, predictions, model, metrics, fig

# Calculate evaluation metrics
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

# Plot actual vs predicted
def plot_results(y_test, predictions, test_dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=predictions, mode='lines', name='Predicted', line=dict(color='red', dash='dash')))

    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)

    fig.update_layout(title='XGBoost', xaxis_title='Date', yaxis_title='Stock Price',
        xaxis=dict(rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(showline=True, linecolor="white", linewidth=1),
        legend_title='Reference',
    )
    return fig

# Run Streamlit app
def run_xgboost_forecast(xg_data, target_column='Close'):
    st.markdown("<h3 style='color: cyan;'>M6: XG Boost</h3>", unsafe_allow_html=True),
    st.write("XGBoost is a gradient boosting framework that uses decision trees to create an ensemble model. It is known for its speed, performance, and ability to handle large datasets with high dimensionality."),
    #st.title('XGBoost'),
    #title_font=dict(color="yellow"),
    if xg_data is None or xg_data.empty:
        st.error('Error: No data available for processing!')
        return
    
    y_test, predictions, _, metrics, fig = train_xgboost(xg_data, target_column)
    
    # Plot results
    st.plotly_chart(fig)
    
    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)

    # Display evaluation metrics
    st.subheader("Evaluation Metrics")
    blue_text = "color: #3498DB;"
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)
