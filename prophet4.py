import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from prophet import Prophet
import streamlit as st

@st.cache_data  # Cache the results of the training and evaluation process
def train_and_evaluate_prophet(data):
    st.title("Prophet Model")
    st.write("Prophet is a forecasting tool created by Facebook, designed to handle time series data that may have missing values and seasonal effects. It is particularly effective for daily observations with strong seasonal patterns.")
    
    # Prepare data for Prophet (use index as 'ds' and 'Close' as 'y')
    data['ds'] = data.index  # Use the index (Date) as 'ds'
    data['y'] = data['Close']
    
    # Train-test split (85-15 split)
    train_data = data.sample(frac=0.85, random_state=0)
    test_data = data.drop(train_data.index)
    
    # Initialize Prophet model with daily seasonality
    model = Prophet(daily_seasonality=True)
    
    # Train the model
    model.fit(train_data)
    
    # Make predictions on the test data
    prediction = model.predict(pd.DataFrame({'ds': test_data.index}))
    
    # Extract actual vs predicted values
    y_actual = test_data['y']
    y_predicted = prediction['yhat']
    y_predicted = y_predicted.astype(int)  # Convert to integer if needed for evaluation
    
    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_predicted)
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = sqrt(mse)
    r2 = r2_score(y_actual, y_predicted)
    mape = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100
    
    # Prepare metrics
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "R^2": r2,
        "MAPE": mape
    }
    
    # Plotting with Plotly (only showing past 15% data for the test)
    fig = go.Figure()

    # Add actual vs predicted line plots for the test data only
    fig.add_trace(go.Scatter(x=test_data.index, y=y_actual, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data.index, y=y_predicted, mode='lines', name='Predicted', line=dict(color='orange')))
    
    # Add a range slider to the x-axis
    fig.update_layout(
        title="Forecasting Model for Prophet",
        xaxis_title="Date",
        yaxis_title="Close Price",
        xaxis=dict(rangeslider=dict(visible=True)),
        template="plotly_dark"
    )
    
    # Display the plot
    st.plotly_chart(fig)
    
    # Display evaluation metrics with custom color
    st.subheader("Evaluation Metrics")
    blue_text = "color: #3498DB;"
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model’s absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    return metrics
