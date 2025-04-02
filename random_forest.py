import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

# Cache the evaluation metrics function to avoid redundant calculations
@st.cache_data
def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R^2': r2
    }
    
    return metrics

# Cache the model training and prediction to avoid re-training the model
@st.cache_resource
def perform_random_forest(rf_data):
    #st.title("Random Forest Regression Model")
    st.markdown("<h1 style='color: cyan;'>Tree-Based & Ensemble Learning</h1>", unsafe_allow_html=True),
    st.markdown("<h3 style='color: cyan;'>M5: Random Forest Regression Model</h3>", unsafe_allow_html=True),
    st.write("Random Forest is powerful for financial prediction models due to its ability to handle complex, non-linear data and prevent overfitting through ensemble learning. It can rank feature importance, making it useful for identifying key financial factors. Its robustness to missing data and non-linear relationships makes it versatile and accurate for tasks like stock price forecasting and risk assessment.")    
    
    # Ensure the index is a DateTime index
    rf_data.index = pd.to_datetime(rf_data.index)
    rf_data = rf_data.sort_index()  # Sort by Date
    
    target_column = "Close"  # Set the target column for prediction
    
    # Splitting data into features (X) and target (y)
    X = rf_data.drop(columns=[target_column])
    y = rf_data[target_column]
    
    # 85%-15% Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set (15% of data)
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    metrics = evaluate_metrics(y_test, y_pred)
    
    # Plot actual vs. predicted values using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted', line=dict(color='orange')))
    
    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)

    fig.update_layout(title='Random Forest',
                    xaxis=dict(title="Date", rangeslider=dict(visible=True)), 
                    yaxis_title=target_column,
                    legend_title='Reference')
    
    # Display Plotly chart
    st.plotly_chart(fig)

    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)

    st.subheader("Evaluation Metrics")
    blue_text = "color: #3498DB;"
    #st.markdown(f'<p style="{blue_text}"><b>MSE:</b> {metrics["MSE"]:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model’s absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    predicted_price = y_pred[-1] if len(y_pred) > 0 else None
    st.markdown("`CLOSING PRICE PREDICTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="Random Forest", value=f"₹{predicted_price:.2f}" if predicted_price is not None else "N/A")

    return metrics
