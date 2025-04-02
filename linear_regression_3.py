import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lr_data = pd.read_csv("stock_data.csv")  # Read and store data
lr_data.to_csv("stock_data_copy.csv")  # Save copy

# Cache the evaluation metrics to avoid redundant calculations
@st.cache_data
def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    # metrics = {'MSE': mse,'RMSE': rmse,'MAE': mae,'MAPE': mape,'R^2': r2}
    # return metrics

    return{'MSE': mse,'RMSE': rmse,'MAE': mae,'MAPE': mape,'R^2': r2}

# Cache the data preprocessing to avoid redundant operations on data
@st.cache_data
def create_train_test_set(lr_data):
    lr_data = lr_data.iloc[1:].drop(columns=['Volatility_30', 'MA_30', 'MA_90', 'MA_60', 'Volatility_90'])
    lr_data.index = pd.to_datetime(lr_data.index)
    return lr_data

# Cache the model training and prediction to avoid retraining the model every time
@st.cache_resource
def perform_linear_regression(lr_data):
    #st.title("Linear Regression Model")

    st.markdown("<h1 style='color: cyan;'>Traditional Regression & Statistical Models</h1>", unsafe_allow_html=True),
    st.markdown("<h3 style='color: cyan;'>M1: Linear Regression Model</h3>", unsafe_allow_html=True),
    st.write("Linear Regression is a fundamental supervised machine learning algorithm used for modeling the relationship between a dependent variable (target) and one or more independent variables (predictors).")
    st.latex(r"y = mx + c")
    st.write("Y → Dependent variable (Target) → The value we want to predict (e.g., future stock price).")
    st.write("X → Independent variable (Predictor) → The input used to predict Y (e.g., time, past stock prices).")
    st.write("m → Slope (Coefficient) → Represents how much Y changes when X increases by 1 unit.")
    st.write("c → Y-intercept → The value of Y when X is zero.")

    # Preprocess data
    lr_data = create_train_test_set(lr_data)
    target_column = "Close"  # Default target column
    
    # Ensure the index is a DateTime index
    lr_data = lr_data.sort_index()
    
    # Splitting data into features (X) and target (y)
    X = lr_data.drop(columns=[target_column])
    y = lr_data[target_column]
    
    # 85%-15% Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = evaluate_metrics(y_test, y_pred)
    
    # Plot actual vs. predicted values using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=X_test.index, y=y_pred, mode='lines', name='Predicted', line=dict(color='orange')))
    
    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)
    fig.update_layout(title='Linear Regression',
                    legend_title='Reference',
                    xaxis=dict(title="Date",rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
                    yaxis=dict(title=target_column, showline=True, linecolor="white", linewidth=1))
    
    # Display plot in Streamlit
    st.plotly_chart(fig)
    
    # Display metrics in Streamlit
    blue_text = "color: #3498DB;"  # Hex code for light blue

    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)
    st.subheader("Evaluation Metrics")
    #st.markdown(f'<p style="{blue_text}"><b>MSE:</b> {metrics["MSE"]:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model’s absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    predicted_price = y_pred[-1]
    st.markdown("`CLOSING PRICE PREDECTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="Linear Regression", value=f"₹{predicted_price:.2f}" if predicted_price else "N/A")

    # Determine Buy/Sell probability based on price trend
    last_actual_price = y_test.iloc[-1]  # Last actual closing price
    if predicted_price > last_actual_price:
        buy_prob = 0.7  # Higher probability for buying
        sell_prob = 0.3
    else:
        buy_prob = 0.3  # Higher probability for selling
        sell_prob = 0.7

    return metrics, predicted_price, {"buy": buy_prob, "sell": sell_prob}
