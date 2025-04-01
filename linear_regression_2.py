import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import plotly.graph_objects as go


# Load data
data = pd.read_csv("stock_data.csv")

# ----------------------------- Data Processing ---------------------------- #
def create_train_test_set(data):
    lr_data = data.copy()
    lr_data = lr_data.iloc[1:].drop(columns=['Volatility_30', 'MA_30', 'MA_90','MA_60','Volatility_90'])
    lr_data.index = pd.to_datetime(lr_data.index)
    # Show the shape of the lr_data table
    #st.write("Shape of the lr_data table:", lr_data.shape)
    
    features = lr_data.drop(columns=['Close'])
    target = lr_data['Close']
    
    # Show first few rows of features and target columns
    #st.write("First few rows of Features:")
    #st.write(features.head())
    #issue arising earlier - volatality etc, were being added in the data table

    #st.write("First few rows of Target (Close):")
    #st.write(target.head())
    
    data_len = len(lr_data)
    train_split = int(data_len * 0.80)
    val_split = train_split + int(data_len * 0.15)
    
    X_train, X_val, X_test = features[:train_split], features[train_split:val_split], features[val_split:]
    Y_train, Y_val, Y_test = target[:train_split], target[train_split:val_split], target[val_split:]
    
    # Display the lengths of X_train, X_val, X_test, Y_train, Y_val, Y_test
    #st.write(f"X_train , Y_train: ({len(X_train)} , {len(Y_train)})")
    #st.write(f"X_test , Y_test: ({len(X_test)} , {len(Y_test)})")
    #st.write(f"X_val , Y_val: ({len(X_val)} , {len(Y_val)})")
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# ----------------------------- MAPE Calculation ---------------------------- #
def get_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ----------------------------- Model Training ---------------------------- #
def process_data_for_regression(lr_data):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = create_train_test_set(lr_data)
    
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    
    Y_train_pred = lr.predict(X_train)
    Y_val_pred = lr.predict(X_val)
    Y_test_pred = lr.predict(X_test)
    
    result_metrics = {
        "Training R² Score": round(metrics.r2_score(Y_train, Y_train_pred), 2),
        "Training MAPE": f"{round(get_mape(Y_train, Y_train_pred), 2)}%",

        "Test R² Score": round(metrics.r2_score(Y_test, Y_test_pred), 2),
        "Test MAPE": f"{round(get_mape(Y_test, Y_test_pred), 2)}%",

        "Validation R² Score": round(metrics.r2_score(Y_val, Y_val_pred), 2),
        "Validation MAPE": f"{round(get_mape(Y_val, Y_val_pred), 2)}%",
        
    }
    
    plot = plot_predictions(Y_val, Y_val_pred)
    
    return result_metrics, plot

# ----------------------------- Visualization ---------------------------- #
def plot_predictions(Y_val, Y_val_pred):
    df_pred = pd.DataFrame({'Actual': Y_val.values, 'Predicted': Y_val_pred}, index=Y_val.index)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Predicted'], mode='lines', name='Predicted'))
    
    fig.update_layout(title="Stock Price Prediction using Linear Regression",
                      yaxis_title="Stock Price",
                      template="plotly_dark",
                      xaxis=dict(title="Date", rangeslider=dict(visible=True)))
    return fig
