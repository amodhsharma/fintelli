import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

# Function to predict the next 30 days using Random Forest
@st.cache_resource
def predict_next_month(rf_data):
    st.markdown("<h3 style='color: cyan;'>M5.1: Future Predictions using Random Forest</h3>", unsafe_allow_html=True)
    st.markdown("`FUTURE PREDICTION PLOT`", unsafe_allow_html=True)

    # Ensure index is DateTime and sort by date
    rf_data.index = pd.to_datetime(rf_data.index)
    rf_data = rf_data.sort_index()

    target_column = "Close"  # Target variable

    # Splitting into features (X) and target (y)
    X = rf_data.drop(columns=[target_column])
    y = rf_data[target_column]

    # Train model using the entire dataset
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Generate future dates for the next month
    last_date = rf_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')  # 'B' for business days

    # Use last available feature values as a base for future predictions
    last_features = X.iloc[-1].values.reshape(1, -1)

    future_predictions = []
    
    for _ in range(30):
        predicted_price = model.predict(last_features)[0]
        future_predictions.append(predicted_price)

        # Update features for next iteration (assuming features do not drastically change)
        last_features = np.append(last_features[:, 1:], predicted_price).reshape(1, -1)

    # Convert predictions into DataFrame
    future_df = pd.DataFrame({target_column: future_predictions}, index=future_dates)

    # Combine last 100 actual data points with predicted values
    recent_actuals = rf_data.iloc[-100:][target_column]
    
    # Plot using Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recent_actuals.index,
        y=recent_actuals,
        mode='lines',
        name='Actual (Last 100 Days)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_df.index,
        y=future_df[target_column],
        mode='lines',
        name='Predicted (Next 30 Days)',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.update_layout(
        title="Future Predictions using Random Forest",
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis_title="Stock Closing Price",
        legend_title="Legend"
    )
    
    st.plotly_chart(fig)
    
    return future_df  # Returning predictions if needed elsewhere in the app
