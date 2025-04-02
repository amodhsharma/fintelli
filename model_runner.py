import pandas as pd
from linear_regression_3 import perform_linear_regression

# Load your dataset
data = pd.read_csv("stock_data.csv")  # Modify with the correct path

# Run Linear Regression model
metrics, predicted_price, buy_sell = perform_linear_regression(data)

# Store model results
models = {"LinearRegression": buy_sell}  
metrics_dict = {"LinearRegression": metrics}

# Function to retrieve model results
def get_model_results():
    return models, metrics_dict