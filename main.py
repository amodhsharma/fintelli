# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt


from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

import pandas as pd

st.set_page_config(page_title="Fintelli", page_icon="logo2.jpg")


from prophet_script import forecast_stock_prices  


st.title('Fintelli: Your very own stock forecast app')
from stocks import stocks
print(stocks)
ticker = st.selectbox("Select Stock Ticker:", stocks) 

# ---------------------------------------------------------------
# Section change - below code snippets loads data
# ---------------------------------------------------------------

#caching data for easier retrieval
import yfinance as yf
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, "2015-01-01", date.today().strftime("%Y-%m-%d"))
    data.index = pd.to_datetime(data.index) # Ensure 'Date' is datetime and set as index
    data = data.sort_index()  # Ensure chronological order
    data.reset_index(inplace=True)  #resetting index to get date as main column
    return data

data = load_data(ticker)    #loading data
st.write(f"Showing data for {ticker}")  # Display the loaded data

data.to_csv("stock_data.csv")

# ---------------------------------------------------------------
# Section change 
# ---------------------------------------------------------------

#plotting the tail of the raw data
st.write('Raw data - From the beginning')
st.write(data.head())
st.write('Raw data - Towards the end ( more recent )')
st.write(data.tail())

#st.write("Data types:")
#st.write(data.dtypes)

#st.write("shape of dataset")
#st.write(data.shape) 

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)  # Drop the second level

#st.write("Fixed Column Names:", data.columns)
#st.write("Updated Shape:", data.shape)

# ---------------------------------------------------------------
# Section change - below code snippets displays history
# ---------------------------------------------------------------
from history_plot import plot_history
plot_history(data)

# ---------------------------------------------------------------
# Section change - below code snippets displays volitality
# ---------------------------------------------------------------
from volatility import plot_volatility
plot_volatility(data, ticker)

# ---------------------------------------------------------------
# Section change - below code snippets displays moving averages
# ---------------------------------------------------------------
from moving_average import moving_average, plot_moving_average
data_moving = moving_average(data)
fig_moving = plot_moving_average(data, ticker)
st.plotly_chart(fig_moving)


# ---------------------------------------------------------------
# Section change - below code snippets displays linear regression 
# ---------------------------------------------------------------
from linear_regression_3 import perform_linear_regression
perform_linear_regression(data)

# ---------------------------------------------------------------
# Section change - below code snippets displays random forest
# ---------------------------------------------------------------
from random_forest import perform_random_forest
perform_random_forest(data)

#causing issues
# from random_forest_2 import train_and_forecast_random_forest
# train_and_forecast_random_forest(data)

# ---------------------------------------------------------------
# Section change - below code snippets displays LSTM
# ---------------------------------------------------------------
# from lstm_2 import forecast_lstm
# forecast_lstm(data, ticker)

from lstm3 import lstm
lstm(data)

# ---------------------------------------------------------------
# Section change - below code snippets displays prophet
# ---------------------------------------------------------------

# st.subheader('Prophet Prediction')
# n_years = st.slider('Years of prediction:', 1, 4)
# period = n_years * 365
# forecast_stock_prices(data, period, n_years)

# from prophet2 import train_and_forecast_prophet
# train_and_forecast_prophet(data)

# from prophet3 import train_and_forecast_prophet
# train_and_forecast_prophet(data)

from prophet4 import train_and_evaluate_prophet
train_and_evaluate_prophet(data)

# ---------------------------------------------------------------
# Section change - below code snippets displays arima
# ---------------------------------------------------------------
from arima import forecast_stock_prices_arima
forecast_stock_prices_arima(data, order=(6,1,0))

# from arima2 import arima_model
# arima_model(data)

# ---------------------------------------------------------------
# Section change - below code snippets displays sarima
# ---------------------------------------------------------------

# from sarima import sarima_forecast
# sarima_forecast(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

from sarima2 import sarima_forecast
sarima_forecast(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# ---------------------------------------------------------------
# Section change - below code snippets displays exponential smoothening
# ---------------------------------------------------------------

# from exponential import forecast_stock_prices_expsmoothing
# forecast_stock_prices_expsmoothing(data)

from exponential2 import forecast_stock_prices_expsmoothing
forecast_stock_prices_expsmoothing(data)