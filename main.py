# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

import pandas as pd

st.set_page_config(page_title="Fintelli", page_icon="logo2.jpg")

from stocks import stocks
from history_plot import plot_history
from volatility import plot_volatility
from moving_average import moving_average, plot_moving_average
#linear regression
from prophet_script import forecast_stock_prices  
from lstm import forecast_lstm

#cant change format as yfinance requires yyyymmdd format
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')
print(stocks)

ticker = st.selectbox("Select Stock Ticker:", stocks) 

# ---------------------------------------------------------------
# Section change - below code snippets loads data
# ---------------------------------------------------------------

#caching data for easier retrieval
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)

    # Ensure 'Date' is datetime and set as index
    data.index = pd.to_datetime(data.index)  
    data = data.sort_index()  # Ensure chronological order

    #resetting index to get date as main column
    data.reset_index(inplace=True)
    return data
	
data_load_state = st.text('Loading data...')
#main dataframe name is data
data = load_data(ticker)

#to load it into csv for easy loading in csv
data.to_csv("stock_data.csv")

data_load_state.text('Loading data... done!')

data["Date"] = pd.to_datetime(data["Date"])

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
# Section change - below code snippets displays code WITH range slider
# ---------------------------------------------------------------

plot_history(data)

# ---------------------------------------------------------------
# Section change - below code snippets displays volitality
# ---------------------------------------------------------------

plot_volatility(data, ticker)

# ---------------------------------------------------------------
# Section change - below code snippets displays moving averages
# ---------------------------------------------------------------

st.subheader('Moving Averages')
data_moving = moving_average(data)
fig_moving = plot_moving_average(data, ticker)
st.plotly_chart(fig_moving, use_container_width=True)


# ---------------------------------------------------------------
# Section change - below code snippets displays linear regression 
# ---------------------------------------------------------------
from linear_regression import forecast_stock_prices_linear

# Define START and TODAY
START = "2015-01-01"  # Change as needed
TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")

# Run forecast
forecast_stock_prices_linear(data, period=30, start_date=START, end_date=TODAY)

# ---------------------------------------------------------------
# Section change - below code snippets displays LSTM
# ---------------------------------------------------------------
forecast_lstm(data, ticker)


# ---------------------------------------------------------------
# Section change - below code snippets displays prophet
# ---------------------------------------------------------------

st.subheader('Prophet Prediction')
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365
forecast_stock_prices(data, period, n_years)

# ---------------------------------------------------------------
# Section change - below code snippets displays arima
# ---------------------------------------------------------------
from arima import forecast_stock_prices_arima
forecast_stock_prices_arima(data, order=(6,1,0))

# ---------------------------------------------------------------
# Section change - below code snippets displays sarima
# ---------------------------------------------------------------

from sarima import sarima_forecast
sarima_forecast(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# ---------------------------------------------------------------
# Section change - below code snippets displays exponential smoothening
# ---------------------------------------------------------------

from exponential import forecast_stock_prices_expsmoothing
forecast_stock_prices_expsmoothing(data)