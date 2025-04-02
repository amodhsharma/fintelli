import streamlit as st
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt


from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

import pandas as pd

st.set_page_config(page_title="Fintelli", page_icon="logo2.jpg")


#from prophet_script import forecast_stock_prices  


st.title('Fintelli: Your very own stock forecast app')
from stocks import stocks
print(stocks)

ticker = st.selectbox("`USER INPUT` Select Stock Ticker:", stocks) 

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
st.markdown(f"✅ **Showing data for `{ticker}`**", unsafe_allow_html=True)

data.to_csv("stock_data.csv")

# ---------------------------------------------------------------
# Section change 
# ---------------------------------------------------------------

#plotting the tail of the raw data
st.markdown("`PREPROCESSING` Raw data - From the beginning ", unsafe_allow_html=True)
#st.write('Raw data - From the beginning')
st.write(data.head())
st.markdown("`PREPROCESSING` Raw data - Towards the end ", unsafe_allow_html=True)
# st.markdown("`Raw data - Towards the end`", unsafe_allow_html=True)
#st.markdown(f" **`Raw data - Towards the end` **", unsafe_allow_html=True)
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
# EDA Section change - below code snippets displays history
# ---------------------------------------------------------------
from history_plot import plot_history
plot_history(data)
# ---------------------------------------------------------------
# EDASection change - below code snippets displays volitality
# ---------------------------------------------------------------
from volatility import plot_volatility
plot_volatility(data, ticker)
# ---------------------------------------------------------------
# EDA Section change - below code snippets displays moving averages
# ---------------------------------------------------------------
from moving_average import moving_average, plot_moving_average
data_moving = moving_average(data)
fig_moving = plot_moving_average(data, ticker)
st.plotly_chart(fig_moving)
# ---------------------------------------------------------------
# MODEL 1 Section change - below code snippets displays linear regression 
# ---------------------------------------------------------------
from linear_regression_3 import perform_linear_regression
perform_linear_regression(data)
# ---------------------------------------------------------------
# M2 Section change - below code snippets displays exponential smoothening
# ---------------------------------------------------------------
from exponential2 import forecast_stock_prices_expsmoothing
forecast_stock_prices_expsmoothing(data)
# ---------------------------------------------------------------
# M3: section change - below code snippets displays arima
# ---------------------------------------------------------------
from arima import forecast_stock_prices_arima
forecast_stock_prices_arima(data, order=(6,1,0))
# ---------------------------------------------------------------
# M4: Section change - below code snippets displays sarima
# ---------------------------------------------------------------
from sarima2 import sarima_forecast
sarima_forecast(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
# ---------------------------------------------------------------
# M5: Section change - below code snippets displays random forest
# ---------------------------------------------------------------
from random_forest import perform_random_forest
perform_random_forest(data)
# ---------------------------------------------------------------
# M6: Section change - below code snippets displays xgboost
# ---------------------------------------------------------------
from xgboost_2 import run_xgboost_forecast
run_xgboost_forecast(data, target_column='Close')
# ---------------------------------------------------------------
# M7: Section change - below code snippets displays prophet
# ---------------------------------------------------------------
from prophet4 import train_and_evaluate_prophet
train_and_evaluate_prophet(data)
# ---------------------------------------------------------------
# M8: Section change - below code snippets displays LSTM
# ---------------------------------------------------------------
from lstm3 import lstm
lstm(data)







