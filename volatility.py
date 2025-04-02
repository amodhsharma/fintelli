import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def plot_volatility(data, ticker):
    st.markdown("<h3 style='color: cyan;'>EDA: Stock Price Volatility</h3>", unsafe_allow_html=True),
    st.write("The degree of variation in a stock's price over time, often measured by standard deviation or the average absolute change in price. 30-day volatility tracks price fluctuations over the past month, while 90-day volatility captures price movements over a three-month period, helping you assess short-term versus medium-term risk.")
    
    # Ensure 'Date' is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Calculate rolling volatility
    data['Volatility_30'] = data['Close'].rolling(window=30).std()
    data['Volatility_90'] = data['Close'].rolling(window=90).std()
    
    # Create figure
    fig = go.Figure(data=[
        go.Scatter(x=data.index, y=data['Volatility_30'], mode='lines', name='30-Day Rolling Volatility',line=dict(color='blue')),
        go.Scatter(x=data.index, y=data['Volatility_90'], mode='lines', name='90-Day Rolling Volatility',line=dict(color='red'))
    ])
    
    # Configure layout
    st.markdown("`EXPLORATORY DATA ANALYSIS`", unsafe_allow_html=True)

    fig.update_layout(
        autosize=True,
        title="Stock Price Volatility",
        legend_title='Reference',
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),  # Enable range slider
            type="date",
            showline=True,
            linecolor="white",  # Black x-axis line
            linewidth=1  # Make it bold
        ),
        yaxis=dict(title="Volatility",
            showline=True,  # Show y-axis line
            linecolor="white",  # Black y-axis line
            linewidth=1  # Make it bold)
        )
    )
    
    # Show the interactive plot
    st.plotly_chart(fig)
