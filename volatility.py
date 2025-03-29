import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def plot_volatility(data, ticker):
    st.subheader(f'Stock Volatility for {ticker}')
    
    # Ensure 'Date' is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Calculate rolling volatility
    data['Volatility_30'] = data['Close'].rolling(window=30).std()
    data['Volatility_90'] = data['Close'].rolling(window=90).std()
    
    # Create figure
    fig = go.Figure()
    
    # 30-Day Rolling Volatility
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Volatility_30'], 
        mode='lines', 
        name='30-Day Rolling Volatility',
        line=dict(color='blue')
    ))
    
    # 90-Day Rolling Volatility
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Volatility_90'], 
        mode='lines', 
        name='90-Day Rolling Volatility',
        line=dict(color='red')
    ))
    
    # Configure layout
    fig.update_layout(
        #title=f"Stock Price Volatility for {ticker}",
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),  # Enable range slider
            type="date"
        ),
        yaxis=dict(title="Volatility"),
        showlegend=True
    )
    
    # Show the interactive plot
    st.plotly_chart(fig, use_container_width=True)
