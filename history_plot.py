import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def updatefigure(fig):
    fig.update_layout(
        autosize=True,
        title="Time Series Plot",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title='Reference',
        xaxis=dict(
            rangeslider=dict(visible=True),
            showline=True,  # Show x-axis line
            linecolor="white",  # Black x-axis line
            linewidth=1  # Make it bold
        ),
        yaxis=dict(
            showline=True,  # Show y-axis line
            linecolor="white",  # Black y-axis line
            linewidth=1  # Make it bold
        ),
    )

def plot_history(data):
    st.markdown("<h3 style='color: cyan;'>EDA: Time Series Plot</h3>", unsafe_allow_html=True),
    st.write("Historical data plotting for opening and closing prices with opposed to Date, plotted below. Allows for an easier visualisation of the stock price movement over time.")

    fig = go.Figure(data=[
        go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name="stock_open", line=dict(color="green")),
        go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name="stock_close", line=dict(color="red"))
    ])
    st.markdown("`EXPLORATORY DATA ANALYSIS`", unsafe_allow_html=True)

    updatefigure(fig)
    st.plotly_chart(fig)
