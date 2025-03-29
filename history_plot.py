import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def updatefigure(fig):
    fig.update_layout(
        autosize=True,
        title="Time Series Data with Range Slider",
        xaxis_title="Date",
        yaxis_title="Stock Price",
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
        #plot_bgcolor="#E5ECF7",  # Background color
        #paper_bgcolor="#2E2E2E",  # Outer frame color
    )

def plot_history(data):
    fig = go.Figure(data=[
        go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name="stock_open", line=dict(color="green")),
        go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name="stock_close", line=dict(color="red"))
    ])
    updatefigure(fig)
    st.plotly_chart(fig, use_container_width=True)