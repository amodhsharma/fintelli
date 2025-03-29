import pandas as pd 
import plotly.graph_objects as go

def moving_average(data, month=30, two_months=60, quarter=90):
    # Calculate 30, 60, 90-day moving averages
    data["MA_30"] = data["Close"].rolling(window=month).mean()
    data["MA_60"] = data["Close"].rolling(window=two_months).mean()
    data["MA_90"] = data["Close"].rolling(window=quarter).mean()
    return data

def update_figure(fig_moving):
    fig_moving.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis=dict(
            rangeslider=dict(visible=True),
            showline=True,
            linecolor="white",
            linewidth=1
        ),
        yaxis=dict(
            showline=True,
            linecolor="white",
            linewidth=1
        ),
        font=dict(color="white")
    )

def plot_moving_average(data, ticker):
    fig_moving = go.Figure()

    # Use data.index as the date reference
    fig_moving.add_trace(go.Scatter(x=data.index, y=data["MA_30"], mode="lines", name="30-Day MA", line=dict(color="cyan", dash="dot")))
    fig_moving.add_trace(go.Scatter(x=data.index, y=data["MA_60"], mode="lines", name="60-Day MA", line=dict(color="yellow", dash="dot")))
    fig_moving.add_trace(go.Scatter(x=data.index, y=data["MA_90"], mode="lines", name="90-Day MA", line=dict(color="purple", dash="dot")))

    update_figure(fig_moving)  # Apply styling
    return fig_moving
