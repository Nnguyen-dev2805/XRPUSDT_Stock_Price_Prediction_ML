"""
Visualization Utilities Module
Tạo các biểu đồ và visualizations
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_price_history(df, n_days=100):
    """
    Vẽ biểu đồ giá lịch sử với moving averages
    
    Args:
        df: DataFrame chứa dữ liệu
        n_days: Số ngày hiển thị
    
    Returns:
        Plotly figure
    """
    df_plot = df.tail(n_days).copy()
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df_plot['Date'],
        y=df_plot['Price'],
        mode='lines',
        name='Price',
        line=dict(color='#00D9FF', width=2)
    ))
    
    # SMA lines if available
    if 'SMA_7' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['Date'],
            y=df_plot['SMA_7'],
            mode='lines',
            name='SMA 7',
            line=dict(color='#FF6B6B', width=1, dash='dash')
        ))
    
    if 'SMA_20' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['Date'],
            y=df_plot['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='#4ECDC4', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title='XRP/USDT Price History',
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        template='plotly_dark',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_candlestick(df, n_days=60):
    """
    Vẽ candlestick chart
    
    Args:
        df: DataFrame chứa OHLC data
        n_days: Số ngày hiển thị
    
    Returns:
        Plotly figure
    """
    df_plot = df.tail(n_days).copy()
    
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['Date'],
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Price'],
        name='OHLC'
    )])
    
    fig.update_layout(
        title='XRP/USDT Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=400
    )
    
    return fig


def plot_volume(df, n_days=60):
    """
    Vẽ biểu đồ volume
    
    Args:
        df: DataFrame chứa volume data
        n_days: Số ngày hiển thị
    
    Returns:
        Plotly figure
    """
    df_plot = df.tail(n_days).copy()
    
    # Color based on price change
    colors = ['#00FF00' if df_plot['Price'].iloc[i] >= df_plot['Open'].iloc[i] 
              else '#FF0000' for i in range(len(df_plot))]
    
    fig = go.Figure(data=[go.Bar(
        x=df_plot['Date'],
        y=df_plot['Vol'],
        marker_color=colors,
        name='Volume'
    )])
    
    fig.update_layout(
        title='Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_dark',
        height=250
    )
    
    return fig


def plot_technical_indicators(df, n_days=60):
    """
    Vẽ các technical indicators (RSI, MACD)
    
    Args:
        df: DataFrame chứa technical indicators
        n_days: Số ngày hiển thị
    
    Returns:
        Plotly figure
    """
    df_plot = df.tail(n_days).copy()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('RSI (14)', 'MACD'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # RSI
    if 'RSI_14' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['Date'],
            y=df_plot['RSI_14'],
            mode='lines',
            name='RSI',
            line=dict(color='#FFD700', width=2)
        ), row=1, col=1)
        
        # Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    if 'MACD' in df_plot.columns and 'MACD_Signal' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['Date'],
            y=df_plot['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='#00D9FF', width=2)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_plot['Date'],
            y=df_plot['MACD_Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='#FF6B6B', width=2)
        ), row=2, col=1)
        
        if 'MACD_Histogram' in df_plot.columns:
            colors = ['green' if val >= 0 else 'red' for val in df_plot['MACD_Histogram']]
            fig.add_trace(go.Bar(
                x=df_plot['Date'],
                y=df_plot['MACD_Histogram'],
                name='Histogram',
                marker_color=colors
            ), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_prediction_result(actual_prices, predicted_prices, dates):
    """
    Vẽ biểu đồ so sánh actual vs predicted
    
    Args:
        actual_prices: Array giá thực tế
        predicted_prices: Array giá dự đoán
        dates: Array dates
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_prices,
        mode='lines',
        name='Actual Price',
        line=dict(color='#00D9FF', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted_prices,
        mode='lines',
        name='Predicted Price',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        template='plotly_dark',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_feature_importance(feature_importance_df, top_n=15):
    """
    Vẽ biểu đồ feature importance
    
    Args:
        feature_importance_df: DataFrame chứa feature importance
        top_n: Số lượng top features
    
    Returns:
        Plotly figure
    """
    df_plot = feature_importance_df.head(top_n)
    
    fig = go.Figure(data=[go.Bar(
        x=df_plot['Importance'],
        y=df_plot['Feature'],
        orientation='h',
        marker=dict(
            color=df_plot['Importance'],
            colorscale='Viridis'
        )
    )])
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_dark',
        height=500
    )
    
    return fig


def plot_prediction_30d(df, predictions, target_date):
    """
    Vẽ biểu đồ 30 ngày lịch sử và điểm dự báo T+1
    
    Args:
        df: DataFrame chứa dữ liệu lịch sử
        predictions: Dictionary các dự báo (e.g., {'RF': {'price': ...}, 'SVR': {...}})
        target_date: Ngày mục tiêu dự báo
    
    Returns:
        Plotly figure
    """
    # Lấy 30 ngày cuối
    df_plot = df.tail(30).copy()
    
    fig = go.Figure()
    
    # 1. Historical Data (Green/Cyan line)
    fig.add_trace(go.Scatter(
        x=df_plot['Date'],
        y=df_plot['Price'],
        mode='lines+markers',
        name='Lịch sử (30 ngày)',
        line=dict(color='#00D9FF', width=3),
        marker=dict(size=6)
    ))
    
    # 2. Add Predictions (Red points)
    colors_pred = ['#FF4B4B', '#FF7675'] # Shades of red
    for i, (m_type, data) in enumerate(predictions.items()):
        name = f"Dự báo {m_type} (T+1)"
        fig.add_trace(go.Scatter(
            x=[target_date],
            y=[data['price']],
            mode='markers+text',
            name=name,
            marker=dict(color=colors_pred[i % len(colors_pred)], size=12, symbol='star'),
            text=[f"${data['price']:.4f}"],
            textposition="top center"
        ))
        
        # Add a connecting dashed line from last point to prediction
        last_date = df_plot['Date'].iloc[-1]
        last_price = df_plot['Price'].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_date, target_date],
            y=[last_price, data['price']],
            mode='lines',
            name=f'Trend {m_type}',
            line=dict(color=colors_pred[i % len(colors_pred)], width=2, dash='dot'),
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text=f'Bối cảnh Thị trường & Dự đoán Ngày {target_date.strftime("%d/%m/%Y")}',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Ngày',
        yaxis_title='Giá XRP (USDT)',
        template='plotly_dark',
        hovermode='x unified',
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

