import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
from dashboard.utils.strategy_handler import StrategyType, StrategyHandler

class StrategyVisualizer:
    def __init__(self, strategy_handler: StrategyHandler):
        self.strategy_handler = strategy_handler

    def show_strategy_selector(self) -> str:
        """Display strategy selection widget."""
        return st.selectbox(
            "Select Strategy",
            [strategy.value for strategy in StrategyType]
        )

    def show_strategy_parameters(self, strategy_type: str) -> Dict:
        """Display strategy parameter inputs."""
        st.subheader("Strategy Parameters")
        
        parameters = {}
        col1, col2 = st.columns(2)
        
        if strategy_type == StrategyType.MOVING_AVERAGE_CROSSOVER.value:
            with col1:
                parameters['fast_ma'] = st.number_input("Fast MA Period", value=9, min_value=1)
                parameters['slow_ma'] = st.number_input("Slow MA Period", value=21, min_value=1)
        
        elif strategy_type == StrategyType.RSI.value:
            with col1:
                parameters['rsi_period'] = st.number_input("RSI Period", value=14, min_value=1)
                parameters['oversold'] = st.number_input("Oversold Level", value=30, min_value=0, max_value=100)
            with col2:
                parameters['overbought'] = st.number_input("Overbought Level", value=70, min_value=0, max_value=100)
        
        elif strategy_type == StrategyType.MACD.value:
            with col1:
                parameters['fast_period'] = st.number_input("Fast Period", value=12, min_value=1)
                parameters['slow_period'] = st.number_input("Slow Period", value=26, min_value=1)
            with col2:
                parameters['signal_period'] = st.number_input("Signal Period", value=9, min_value=1)
        
        elif strategy_type == StrategyType.BOLLINGER_BANDS.value:
            with col1:
                parameters['window'] = st.number_input("Window Period", value=20, min_value=1)
                parameters['std_dev'] = st.number_input("Standard Deviation", value=2.0, min_value=0.1)
        
        return parameters

    def plot_strategy_results(self, data: pd.DataFrame, signals: pd.DataFrame):
        """Plot strategy results with indicators."""
        st.subheader("Strategy Analysis")
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ), row=1, col=1)
        
        # Add buy signals
        buy_signals = signals[signals['signal'] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['close'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='green'),
            name='Buy Signal'
        ), row=1, col=1)
        
        # Add sell signals
        sell_signals = signals[signals['signal'] == -1]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['close'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name='Sell Signal'
        ), row=1, col=1)
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume'
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=800,
            title_text="Strategy Performance"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_performance_metrics(self, metrics: Dict):
        """Display strategy performance metrics."""
        st.subheader("Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Win Rate",
                value=f"{metrics['win_rate']:.2%}",
                delta=f"{metrics['winning_trades']} / {metrics['total_trades']} trades"
            )
        
        with col2:
            st.metric(
                label="Profit Factor",
                value=f"{metrics['profit_factor']:.2f}",
                delta="Good" if metrics['profit_factor'] > 2 else "Fair" if metrics['profit_factor'] > 1 else "Poor"
            )
        
        with col3:
            st.metric(
                label="Max Drawdown",
                value=f"{metrics['max_drawdown']:.2%}",
                delta=f"Total Return: {metrics['total_return']:.2%}"
            )
        
        # Additional metrics
        st.markdown("### Risk Metrics")
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.metric(
                label="Sharpe Ratio",
                value=f"{metrics['sharpe_ratio']:.2f}",
                delta="Good" if metrics['sharpe_ratio'] > 1 else "Fair" if metrics['sharpe_ratio'] > 0 else "Poor"
            )
        
        with risk_col2:
            st.metric(
                label="Sortino Ratio",
                value=f"{metrics['sortino_ratio']:.2f}",
                delta="Good" if metrics['sortino_ratio'] > 1 else "Fair" if metrics['sortino_ratio'] > 0 else "Poor"
            )

    def plot_equity_curve(self, data: pd.DataFrame):
        """Plot strategy equity curve."""
        st.subheader("Equity Curve")
        
        fig = go.Figure()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + data['strategy_returns']).cumprod()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=cumulative_returns,
            mode='lines',
            name='Strategy Performance',
            line=dict(color='green', width=2)
        ))
        
        # Add buy & hold comparison
        buy_hold_returns = (1 + data['returns']).cumprod()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=buy_hold_returns,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            template='plotly_dark',
            title='Strategy vs Buy & Hold Performance',
            xaxis_title='Date',
            yaxis_title='Growth of $1',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_trade_list(self, data: pd.DataFrame):
        """Display list of trades with details."""
        st.subheader("Trade List")
        
        # Get only rows where signals changed
        trades = data[data['signal'] != 0].copy()
        trades['type'] = trades['signal'].map({1: 'BUY', -1: 'SELL'})
        
        trade_data = pd.DataFrame({
            'Date': trades.index,
            'Type': trades['type'],
            'Price': trades['close'].map('${:,.2f}'.format),
            'Return': trades['strategy_returns'].map('{:,.2%}'.format),
            'Volume': trades['volume'].map('{:,.0f}'.format)
        })
        
        st.dataframe(
            trade_data,
            use_container_width=True,
            hide_index=True
        ) 