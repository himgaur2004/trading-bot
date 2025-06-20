import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List

class StrategyVisualizer:
    """Component for visualizing trading strategies and results"""
    
    def __init__(self, strategy_handler):
        self.strategy_handler = strategy_handler
        
    def show_strategy_selector(self) -> str:
        """Show strategy selection dropdown"""
        return st.selectbox(
            "Select Strategy",
            [
                "trend_following",
                "mean_reversion",
                "breakout",
                "grid_trading",
                "momentum"
            ]
        )
        
    def show_strategy_parameters(self, strategy: str) -> Dict:
        """Show parameter inputs for selected strategy"""
        st.subheader("Strategy Parameters")
        
        params = {}
        
        if strategy == "trend_following":
            col1, col2 = st.columns(2)
            with col1:
                params["ema_fast"] = st.number_input("Fast EMA Period", 5, 50, 9)
                params["ema_slow"] = st.number_input("Slow EMA Period", 10, 200, 21)
            with col2:
                params["atr_period"] = st.number_input("ATR Period", 5, 50, 14)
                params["atr_multiplier"] = st.number_input("ATR Multiplier", 1.0, 5.0, 2.0)
                
        elif strategy == "mean_reversion":
            col1, col2 = st.columns(2)
            with col1:
                params["rsi_period"] = st.number_input("RSI Period", 5, 50, 14)
                params["rsi_overbought"] = st.number_input("RSI Overbought", 50, 90, 70)
                params["rsi_oversold"] = st.number_input("RSI Oversold", 10, 50, 30)
            with col2:
                params["bb_period"] = st.number_input("Bollinger Period", 5, 50, 20)
                params["bb_std"] = st.number_input("Bollinger StdDev", 1.0, 3.0, 2.0)
                
        elif strategy == "breakout":
            col1, col2 = st.columns(2)
            with col1:
                params["breakout_period"] = st.number_input("Breakout Period", 5, 100, 20)
                params["volume_factor"] = st.number_input("Volume Factor", 1.0, 5.0, 2.0)
            with col2:
                params["atr_period"] = st.number_input("ATR Period", 5, 50, 14)
                params["atr_multiplier"] = st.number_input("ATR Multiplier", 1.0, 5.0, 2.0)
                
        elif strategy == "grid_trading":
            col1, col2 = st.columns(2)
            with col1:
                params["grid_levels"] = st.number_input("Grid Levels", 3, 20, 5)
                params["grid_spacing"] = st.number_input("Grid Spacing %", 0.5, 10.0, 2.0)
            with col2:
                params["take_profit"] = st.number_input("Take Profit %", 0.5, 10.0, 1.0)
                params["stop_loss"] = st.number_input("Stop Loss %", 0.5, 10.0, 2.0)
                
        elif strategy == "momentum":
            col1, col2 = st.columns(2)
            with col1:
                params["macd_fast"] = st.number_input("MACD Fast", 5, 50, 12)
                params["macd_slow"] = st.number_input("MACD Slow", 10, 100, 26)
                params["macd_signal"] = st.number_input("MACD Signal", 5, 50, 9)
            with col2:
                params["rsi_period"] = st.number_input("RSI Period", 5, 50, 14)
                params["volume_ma"] = st.number_input("Volume MA", 5, 50, 20)
                
        return params
        
    def plot_results(self, data: pd.DataFrame, signals: pd.DataFrame):
        """Plot trading results with signals"""
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ))
        
        # Add buy signals
        if 'buy' in signals.columns:
            buy_points = signals[signals['buy'] == 1]
            fig.add_trace(go.Scatter(
                x=buy_points.index,
                y=buy_points['close'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green'
                ),
                name='Buy Signal'
            ))
            
        # Add sell signals
        if 'sell' in signals.columns:
            sell_points = signals[signals['sell'] == 1]
            fig.add_trace(go.Scatter(
                x=sell_points.index,
                y=sell_points['close'],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red'
                ),
                name='Sell Signal'
            ))
            
        fig.update_layout(
            title='Trading Signals',
            yaxis_title='Price',
            template='plotly_dark',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def show_metrics(self, metrics: Dict):
        """Display strategy performance metrics"""
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return', 0):.2f}%",
                delta=None
            )
            
        with col2:
            st.metric(
                "Win Rate",
                f"{metrics.get('win_rate', 0):.1f}%",
                delta=None
            )
            
        with col3:
            st.metric(
                "Profit Factor",
                f"{metrics.get('profit_factor', 0):.2f}",
                delta=None
            )
            
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.2f}%",
                delta=None
            )
            
    def plot_equity_curve(self, signals: pd.DataFrame):
        """Plot equity curve"""
        if 'equity' not in signals.columns:
            return
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals['equity'],
            mode='lines',
            name='Equity'
        ))
        
        fig.update_layout(
            title='Equity Curve',
            yaxis_title='Equity',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def show_trade_list(self, signals: pd.DataFrame):
        """Display list of trades"""
        if 'trade_type' not in signals.columns:
            return
            
        st.subheader("Trade List")
        
        trades = signals[signals['trade_type'].notna()].copy()
        trades['profit'] = trades['profit'].fillna(0)
        trades['profit_pct'] = trades['profit_pct'].fillna(0)
        
        st.dataframe(
            trades[[
                'trade_type',
                'price',
                'profit',
                'profit_pct'
            ]],
            use_container_width=True
        ) 