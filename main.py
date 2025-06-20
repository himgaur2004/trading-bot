import os
import asyncio
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
from config import Config
from trading_bot import TradingBot

# Load environment variables
load_dotenv()

def run_async(coroutine):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()

def init_bot():
    if 'bot' not in st.session_state:
        try:
            # Initialize configuration
            config = Config()
            
            # Override with environment variables if provided
            if os.getenv('EXCHANGE_NAME'):
                config.exchange.name = os.getenv('EXCHANGE_NAME')
            if os.getenv('TRADING_PAIRS'):
                config.trading.trading_pairs = os.getenv('TRADING_PAIRS').split(',')
                
            # Initialize bot
            st.session_state.bot = TradingBot(config)
            logger.info("Bot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing bot: {e}")
            st.error(f"Failed to initialize bot: {str(e)}")

def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])
    fig.update_layout(title='Price Chart',
                     yaxis_title='Price',
                     xaxis_title='Time')
    return fig

def display_portfolio_metrics(bot):
    try:
        balance = bot.exchange.fetch_balance()
        
        # Portfolio Overview
        st.subheader("Portfolio Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Balance (USDT)", 
                     f"{balance['total']['USDT']:.2f}",
                     f"{balance['free']['USDT']:.2f} Available")
        
        with col2:
            pnl = sum([pos['unrealizedPnl'] for pos in bot.exchange.fetch_positions() if pos['size'] > 0])
            st.metric("Unrealized P&L", 
                     f"{pnl:.2f}",
                     f"{(pnl/balance['total']['USDT']*100):.2f}%")
        
        with col3:
            positions = [pos for pos in bot.exchange.fetch_positions() if pos['size'] > 0]
            st.metric("Active Positions", 
                     len(positions),
                     f"{sum([float(pos['size']) for pos in positions]):.4f} Total Size")
    except Exception as e:
        st.warning(f"Could not fetch portfolio metrics: {str(e)}")

def display_trading_signals(bot, symbol, market_data):
    try:
        st.subheader("Trading Signals")
        
        # Generate signals from each strategy
        signals = {}
        for name, strategy in bot.strategies.items():
            signal_df = strategy.generate_signals(market_data)
            signals[name] = {
                'signal': signal_df['signal'].iloc[-1],
                'strength': signal_df['strength'].iloc[-1]
            }
        
        # Display signals
        cols = st.columns(len(signals))
        for i, (name, signal) in enumerate(signals.items()):
            with cols[i]:
                signal_value = signal['signal'] * signal['strength']
                color = 'green' if signal_value > 0 else 'red' if signal_value < 0 else 'gray'
                st.metric(
                    f"{name.replace('_', ' ').title()}",
                    f"{signal_value:.2f}",
                    f"Strength: {signal['strength']:.2f}",
                    delta_color=color
                )
    except Exception as e:
        st.warning(f"Could not generate trading signals: {str(e)}")

def display_order_book(order_book):
    st.subheader("Order Book")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Bids (Buy Orders)")
        bids_df = pd.DataFrame(order_book['bids'], columns=['Price', 'Size'])
        st.dataframe(bids_df.head())
        
    with col2:
        st.write("Asks (Sell Orders)")
        asks_df = pd.DataFrame(order_book['asks'], columns=['Price', 'Size'])
        st.dataframe(asks_df.head())

def main():
    st.set_page_config(page_title="Crypto Trading Bot Dashboard", layout="wide")
    st.title("Crypto Trading Bot Dashboard")
    
    # Initialize bot
    init_bot()
    
    if 'bot' in st.session_state:
        # Sidebar controls
        st.sidebar.header("Trading Controls")
        
        # Trading pair selection
        selected_pair = st.sidebar.selectbox(
            "Select Trading Pair",
            st.session_state.bot.config.trading.trading_pairs
        )
        
        # Trading actions
        if st.sidebar.button("Start Trading"):
            try:
                run_async(st.session_state.bot.run_once())
                st.sidebar.success("Trading cycle completed successfully")
            except Exception as e:
                st.sidebar.error(f"Trading error: {str(e)}")
        
        # Risk management settings
        st.sidebar.header("Risk Management")
        stop_loss = st.sidebar.slider("Stop Loss %", 0.0, 10.0, 2.0, 0.1)
        take_profit = st.sidebar.slider("Take Profit %", 0.0, 20.0, 5.0, 0.1)
        
        # Main content area
        if selected_pair:
            # Fetch and display market data
            market_data = run_async(st.session_state.bot.fetch_market_data(selected_pair))
            if market_data:
                # Portfolio metrics
                display_portfolio_metrics(st.session_state.bot)
                
                # Price chart
                st.plotly_chart(plot_candlestick(market_data['market']))
                
                # Trading signals
                display_trading_signals(st.session_state.bot, selected_pair, market_data)
                
                # Order book
                display_order_book(market_data['order_book'])
                
                # Recent trades
                st.subheader("Recent Trades")
                trades_df = pd.DataFrame(market_data['trades'])
                if not trades_df.empty:
                    trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='ms')
                    st.dataframe(trades_df[['datetime', 'price', 'amount', 'side']].tail())

if __name__ == "__main__":
    main() 