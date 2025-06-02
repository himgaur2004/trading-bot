import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from dashboard.utils.data_handler import CoinDCXDataHandler
from dashboard.utils.strategy_handler import StrategyHandler, StrategyType
from dashboard.components.strategy_viz import StrategyVisualizer

# Set page config
st.set_page_config(
    page_title="CryptoTrading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTabs > div > div > div > div {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'strategy_handler' not in st.session_state:
    st.session_state.strategy_handler = StrategyHandler()

if 'data_handler' not in st.session_state:
    st.session_state.data_handler = None

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Portfolio", "Trading Strategies", "Market Analysis", "Settings"]
    )

    # Main content
    if page == "Dashboard":
        show_dashboard()
    elif page == "Portfolio":
        show_portfolio()
    elif page == "Trading Strategies":
        show_strategies()
    elif page == "Market Analysis":
        show_market_analysis()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    st.title("📊 Crypto Trading Dashboard")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Portfolio Value",
            value="$10,243.50",
            delta="↑ $142.80 (1.4%)"
        )
    
    with col2:
        st.metric(
            label="24h Trading Volume",
            value="$5,432.20",
            delta="↓ $230.45 (4.2%)"
        )
    
    with col3:
        st.metric(
            label="Active Positions",
            value="8",
            delta="↑ 2"
        )
    
    with col4:
        st.metric(
            label="Profit/Loss (24h)",
            value="$142.80",
            delta="1.4%"
        )

    # Charts section
    st.subheader("Portfolio Performance")
    
    # Sample data for the chart
    dates = pd.date_range(start='2024-01-01', end='2024-01-14', freq='D')
    values = np.random.normal(10000, 200, len(dates)).cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00ff00', width=2)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Value (USD)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Active trades
    st.subheader("Active Trades")
    
    trades_data = {
        'Pair': ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT'],
        'Position': ['Long', 'Short', 'Long', 'Long'],
        'Entry Price': ['$42,150', '$2,250', '$0.58', '$95.20'],
        'Current Price': ['$42,800', '$2,200', '$0.62', '$98.50'],
        'PnL': ['↑ 1.54%', '↑ 2.22%', '↑ 6.90%', '↑ 3.47%']
    }
    
    st.dataframe(
        pd.DataFrame(trades_data),
        use_container_width=True,
        hide_index=True
    )

def show_portfolio():
    st.title("💼 Portfolio")
    
    # Asset allocation chart
    st.subheader("Asset Allocation")
    
    # Sample portfolio data
    portfolio_data = {
        'Asset': ['BTC', 'ETH', 'XRP', 'SOL', 'USDT'],
        'Value': [5000, 3000, 1000, 800, 443.50]
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=portfolio_data['Asset'],
        values=portfolio_data['Value'],
        hole=.3
    )])
    
    fig.update_layout(
        template='plotly_dark',
        title='Portfolio Distribution',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_strategies():
    st.title("🎯 Trading Strategies")
    
    # Initialize strategy visualizer
    strategy_viz = StrategyVisualizer(st.session_state.strategy_handler)
    
    # Strategy selector
    selected_strategy = strategy_viz.show_strategy_selector()
    
    # Get strategy parameters
    parameters = strategy_viz.show_strategy_parameters(selected_strategy)
    
    # Trading pair selector
    pair = st.selectbox(
        "Select Trading Pair",
        ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"]
    )
    
    # Timeframe selector
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    
    # Fetch data button
    if st.button("Run Strategy Backtest"):
        with st.spinner("Fetching data and running backtest..."):
            try:
                # Initialize data handler if not already done
                if not st.session_state.data_handler:
                    api_key = st.secrets["COINDCX_API_KEY"]
                    api_secret = st.secrets["COINDCX_API_SECRET"]
                    st.session_state.data_handler = CoinDCXDataHandler(api_key, api_secret)
                
                # Fetch historical data
                data = st.session_state.data_handler.get_candles(pair, timeframe)
                
                # Add strategy to handler
                strategy_name = f"{selected_strategy}_{pair}_{timeframe}"
                st.session_state.strategy_handler.add_strategy(
                    strategy_name,
                    StrategyType(selected_strategy),
                    parameters
                )
                
                # Get signals and performance metrics
                signals = st.session_state.strategy_handler.get_strategy_signals(strategy_name, data)
                metrics = st.session_state.strategy_handler.get_strategy_performance(strategy_name, signals)
                
                # Display results
                strategy_viz.plot_strategy_results(data, signals)
                strategy_viz.show_performance_metrics(metrics)
                strategy_viz.plot_equity_curve(signals)
                strategy_viz.show_trade_list(signals)
                
            except Exception as e:
                st.error(f"Error running strategy: {str(e)}")

def show_market_analysis():
    st.title("📊 Market Analysis")
    
    # Market overview
    st.subheader("Market Overview")
    
    # Sample market data
    market_data = {
        'Pair': ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT'],
        'Price': ['$42,800', '$2,200', '$0.62', '$98.50'],
        'Change 24h': ['↑ 1.54%', '↓ 0.22%', '↑ 3.90%', '↑ 2.47%'],
        'Volume 24h': ['$28.5B', '$12.2B', '$2.1B', '$1.8B']
    }
    
    st.dataframe(
        pd.DataFrame(market_data),
        use_container_width=True,
        hide_index=True
    )
    
    # Technical indicators
    st.subheader("Technical Indicators")
    
    indicators_col1, indicators_col2, indicators_col3 = st.columns(3)
    
    with indicators_col1:
        st.metric("RSI (14)", "58", "Neutral")
    
    with indicators_col2:
        st.metric("MACD", "Bullish", "↑")
    
    with indicators_col3:
        st.metric("MA Cross", "Golden Cross", "↑")

def show_settings():
    st.title("⚙️ Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret", type="password")
    
    # Risk Management
    st.subheader("Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("Max Position Size (%)", value=5.0)
        st.number_input("Stop Loss (%)", value=2.0)
    
    with col2:
        st.number_input("Take Profit (%)", value=6.0)
        st.number_input("Max Daily Loss (%)", value=10.0)
    
    # Save settings button
    if st.button("Save Settings"):
        if api_key and api_secret:
            # Initialize data handler with new credentials
            st.session_state.data_handler = CoinDCXDataHandler(api_key, api_secret)
            st.success("Settings saved successfully!")
        else:
            st.error("Please provide both API Key and Secret")

if __name__ == "__main__":
    main() 