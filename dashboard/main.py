import os
import sys
from pathlib import Path
import json
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Local imports
from utils.data_handler import CoinDCXDataHandler
from utils.strategy_ai import StrategyAI, MarketCondition
from utils.trailing import TrailingOrderManager, TrailingConfig
from utils.config_pool import MarketConfigPool, PoolConfig
from utils.strategy_handler import StrategyHandler, StrategyType
from dashboard.components.strategy_viz import StrategyVisualizer
from config import API_KEY, API_SECRET
from multi_pair_trading import MultiPairTradingBot

# Set page config
st.set_page_config(
    page_title="Binance Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stDataFrame { background-color: #181c24; }
    .bullish { color: #00ff99 !important; font-weight: bold; }
    .bearish { color: #ff4b4b !important; font-weight: bold; }
    .star { color: gold; font-size: 1.2em; }
    .status-on { color: #00ff99; font-weight: bold; }
    .status-off { color: #ff4b4b; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("⚙️ Controls")
auto_mode = st.sidebar.toggle("Auto Trading Mode", value=False)
scan_interval = st.sidebar.slider("Scan Interval (sec)", 10, 300, 60, 10)

# Status indicator
if auto_mode:
    st.sidebar.markdown('<span class="status-on">🟢 Auto Mode ON</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-off">🔴 Auto Mode OFF</span>', unsafe_allow_html=True)

# Initialize bot
if 'binance_bot' not in st.session_state:
    st.session_state.binance_bot = MultiPairTradingBot()
    st.session_state.binance_bot.load_working_pairs()

bot = st.session_state.binance_bot

# Main dashboard
st.title("📊 Binance Trading Dashboard")

# Debug: Show loaded pairs
st.write(f"Loaded {len(bot.working_pairs)} pairs:", bot.working_pairs[:10])

if not bot.working_pairs:
    st.error("No Binance pairs loaded. Check your API/network.")

# Use st_autorefresh for auto mode
if auto_mode:
    st_autorefresh = getattr(st, 'autorefresh', None)
    if st_autorefresh:
        st_autorefresh(interval=scan_interval * 1000, key="refresh")

# Scan and get opportunities
def get_opportunities():
    try:
        bot.candle_data = bot.fetch_all_binance_candles(bot.working_pairs[:50], interval='15m', limit=100)
        analyses = bot.analyze_all_pairs(bot.candle_data)
        st.write(f"Fetched {len(analyses)} analyses. Sample:", analyses[:2])
        opportunities = bot.filter_trading_opportunities(analyses)
        st.write(f"Opportunities found: {len(opportunities)}. Sample:", opportunities[:2])
        # Only bullish/bearish
        filtered = [opp for opp in opportunities if opp.get('direction') in ['bullish', 'bearish']]
        # Add star for best
        if filtered:
            best = max(filtered, key=lambda x: x.get('signal_count', 0))
            for opp in filtered:
                opp['star'] = (opp is best)
        return filtered, analyses
    except Exception as e:
        st.error(f"Error fetching opportunities: {e}")
        return [], []

# Get and display opportunities
data, all_analyses = get_opportunities()

# DataFrame for table
if data:
    df = pd.DataFrame([
        {
            '★': '★' if opp.get('star') else '',
            'Pair': opp['pair'],
            'Price': opp['current_price'],
            'Direction': '↑ Bullish' if opp['direction']=='bullish' else '↓ Bearish',
            'Best Strategy': opp.get('best_strategy','-'),
            'SL': opp.get('sl','-'),
            'TP': opp.get('tp','-'),
            'Lev': opp.get('lev','-'),
            'Signals': ', '.join(opp['signals'].keys()) if opp['signals'] else '-',
            'RSI': opp['indicators']['rsi'],
            'MACD': opp['indicators']['macd'],
        } for opp in data
    ])
    # Color and icon formatting
    def highlight_direction(val):
        if 'Bullish' in val:
            return 'bullish'
        elif 'Bearish' in val:
            return 'bearish'
        return ''
    st.dataframe(df.style.applymap(highlight_direction, subset=['Direction']), use_container_width=True, hide_index=True)
    # Click to expand chart
    st.subheader("🔎 Click a pair for details and chart:")
    selected = st.selectbox("Select Pair", df['Pair'])
    opp = next((o for o in data if o['pair']==selected), None)
    if opp:
        st.markdown(f"### {opp['pair']} {'★' if opp.get('star') else ''}")
        st.write(f"**Direction:** {'↑ Bullish' if opp['direction']=='bullish' else '↓ Bearish'}  ")
        st.write(f"**Best Strategy:** {opp.get('best_strategy','-')}  ")
        st.write(f"**SL:** {opp.get('sl','-')}  |  **TP:** {opp.get('tp','-')}  |  **Leverage:** {opp.get('lev','-')}")
        st.write(f"**Signals:** {', '.join(opp['signals'].keys()) if opp['signals'] else '-'}")
        # Chart
        df_chart = bot.candle_data[selected]
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart['open'],
            high=df_chart['high'],
            low=df_chart['low'],
            close=df_chart['close'],
            name='Price'))
        # SL/TP lines
        fig.add_hline(y=opp.get('sl'), line_dash="dash", line_color="#ff4b4b", annotation_text="SL", annotation_position="bottom right")
        fig.add_hline(y=opp.get('tp'), line_dash="dash", line_color="#00ff99", annotation_text="TP", annotation_position="top right")
        fig.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No actionable trading opportunities found. Showing all analyses for debugging:")
    st.write(all_analyses[:5])

# Performance metrics (mock)
st.subheader("Performance Metrics (Mock)")
st.metric("Total Trades", value=len(data) if data else 0)
st.metric("Best Win Rate", value=f"{max([1 for d in data if d.get('star')], default=0)*100:.1f}%" if data else "0%")

def show_api_status():
    """Display API connection status."""
    if st.session_state.data_handler is None:
        st.error("❌ API Connection Failed")
        st.error("Please check your API credentials in config.py")
        
        # Show current API key (masked)
        masked_key = API_KEY[:4] + '*' * (len(API_KEY) - 8) + API_KEY[-4:] if API_KEY else ''
        st.code(f"""Current API Settings:
API Key: {masked_key}
API Secret: {'*' * 10}

To fix this:
1. Update API credentials in config.py
2. Ensure API key is active on CoinDCX
3. Check API permissions""")
        return False
    return True

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
    st.title("📊 Crypto Trading Dashboard (Binance)")
    # Use Binance data only
    try:
        # Show top trading opportunities
        analyses = bot.analyze_all_pairs(bot.candle_data)
        opportunities = bot.filter_trading_opportunities(analyses)
        bot.print_trading_opportunities(opportunities, limit=10)
    except Exception as e:
        st.error(f"Error loading Binance data: {e}")

def show_portfolio():
    st.title("💼 Portfolio")
    
    try:
        # Fetch real account data
        balances = st.session_state.data_handler.get_balances()
        market_data = st.session_state.data_handler.get_market_data()
        
        # Calculate portfolio composition
        portfolio_data = {
            'Asset': [],
            'Value': []
        }
        
        total_value = 0
        for balance in balances:
            if float(balance['balance']) > 0:
                currency = balance['currency']
                amount = float(balance['balance'])
                
                if currency == 'USDT':
                    value = amount
                else:
                    # Find the USDT pair for this currency
                    pair = f"{currency}USDT"
                    price_data = market_data[market_data['pair'] == pair]
                    if not price_data.empty:
                        price = float(price_data.iloc[0]['last_price'])
                        value = amount * price
                    else:
                        continue
                
                portfolio_data['Asset'].append(currency)
                portfolio_data['Value'].append(value)
                total_value += value
        
        if portfolio_data['Asset']:
            st.subheader("Asset Allocation")
            
            fig = go.Figure(data=[go.Pie(
                labels=portfolio_data['Asset'],
                values=portfolio_data['Value'],
                hole=.3,
                hovertemplate="<b>%{label}</b><br>" +
                            "Value: $%{value:,.2f}<br>" +
                            "Percentage: %{percent:.1%}<extra></extra>"
            )])
            
            fig.update_layout(
                template='plotly_dark',
                title=f'Total Portfolio Value: ${total_value:,.2f}',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed balance table
            st.subheader("Asset Details")
            details_data = []
            for asset, value in zip(portfolio_data['Asset'], portfolio_data['Value']):
                balance_info = next(b for b in balances if b['currency'] == asset)
                details_data.append({
                    'Asset': asset,
                    'Balance': float(balance_info['balance']),
                    'Value (USDT)': value,
                    'Allocation': f"{(value/total_value*100):.2f}%"
                })
            
            st.dataframe(
                pd.DataFrame(details_data),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No assets found in your portfolio")
            
    except Exception as e:
        st.error(f"Error fetching portfolio data: {str(e)}")
        st.error("Please check your API credentials and internet connection")

def show_strategies():
    st.title("🎯 Trading Strategies")
    
    # Initialize components
    strategy_viz = StrategyVisualizer(st.session_state.strategy_handler)
    
    # Add AI Strategy Selection
    st.subheader("AI Strategy Selection")
    
    if 'strategy_ai' not in st.session_state:
        st.session_state.strategy_ai = StrategyAI()
        
    # Market condition analysis
    if st.button("Analyze Market Condition"):
        try:
            # Get market data
            data = st.session_state.data_handler.get_market_data("BTC-USDT")
            
            # Analyze market condition
            condition = st.session_state.strategy_ai.analyze_market_condition(data)
            st.info(f"Current Market Condition: {condition.value}")
            
            # Get suitable strategy
            strategy = st.session_state.strategy_ai.select_best_strategy(data)
            if strategy:
                st.success(f"Selected Strategy: {st.session_state.strategy_ai.get_current_strategy()}")
            else:
                st.warning("No suitable strategy found for current conditions")
                
        except Exception as e:
            st.error(f"Error analyzing market: {str(e)}")
            
    # Trailing Settings
    st.subheader("Trailing Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        activation_pct = st.number_input(
            "Activation Percentage",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.1,
            help="Percentage from entry to activate trailing"
        )
        
        callback_rate = st.number_input(
            "Callback Rate",
            min_value=0.0,
            max_value=100.0,
            value=0.5,
            step=0.1,
            help="How far price can move against position before triggering"
        )
        
    with col2:
        step_size = st.number_input(
            "Step Size",
            min_value=0.00001,
            max_value=1.0,
            value=0.001,
            format="%.5f",
            help="Minimum price movement increment"
        )
        
        arm_price = st.number_input(
            "Arm Price (Optional)",
            min_value=0.0,
            value=0.0,
            help="Price at which trailing becomes active (0 for automatic)"
        )
        
    # Config Pools
    st.subheader("Config Pools")
    
    if 'config_pool' not in st.session_state:
        st.session_state.config_pool = MarketConfigPool({})  # Use empty dict or default config
    
    market_condition = st.selectbox(
        "Select Market Condition",
        ["trending", "ranging", "volatile"]
    )
    
    if st.button("Apply Config Pool"):
        try:
            config = st.session_state.config_pool.get_config(market_condition)
            st.session_state.strategy_handler.update_config(config)
            st.success(f"Applied {market_condition} market configuration")
        except Exception as e:
            st.error(f"Error applying config: {str(e)}")
            
    # Show current config
    if st.checkbox("Show Current Configuration"):
        config = st.session_state.strategy_handler.get_current_config()
        st.json(config)
        
    # Original strategy selector
    selected_strategy = strategy_viz.show_strategy_selector()
    parameters = strategy_viz.show_strategy_parameters(selected_strategy)
    
    # Trading pair selector with updated format for CoinDCX
    pair = st.selectbox(
        "Select Trading Pair",
        ["BTC-USDT", "ETH-USDT", "XRP-USDT", "SOL-USDT"]
    )
    
    # Timeframe selector
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    
    # Run strategy button
    if st.button("Run Strategy Backtest"):
        with st.spinner("Fetching data and running backtest..."):
            try:
                # Fetch historical data
                data = st.session_state.data_handler.get_market_data(pair)
                if data and 'candles' in data:
                    data = data['candles']
                else:
                    data = None
                
                # Add strategy
                strategy_name = f"{selected_strategy}_{pair}_{timeframe}"
                st.session_state.strategy_handler.add_strategy(
                    strategy_name,
                    parameters
                )
                
                # Get signals and performance
                signals = st.session_state.strategy_handler.get_signals(data)
                metrics = st.session_state.strategy_handler.get_performance(signals)
                
                # Display results
                strategy_viz.plot_results(data, signals)
                strategy_viz.show_metrics(metrics)
                strategy_viz.plot_equity_curve(signals)
                strategy_viz.show_trade_list(signals)
                
            except Exception as e:
                st.error(f"Error running strategy: {str(e)}")

def show_market_analysis():
    st.title("📊 Market Analysis")

    # Market overview
    st.subheader("Market Overview")

    # Fetch all market data
    try:
        market_data = st.session_state.data_handler.get_market_data()
        if market_data is None or market_data.empty:
            st.warning("No market data available.")
            return
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return

    # Show top 10 pairs by volume
    if 'volume' in market_data.columns:
        top_pairs = market_data.sort_values('volume', ascending=False).head(10)
    else:
        top_pairs = market_data.head(10)
    st.dataframe(
        top_pairs[[c for c in ['pair', 'last_price', 'change_24_hour', 'volume'] if c in top_pairs.columns]],
        use_container_width=True,
        hide_index=True
    )

    # Candlestick chart and technical indicators
    st.subheader("Candlestick & Technical Indicators")
    pair_list = market_data['pair'].tolist() if 'pair' in market_data.columns else []
    default_pair = 'BTC-USDT' if 'BTC-USDT' in pair_list else (pair_list[0] if pair_list else None)

    # Filter pairs to only those with candle data available
    available_pairs = []
    for pair in pair_list:
        try:
            pair_data = st.session_state.data_handler.get_market_data(pair)
            df = pair_data['candles'] if pair_data and 'candles' in pair_data else None
            if df is not None and not df.empty:
                available_pairs.append(pair)
        except Exception:
            continue
    if not available_pairs:
        st.warning("No pairs with candle data available.")
        return
    if default_pair not in available_pairs:
        default_pair = available_pairs[0]
    pair = st.selectbox("Select Pair for Analysis", available_pairs, index=available_pairs.index(default_pair) if default_pair in available_pairs else 0)

    # Fetch candle data for selected pair
    with st.spinner(f"Loading data for {pair}..."):
        try:
            pair_data = st.session_state.data_handler.get_market_data(pair)
            df = pair_data['candles'] if pair_data and 'candles' in pair_data else None
        except Exception as e:
            st.error(f"Error fetching candle data: {str(e)}")
            df = None

    if df is not None and not df.empty:
        # Candlestick chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=pair
        ))
        fig.update_layout(
            template='plotly_dark',
            title=f'Market Overview ({pair})',
            xaxis_title='Date',
            yaxis_title='Price (USDT)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Technical indicators
        st.subheader("Technical Indicators")
        indicators_col1, indicators_col2, indicators_col3 = st.columns(3)
        # Calculate RSI
        try:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1] if not rsi.isna().all() else None
        except Exception:
            rsi_val = None
        # Calculate MACD
        try:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_val = macd.iloc[-1] if not macd.isna().all() else None
            signal_val = signal.iloc[-1] if not signal.isna().all() else None
        except Exception:
            macd_val = None
            signal_val = None
        # Calculate MA Cross
        try:
            ma_fast = df['close'].rolling(window=50).mean()
            ma_slow = df['close'].rolling(window=200).mean()
            if not ma_fast.isna().all() and not ma_slow.isna().all():
                if ma_fast.iloc[-1] > ma_slow.iloc[-1]:
                    ma_cross = "Golden Cross"
                else:
                    ma_cross = "Death Cross"
            else:
                ma_cross = "N/A"
        except Exception:
            ma_cross = "N/A"
        with indicators_col1:
            st.metric("RSI (14)", f"{rsi_val:.2f}" if rsi_val is not None else "N/A")
        with indicators_col2:
            macd_str = f"MACD: {macd_val:.2f}, Signal: {signal_val:.2f}" if macd_val is not None and signal_val is not None else "N/A"
            st.metric("MACD", macd_str)
        with indicators_col3:
            st.metric("MA Cross", ma_cross)
    else:
        st.warning(f"No candle data available for {pair}")

def show_settings():
    st.title("⚙️ Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    
    # Show current API key (masked)
    current_key = st.session_state.get('api_key', '')
    masked_key = current_key[:4] + '*' * (len(current_key) - 8) + current_key[-4:] if current_key else ''
    st.text(f"Current API Key: {masked_key}")
    
    # Input fields for new credentials
    new_api_key = st.text_input("New API Key", type="password")
    new_api_secret = st.text_input("New API Secret", type="password")
    
    if st.button("Update API Credentials"):
        if new_api_key and new_api_secret:
            try:
                # Test new credentials
                test_handler = CoinDCXDataHandler(new_api_key, new_api_secret)
                test_handler.get_account_info()  # Test the connection
                
                # Update session state
                st.session_state.api_key = new_api_key
                st.session_state.api_secret = new_api_secret
                
                # Reinitialize data handler with new credentials
                st.session_state.data_handler = test_handler
                
                st.success("API credentials updated successfully!")
                st.rerun()  # Rerun the app to refresh all components
            except Exception as e:
                st.error(f"Failed to validate new credentials: {str(e)}")
        else:
            st.warning("Please provide both API Key and Secret")
    
    # Risk Management
    st.subheader("Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_position = st.number_input("Max Position Size (%)", 
                                     value=st.session_state.get('max_position', 5.0),
                                     min_value=0.1,
                                     max_value=100.0)
        stop_loss = st.number_input("Stop Loss (%)", 
                                   value=st.session_state.get('stop_loss', 2.0),
                                   min_value=0.1,
                                   max_value=100.0)
    
    with col2:
        take_profit = st.number_input("Take Profit (%)", 
                                     value=st.session_state.get('take_profit', 6.0),
                                     min_value=0.1,
                                     max_value=100.0)
        max_daily_loss = st.number_input("Max Daily Loss (%)", 
                                        value=st.session_state.get('max_daily_loss', 10.0),
                                        min_value=0.1,
                                        max_value=100.0)
    
    if st.button("Save Risk Settings"):
        # Save risk management settings to session state
        st.session_state.max_position = max_position
        st.session_state.stop_loss = stop_loss
        st.session_state.take_profit = take_profit
        st.session_state.max_daily_loss = max_daily_loss
        st.success("Risk management settings saved successfully!")

if __name__ == "__main__":
    main() 