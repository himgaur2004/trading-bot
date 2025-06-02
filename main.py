import os
import asyncio
import streamlit as st
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

def main():
    st.title("Crypto Trading Bot Dashboard")
    
    # Initialize bot
    init_bot()
    
    if 'bot' in st.session_state:
        # Add your Streamlit UI components here
        st.sidebar.header("Trading Controls")
        
        if st.sidebar.button("Start Trading"):
            try:
                # Run the bot's main loop once
                run_async(st.session_state.bot.run_once())  # We'll modify trading_bot.py to add this method
                st.success("Trading cycle completed successfully")
            except Exception as e:
                logger.error(f"Error during trading: {e}")
                st.error(f"Trading error: {str(e)}")
        
        # Add more UI components as needed
        st.sidebar.header("Market Data")
        selected_pair = st.sidebar.selectbox(
            "Select Trading Pair",
            st.session_state.bot.config.trading.trading_pairs
        )
        
        if selected_pair:
            market_data = run_async(st.session_state.bot.fetch_market_data(selected_pair))
            if market_data:
                st.write("### Market Overview")
                st.write(market_data['market'].tail())

if __name__ == "__main__":
    main() 