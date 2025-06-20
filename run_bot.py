import asyncio
import os
from dotenv import load_dotenv
from loguru import logger
from backend.services.market_data import MarketDataService
from backend.services.trade_executor import TradeExecutor
from backend.services.risk_manager import RiskManager
from backend.services.strategy_manager import StrategyManager
from backend.database.database import DatabaseHandler
import requests
import time
import json

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('COINDCX_API_KEY')
API_SECRET = os.getenv('COINDCX_API_SECRET')
TRADING_PAIRS = os.getenv('TRADING_PAIRS', '').split(',') if os.getenv('TRADING_PAIRS') else []

async def main():
    try:
        logger.info("Initializing trading bot...")
        
        # Fetch all available trading pairs from CoinDCX
        markets_response = requests.get("https://api.coindcx.com/exchange/v1/markets")
        markets = set(markets_response.json())

        # Fetch all pairs with active ticker data
        tickers_response = requests.get("https://api.coindcx.com/exchange/ticker")
        tickers = tickers_response.json()
        active_pairs = set(t['market'] for t in tickers if t['market'].endswith('USDT'))

        # Use only pairs that are both listed and have active ticker data
        valid_pairs = list(markets & active_pairs)
        logger.info(f"Scanning {len(valid_pairs)} valid pairs with active ticker data: {valid_pairs}")

        # Pre-scan: filter for pairs with available candle data
        def filter_pairs_with_candles(pairs):
            valid = []
            for symbol in pairs:
                url = f"https://public.coindcx.com/market_data/candles?pair={symbol}&interval=1m&limit=1"
                resp = requests.get(url)
                try:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        valid.append(symbol)
                except Exception:
                    pass
                time.sleep(0.1)  # To avoid rate limits
            return valid

        valid_pairs = filter_pairs_with_candles(valid_pairs)
        logger.info(f"Final scan list with candle data: {valid_pairs}")

        # Write the final scan list to a file for the frontend
        with open("scanned_pairs.json", "w") as f:
            json.dump(valid_pairs, f)

        # Initialize services
        market_data = MarketDataService(
            api_key=API_KEY,
            api_secret=API_SECRET,
            symbols=valid_pairs  # Use only valid pairs
        )
        
        trade_executor = TradeExecutor(
            api_key=API_KEY,
            api_secret=API_SECRET
        )
        
        risk_manager = RiskManager()
        strategy_manager = StrategyManager()
        db = DatabaseHandler()
        
        async with market_data:
            # Initialize market data service
            await market_data.initialize()
            logger.info("Market data service initialized")
            
            # Add market data callback
            async def on_market_data(symbol: str, data: dict):
                try:
                    # Generate trading signals
                    signals = await strategy_manager.generate_signals(data, symbol)
                    
                    if signals:
                        # Validate signals with risk manager
                        valid_signals = risk_manager.validate_signals(signals)
                        
                        for signal in valid_signals:
                            # Instead of executing trades, just log the trade name/strategy
                            trade_name = signal.get('strategy', signal.get('type', 'Unknown'))
                            logger.info(f"Trade signal found for {symbol} by strategy: {trade_name}")
                            logger.info(f"Signal details: {signal}")
                        # No trade execution here
                except Exception as e:
                    logger.error(f"Error processing market data: {e}")
                    
            market_data.add_callback(on_market_data)
            
            # Start market data service
            await market_data.start()
            
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        raise
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise 