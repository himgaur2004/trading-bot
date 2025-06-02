import asyncio
import os
from dotenv import load_dotenv
from loguru import logger
from backend.services.market_data import MarketDataService
from backend.services.trade_executor import TradeExecutor
from backend.services.risk_manager import RiskManager
from backend.services.strategy_manager import StrategyManager
from backend.database.database import DatabaseHandler

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('COINDCX_API_KEY')
API_SECRET = os.getenv('COINDCX_API_SECRET')
TRADING_PAIRS = os.getenv('TRADING_PAIRS', '').split(',') if os.getenv('TRADING_PAIRS') else []

async def main():
    try:
        logger.info("Initializing trading bot...")
        
        # Initialize services
        market_data = MarketDataService(
            api_key=API_KEY,
            api_secret=API_SECRET,
            symbols=TRADING_PAIRS  # Empty list will trigger fetching all futures pairs
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
                            # Execute trades
                            if signal.get('type') == 'limit':
                                await trade_executor.place_order(
                                    symbol=signal['symbol'],
                                    side=signal['side'],
                                    order_type='limit',
                                    quantity=signal['quantity'],
                                    price=signal['price']
                                )
                            else:
                                await trade_executor.place_order(
                                    symbol=signal['symbol'],
                                    side=signal['side'],
                                    order_type='market',
                                    quantity=signal['quantity']
                                )
                                
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