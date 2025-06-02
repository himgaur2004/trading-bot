import os
import asyncio
from dotenv import load_dotenv
from loguru import logger
from config import Config
from trading_bot import TradingBot

# Load environment variables
load_dotenv()

async def main():
    try:
        # Initialize configuration
        config = Config()
        
        # Override with environment variables if provided
        if os.getenv('EXCHANGE_NAME'):
            config.exchange.name = os.getenv('EXCHANGE_NAME')
        if os.getenv('TRADING_PAIRS'):
            config.trading.trading_pairs = os.getenv('TRADING_PAIRS').split(',')
            
        # Initialize and run bot
        bot = TradingBot(config)
        await bot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logger.add("error.log", level="ERROR", rotation="1 day")
    
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise 