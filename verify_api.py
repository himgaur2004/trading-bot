import os
import hmac
import hashlib
import json
import time
import aiohttp
import asyncio
from dotenv import load_dotenv
from loguru import logger

# Configure logger to output to console
logger.remove()  # Remove default handler
logger.add(lambda msg: print(msg, flush=True), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

async def test_api_connection():
    """Test CoinDCX API connection with credentials."""
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('COINDCX_API_KEY')
    api_secret = os.getenv('COINDCX_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("API credentials not found in .env file")
        logger.info("Please ensure your .env file contains:")
        logger.info("COINDCX_API_KEY=your_api_key")
        logger.info("COINDCX_API_SECRET=your_api_secret")
        return False
        
    # Prepare authentication
    timestamp = int(time.time() * 1000)
    body = {
        "timestamp": timestamp
    }
    
    try:
        # Generate signature
        signature = hmac.new(
            api_secret.encode('utf-8'),
            json.dumps(body, separators=(',', ':')).encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': api_key,
            'X-AUTH-SIGNATURE': signature
        }
        
        logger.info("Testing API connection...")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"API Key (first/last 4 chars): {api_key[:4]}...{api_key[-4:]}")
        
        async with aiohttp.ClientSession() as session:
            # Test private endpoint
            logger.info("\nTesting private API endpoint...")
            async with session.post(
                'https://api.coindcx.com/exchange/v1/users/info',
                headers=headers,
                json=body,
                ssl=False
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("\n✅ API connection successful!")
                    logger.info(f"Account verified for user: {data.get('username', 'Unknown')}")
                    logger.info("Your API keys have the correct permissions")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"\n❌ API connection failed with status {response.status}")
                    logger.error(f"Error: {error_text}")
                    
                    if response.status == 401:
                        logger.info("\nPossible issues:")
                        logger.info("1. API key or secret may be incorrect")
                        logger.info("2. API keys may not have the required permissions")
                        logger.info("3. The request may be improperly signed")
                        
                        # Additional debugging info
                        logger.info("\nRequest details:")
                        logger.info(f"Headers: {json.dumps(headers, indent=2)}")
                        logger.info(f"Body: {json.dumps(body, indent=2)}")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ Error testing API connection: {e}")
        return False

async def main():
    """Main function to run API tests."""
    logger.info("Starting API verification...")
    success = await test_api_connection()
    
    if success:
        logger.info("\nNext steps:")
        logger.info("1. You can now proceed with trading")
        logger.info("2. Make sure to set appropriate risk parameters")
        logger.info("3. Start with small test trades")
    else:
        logger.info("\nTroubleshooting steps:")
        logger.info("1. Double-check your API credentials in .env")
        logger.info("2. Verify API key permissions on CoinDCX")
        logger.info("3. Ensure your system time is synchronized")
        logger.info("4. Check your internet connection")

if __name__ == "__main__":
    asyncio.run(main()) 