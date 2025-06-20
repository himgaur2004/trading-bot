import aiohttp
import asyncio
import hmac
import hashlib
import json
import time
import os
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, List, Optional
import requests
from datetime import datetime
from dashboard.utils.data_handler import CoinDCXDataHandler

# Load environment variables
load_dotenv()

API_KEY = os.getenv('COINDCX_API_KEY')
API_SECRET = os.getenv('COINDCX_API_SECRET')

def validate_market_data(market: Dict) -> List[str]:
    """Validate market data and return any warnings."""
    warnings = []
    
    # Check for required fields
    required_fields = ['pair', 'base_currency', 'target_currency', 'last_price']
    missing_fields = [field for field in required_fields if not market.get(field)]
    if missing_fields:
        warnings.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Validate numeric values
    try:
        if float(market.get('last_price', 0)) == 0:
            warnings.append("Zero last price")
        
        if float(market.get('volume', 0)) == 0:
            warnings.append("Zero trading volume")
    except (ValueError, TypeError):
        warnings.append("Invalid numeric values")
        
    return warnings

async def test_public_endpoints():
    """Test public API endpoints that don't require authentication."""
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        try:
            # Test markets endpoint
            logger.info("\nTesting public markets endpoint...")
            async with session.get('https://api.coindcx.com/exchange/v1/markets_details') as response:
                if response.status != 200:
                    logger.error(f"Markets endpoint failed with status {response.status}")
                    return
                    
                markets = await response.json()
                logger.info(f"Markets endpoint response status: {response.status}")
                
                if not markets:
                    logger.error("Empty markets response")
                    return
                    
                if not isinstance(markets, list):
                    logger.error(f"Unexpected markets response type: {type(markets)}")
                    return
                
                # Analyze USDT pairs
                usdt_pairs = [m for m in markets if 'USDT' in m.get('pair', '')]
                logger.info(f"\nFound {len(usdt_pairs)} USDT pairs")
                
                # Detailed market analysis
                for pair in usdt_pairs[:5]:
                    warnings = validate_market_data(pair)
                    logger.info(f"\n• {pair['pair']}:")
                    logger.info(f"  - Last Price: {pair.get('last_price', 'N/A')}")
                    logger.info(f"  - 24h Volume: {pair.get('volume', 'N/A')}")
                    logger.info(f"  - 24h Change: {pair.get('change_24_hour', 'N/A')}%")
                    if warnings:
                        logger.warning(f"  - Warnings: {', '.join(warnings)}")

            # Test ticker endpoint with error handling
            logger.info("\nTesting ticker endpoint...")
            async with session.get('https://api.coindcx.com/exchange/ticker') as response:
                if response.status != 200:
                    logger.error(f"Ticker endpoint failed with status {response.status}")
                    return
                    
                tickers = await response.json()
                
                if not tickers:
                    logger.error("Empty ticker response")
                    return
                    
                # Analyze ticker data
                active_pairs = [t for t in tickers if float(t.get('last_price', 0)) > 0]
                zero_volume_pairs = [t for t in tickers if float(t.get('volume', 0)) == 0]
                
                logger.info(f"\nTicker Analysis:")
                logger.info(f"• Total Pairs: {len(tickers)}")
                logger.info(f"• Active Pairs: {len(active_pairs)}")
                logger.info(f"• Zero Volume Pairs: {len(zero_volume_pairs)}")
                
                # Show some sample ticker data
                logger.info("\nSample Active Pairs (Top by Volume):")
                sorted_pairs = sorted(
                    active_pairs, 
                    key=lambda x: float(x.get('volume', 0)), 
                    reverse=True
                )[:5]
                
                for ticker in sorted_pairs:
                    try:
                        market = ticker.get('market', 'UNKNOWN')
                        last_price = float(ticker.get('last_price', 0))
                        volume = float(ticker.get('volume', 0))
                        bid = float(ticker.get('bid', 0))
                        ask = float(ticker.get('ask', 0))
                        spread = ask - bid if bid > 0 and ask > 0 else 0
                        
                        logger.info(f"\n• {market}:")
                        logger.info(f"  - Last Price: {last_price:.8f}")
                        logger.info(f"  - Volume: {volume:.2f}")
                        logger.info(f"  - Bid/Ask: {bid:.8f}/{ask:.8f}")
                        logger.info(f"  - Spread: {spread:.8f} ({(spread/bid)*100:.2f}% if bid > 0)")
                    except (ValueError, TypeError, ZeroDivisionError) as e:
                        logger.warning(f"Error processing ticker data for {market}: {e}")

        except aiohttp.ClientError as e:
            logger.error(f"Network error testing public endpoints: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing public endpoints: {e}")

async def test_private_endpoints():
    """Test private API endpoints that require authentication."""
    if not API_KEY or not API_SECRET:
        logger.error("\nAPI credentials not found in environment variables")
        logger.info("Please set COINDCX_API_KEY and COINDCX_API_SECRET in your .env file")
        logger.info("Example .env file:")
        logger.info("COINDCX_API_KEY=your_api_key_here")
        logger.info("COINDCX_API_SECRET=your_api_secret_here")
        return

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        try:
            # Prepare authentication
            timestamp = int(time.time() * 1000)
            body = {
                "timestamp": timestamp
            }
            
            signature = hmac.new(
                API_SECRET.encode('utf-8'),
                json.dumps(body, separators=(',', ':')).encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            headers = {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': API_KEY,
                'X-AUTH-SIGNATURE': signature
            }

            # Test account balance endpoint
            logger.info("\nTesting private balance endpoint...")
            async with session.post(
                'https://api.coindcx.com/exchange/v1/users/balances',
                headers=headers,
                json=body
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    logger.error(f"Balance endpoint failed with status {response.status}")
                    logger.error(f"Error response: {error_body}")
                    
                    if response.status == 401:
                        logger.info("\nAuthentication failed. Please check your API credentials.")
                        logger.info("Make sure you have:")
                        logger.info("1. Created API keys on CoinDCX")
                        logger.info("2. Added them to your .env file")
                        logger.info("3. Given appropriate permissions to the API keys")
                    return
                    
                balances = await response.json()
                
                if not isinstance(balances, list):
                    logger.error(f"Unexpected balance response type: {type(balances)}")
                    return
                
                # Analyze balances
                total_balance = 0.0
                logger.info("\nAccount Balances:")
                
                for balance in balances:
                    if float(balance.get('balance', 0)) > 0:
                        currency = balance.get('currency', 'UNKNOWN')
                        amount = float(balance.get('balance', 0))
                        logger.info(f"• {currency}: {amount:.8f}")
                        
                        if currency == 'USDT':
                            total_balance = amount
                
                if total_balance > 0:
                    logger.info(f"\nTotal USDT Balance: {total_balance:.2f}")
                    
                    # Calculate trading parameters based on balance
                    max_position = total_balance * 0.02  # 2% max position size
                    logger.info("\nTrading Parameters:")
                    logger.info(f"• Maximum Position Size (2%): {max_position:.2f} USDT")
                    logger.info(f"• Maximum Leverage: 5x")
                    logger.info(f"• Maximum Leveraged Position: {max_position * 5:.2f} USDT")
                else:
                    logger.warning("\nNo USDT balance found")

        except aiohttp.ClientError as e:
            logger.error(f"Network error testing private endpoints: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing private endpoints: {e}")

async def main():
    """Run all API tests with proper error handling."""
    logger.info("Starting CoinDCX API tests...")
    
    try:
        # Test public endpoints
        await test_public_endpoints()
        
        # Test private endpoints
        await test_private_endpoints()
        
    except Exception as e:
        logger.error(f"Fatal error during API testing: {e}")
    finally:
        logger.info("\nAPI tests completed")

def test_api_connection():
    # API credentials
    api_key = "ea2224143d465699a2269a98a7a5cd0961252b4705e87973"
    api_secret = "e3646cd3e8a59d94d41bedcbd95b20ad6cf2b4fcbe62031fe30927d258e836f0"

    print("Testing CoinDCX API Connection...")
    print("-" * 50)

    try:
        # Initialize the data handler
        handler = CoinDCXDataHandler(api_key, api_secret)
        
        # Test account info
        print("\nFetching account information...")
        account_info = handler.get_account_info()
        print("Account Info:")
        print(f"Name: {account_info.get('name', 'N/A')}")
        print(f"Email: {account_info.get('email', 'N/A')}")
        print(f"Status: {account_info.get('status', 'N/A')}")
        
        # Test balances
        print("\nFetching account balances...")
        balances = handler.get_balances()
        print("\nNon-zero balances:")
        for balance in balances:
            if float(balance['balance']) > 0:
                print(f"Currency: {balance['currency']}")
                print(f"Balance: {balance['balance']}")
                print(f"Locked: {balance['locked_balance']}")
                print("-" * 30)
        
        print("\nAPI connection test completed successfully!")
        
    except Exception as e:
        print(f"\nError testing API connection: {str(e)}")
        print("\nPlease check:")
        print("1. API key and secret are correct")
        print("2. Internet connection is working")
        print("3. API has necessary permissions")

if __name__ == "__main__":
    test_api_connection() 