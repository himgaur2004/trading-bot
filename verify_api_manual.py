import os
import hmac
import hashlib
import json
import time
import aiohttp
import asyncio
import argparse
from loguru import logger

# Configure logger to output to console
logger.remove()  # Remove default handler
logger.add(lambda msg: print(msg, flush=True), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

class CoinDCXAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coindcx.com"

    async def generate_signature(self, body: dict) -> str:
        """Generate signature for API request."""
        # Sort the body parameters to ensure consistent ordering
        sorted_body = dict(sorted(body.items()))
        message = json.dumps(sorted_body, separators=(',', ':'))
        return hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def make_request(self, endpoint: str, body: dict = None, method: str = "POST") -> tuple:
        """Make an authenticated request to CoinDCX API."""
        if body is None:
            body = {}
        
        # Add timestamp if not present
        if 'timestamp' not in body:
            body['timestamp'] = str(int(time.time() * 1000))
        
        # Generate signature
        signature = await self.generate_signature(body)
        
        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-SIGNATURE': signature
        }
        
        url = f"{self.base_url}{endpoint}"
        logger.info(f"\nMaking request to: {url}")
        logger.info("Request details:")
        logger.info(f"Headers: {json.dumps(headers, indent=2)}")
        logger.info(f"Body: {json.dumps(body, indent=2)}")
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, json=body, ssl=False) as response:
                return response.status, await response.text()

    async def test_public_endpoint(self) -> bool:
        """Test public endpoint access."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/exchange/v1/markets_details", ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Public API accessible")
                    logger.info(f"Found {len(data)} markets")
                    return True
                logger.error("❌ Cannot access public API")
                return False

    async def test_private_endpoints(self) -> bool:
        """Test private endpoint access."""
        endpoints = [
            {
                'name': 'User Info',
                'endpoint': '/exchange/v1/users/info',
                'method': 'POST'
            },
            {
                'name': 'Balances',
                'endpoint': '/exchange/v1/users/balances',
                'method': 'POST'
            },
            {
                'name': 'Account Details',
                'endpoint': '/exchange/v1/users/account',
                'method': 'POST'
            }
        ]

        success = False
        for endpoint in endpoints:
            logger.info(f"\nTesting {endpoint['name']} endpoint...")
            status, response_text = await self.make_request(
                endpoint['endpoint'],
                {'timestamp': str(int(time.time() * 1000))},
                endpoint['method']
            )

            try:
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response: {response_text}")
                continue

            if status == 200:
                logger.info(f"✅ {endpoint['name']} endpoint successful!")
                logger.info(f"Response: {json.dumps(response_json, indent=2)}")
                success = True
            else:
                error_message = response_json.get('message', 'Unknown error')
                error_code = response_json.get('code', status)
                logger.error(f"❌ {endpoint['name']} endpoint failed with status {status}")
                logger.error(f"Error {error_code}: {error_message}")

                if status == 401:
                    logger.info("\nTrying alternative timestamp format...")
                    status, response_text = await self.make_request(
                        endpoint['endpoint'],
                        {'timestamp': int(time.time() * 1000)},
                        endpoint['method']
                    )
                    if status == 200:
                        logger.info(f"✅ Alternative format worked for {endpoint['name']}!")
                        success = True

        return success

async def test_api_connection(api_key: str, api_secret: str):
    """Test CoinDCX API connection with credentials."""
    if not api_key or not api_secret:
        logger.error("API credentials not provided")
        return False

    api = CoinDCXAPI(api_key, api_secret)
    
    # Test public endpoint first
    if not await api.test_public_endpoint():
        return False

    # Test private endpoints
    if not await api.test_private_endpoints():
        logger.error("\n❌ All authentication attempts failed")
        logger.info("\nPossible issues:")
        logger.info("1. API key or secret may be incorrect")
        logger.info("2. API keys may not have the required permissions")
        logger.info("3. IP address may not be whitelisted")
        logger.info("4. System time may be out of sync")
        logger.info("\nTroubleshooting steps:")
        logger.info("1. Generate new API keys with all required permissions")
        logger.info("2. Make sure your IP address is whitelisted")
        logger.info("3. Verify your system time is accurate")
        logger.info("4. Check CoinDCX API status: https://status.coindcx.com")
        return False

    return True

async def main():
    """Main function to run API tests."""
    parser = argparse.ArgumentParser(description='Test CoinDCX API connection')
    parser.add_argument('--key', required=True, help='API Key')
    parser.add_argument('--secret', required=True, help='API Secret')
    
    args = parser.parse_args()
    
    logger.info("Starting API verification...")
    await test_api_connection(args.key, args.secret)

if __name__ == "__main__":
    asyncio.run(main()) 