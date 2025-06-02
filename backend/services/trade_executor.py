from typing import Dict, Optional
import asyncio
import json
import hmac
import hashlib
import time
from datetime import datetime
import aiohttp
from loguru import logger
from ..database.database import DatabaseHandler

class TradeExecutor:
    def __init__(self,
                 api_key: str,
                 api_secret: str):
        """
        Initialize trade executor for CoinDCX.
        
        Args:
            api_key: CoinDCX API key
            api_secret: CoinDCX API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.db = DatabaseHandler()
        self.base_url = "https://api.coindcx.com"
        
    def _generate_signature(self, body: Dict) -> str:
        """Generate signature for private API calls."""
        serialized_data = json.dumps(body, separators=(',', ':'))
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            serialized_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    async def _make_request(self, endpoint: str, method: str = 'GET', body: Dict = None) -> Dict:
        """Make HTTP request to CoinDCX API."""
        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.api_key,
        }
        
        if body:
            timestamp = int(time.time() * 1000)
            body['timestamp'] = timestamp
            signature = self._generate_signature(body)
            headers['X-AUTH-SIGNATURE'] = signature
            
        # Configure timeout and SSL context
        timeout = aiohttp.ClientTimeout(total=30)
        conn = aiohttp.TCPConnector(ssl=False)  # Disable SSL verification for now
        
        try:
            async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
                url = f"{self.base_url}{endpoint}"
                if method == 'GET':
                    async with session.get(url, headers=headers) as response:
                        return await response.json()
                else:
                    async with session.post(url, headers=headers, json=body) as response:
                        return await response.json()
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
        finally:
            await asyncio.sleep(0.1)  # Small delay to prevent connection issues
                    
    async def place_order(self,
                         symbol: str,
                         side: str,
                         order_type: str,
                         quantity: float,
                         price: Optional[float] = None) -> Dict:
        """
        Place an order on CoinDCX.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            order_type: 'limit' or 'market'
            quantity: Order quantity
            price: Order price (required for limit orders)
            
        Returns:
            Order response from exchange
        """
        try:
            body = {
                'pair': symbol,
                'side': side.lower(),
                'order_type': order_type.lower(),
                'quantity': quantity,
            }
            
            if order_type.lower() == 'limit' and price is not None:
                body['price'] = price
                
            response = await self._make_request(
                '/exchange/v1/orders/create',
                method='POST',
                body=body
            )
            
            # Store order in database
            await self.db.store_trade({
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'price': price if price else 0,
                'status': 'open',
                'order_id': response.get('id', ''),
                'timestamp': datetime.now()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation response
        """
        try:
            body = {'id': order_id}
            response = await self._make_request(
                '/exchange/v1/orders/cancel',
                method='POST',
                body=body
            )
            
            # Update order status in database
            await self.db.update_trade_status(order_id, 'cancelled')
            
            return response
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            raise
            
    async def get_order_status(self, order_id: str) -> Dict:
        """
        Get order status.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status
        """
        try:
            body = {'id': order_id}
            return await self._make_request(
                '/exchange/v1/orders/status',
                method='POST',
                body=body
            )
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            raise
            
    async def get_account_balance(self) -> Dict:
        """
        Get account balance.
        
        Returns:
            Account balance information
        """
        try:
            return await self._make_request(
                '/exchange/v1/users/balances',
                method='POST',
                body={}
            )
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            raise 