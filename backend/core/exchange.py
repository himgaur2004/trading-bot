import os
import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Union
from datetime import datetime
import requests
import pandas as pd
from dataclasses import dataclass

@dataclass
class ExchangeConfig:
    """Exchange Configuration"""
    api_key: str
    api_secret: str
    base_url: str = "https://api.coindcx.com"
    use_testnet: bool = True
    recv_window: int = 5000
    
class CoinDCXExchange:
    def __init__(self, config: Optional[ExchangeConfig] = None):
        """
        Initialize CoinDCX exchange interface.
        
        Args:
            config: Exchange configuration
        """
        self.config = config or ExchangeConfig(
            api_key=os.getenv('COINDCX_API_KEY', ''),
            api_secret=os.getenv('COINDCX_API_SECRET', '')
        )
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.config.api_key
        })
        
    def _generate_signature(self, data: Dict) -> str:
        """Generate HMAC signature for API authentication."""
        data_str = json.dumps(data, separators=(',', ':'))
        return hmac.new(
            self.config.api_secret.encode('utf-8'),
            data_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    def _make_request(self, 
                     method: str,
                     endpoint: str,
                     params: Optional[Dict] = None,
                     signed: bool = False) -> Dict:
        """
        Make HTTP request to exchange API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            signed: Whether request needs signature
            
        Returns:
            API response
        """
        url = f"{self.config.base_url}{endpoint}"
        
        if signed:
            if params is None:
                params = {}
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = self.config.recv_window
            signature = self._generate_signature(params)
            params['signature'] = signature
            
        try:
            if method == 'GET':
                response = self.session.get(url, params=params)
            elif method == 'POST':
                response = self.session.post(url, json=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
            
    def get_exchange_info(self) -> Dict:
        """Get exchange information and trading rules."""
        return self._make_request('GET', '/exchange/v1/markets')
        
    def get_account_balance(self) -> Dict:
        """Get account balances."""
        return self._make_request('POST', '/exchange/v1/users/balances', {}, signed=True)
        
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get market order book.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of price levels
        """
        params = {'symbol': symbol, 'limit': limit}
        return self._make_request('GET', '/exchange/v1/markets/orderbook', params)
        
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades
        """
        params = {'symbol': symbol, 'limit': limit}
        return self._make_request('GET', '/exchange/v1/markets/trades', params)
        
    def get_klines(self,
                   symbol: str,
                   interval: str,
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None,
                   limit: int = 500) -> pd.DataFrame:
        """
        Get candlestick data.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_time: Start timestamp
            end_time: End timestamp
            limit: Number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        data = self._make_request('GET', '/exchange/v1/markets/candles', params)
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df
        
    def create_order(self,
                    symbol: str,
                    side: str,
                    order_type: str,
                    quantity: float,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: str = 'GTC') -> Dict:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY/SELL)
            order_type: Order type (LIMIT/MARKET/STOP_LOSS/TAKE_PROFIT)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            stop_price: Stop price (required for STOP_LOSS/TAKE_PROFIT orders)
            time_in_force: Time in force (GTC/IOC/FOK)
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity,
            'timestamp': int(time.time() * 1000)
        }
        
        if order_type == 'LIMIT':
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            params['price'] = price
            params['timeInForce'] = time_in_force
            
        elif order_type in ['STOP_LOSS', 'TAKE_PROFIT']:
            if stop_price is None:
                raise ValueError("Stop price is required for STOP_LOSS/TAKE_PROFIT orders")
            params['stopPrice'] = stop_price
            
        return self._make_request('POST', '/exchange/v1/orders/create', params, signed=True)
        
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        Cancel an existing order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
        """
        params = {
            'symbol': symbol,
            'orderId': order_id,
            'timestamp': int(time.time() * 1000)
        }
        return self._make_request('POST', '/exchange/v1/orders/cancel', params, signed=True)
        
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open orders.
        
        Args:
            symbol: Optional symbol filter
        """
        params = {'timestamp': int(time.time() * 1000)}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('POST', '/exchange/v1/orders/active', params, signed=True)
        
    def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """
        Get order status.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to query
        """
        params = {
            'symbol': symbol,
            'orderId': order_id,
            'timestamp': int(time.time() * 1000)
        }
        return self._make_request('POST', '/exchange/v1/orders/status', params, signed=True)
        
    def get_trade_history(self, 
                         symbol: Optional[str] = None,
                         limit: int = 500) -> List[Dict]:
        """
        Get trade history.
        
        Args:
            symbol: Optional symbol filter
            limit: Number of trades
        """
        params = {
            'timestamp': int(time.time() * 1000),
            'limit': limit
        }
        if symbol:
            params['symbol'] = symbol
        return self._make_request('POST', '/exchange/v1/orders/trade_history', params, signed=True)
        
    def get_position(self, symbol: str) -> Dict:
        """
        Get current position for futures trading.
        
        Args:
            symbol: Trading pair symbol
        """
        params = {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000)
        }
        return self._make_request('POST', '/exchange/v1/positions', params, signed=True)
        
    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Set leverage for futures trading.
        
        Args:
            symbol: Trading pair symbol
            leverage: Leverage value
        """
        params = {
            'symbol': symbol,
            'leverage': leverage,
            'timestamp': int(time.time() * 1000)
        }
        return self._make_request('POST', '/exchange/v1/positions/leverage', params, signed=True)
        
    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Get current funding rate for futures.
        
        Args:
            symbol: Trading pair symbol
        """
        params = {'symbol': symbol}
        return self._make_request('GET', '/exchange/v1/markets/funding_rate', params) 