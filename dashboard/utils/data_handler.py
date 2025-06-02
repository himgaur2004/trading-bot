import os
import hmac
import hashlib
import json
import time
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

class CoinDCXDataHandler:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coindcx.com"
        
    async def generate_signature(self, body: dict) -> str:
        """Generate signature for API request."""
        sorted_body = dict(sorted(body.items()))
        message = json.dumps(sorted_body, separators=(',', ':'))
        return hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def make_request(self, endpoint: str, body: dict = None, method: str = "POST") -> dict:
        """Make an authenticated request to CoinDCX API."""
        if body is None:
            body = {}
        
        if 'timestamp' not in body:
            body['timestamp'] = str(int(time.time() * 1000))
        
        signature = await self.generate_signature(body)
        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-SIGNATURE': signature
        }
        
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, json=body, ssl=False) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    response_text = await response.text()
                    raise Exception(f"API request failed: {response_text}")

    async def get_market_data(self) -> pd.DataFrame:
        """Get current market data for all trading pairs."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/exchange/v1/markets_details", ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    return pd.DataFrame(data)
                else:
                    raise Exception("Failed to fetch market data")

    async def get_balances(self) -> List[Dict]:
        """Get user account balances."""
        return await self.make_request("/exchange/v1/users/balances")

    async def get_active_orders(self) -> List[Dict]:
        """Get all active orders."""
        return await self.make_request("/exchange/v1/orders/active_orders")

    async def get_order_history(self, pair: str = None) -> List[Dict]:
        """Get order history."""
        body = {'timestamp': str(int(time.time() * 1000))}
        if pair:
            body['pair'] = pair
        return await self.make_request("/exchange/v1/orders/trade_history", body)

    async def get_candles(self, pair: str, interval: str = "1d", limit: int = 100) -> pd.DataFrame:
        """Get candlestick data for a trading pair."""
        url = f"{self.base_url}/market_data/candles"
        params = {
            'pair': pair,
            'interval': interval,
            'limit': limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
                else:
                    raise Exception("Failed to fetch candle data")

    def calculate_portfolio_value(self, balances: List[Dict], market_data: pd.DataFrame) -> float:
        """Calculate total portfolio value in USDT."""
        total_value = 0.0
        
        for balance in balances:
            if float(balance['balance']) > 0:
                currency = balance['currency']
                amount = float(balance['balance'])
                
                if currency == 'USDT':
                    total_value += amount
                else:
                    # Find the USDT pair for this currency
                    pair = f"{currency}USDT"
                    price_data = market_data[market_data['pair'] == pair]
                    
                    if not price_data.empty:
                        price = float(price_data.iloc[0]['last_price'])
                        total_value += amount * price
        
        return total_value

    def calculate_24h_change(self, balances: List[Dict], market_data: pd.DataFrame) -> tuple:
        """Calculate 24h portfolio change."""
        current_value = self.calculate_portfolio_value(balances, market_data)
        
        # Calculate 24h change using market data
        total_change = 0.0
        total_change_percent = 0.0
        
        for balance in balances:
            if float(balance['balance']) > 0:
                currency = balance['currency']
                amount = float(balance['balance'])
                
                if currency != 'USDT':
                    pair = f"{currency}USDT"
                    price_data = market_data[market_data['pair'] == pair]
                    
                    if not price_data.empty:
                        current_price = float(price_data.iloc[0]['last_price'])
                        price_change_24h = float(price_data.iloc[0]['change_24_hour'])
                        
                        value_change = amount * (current_price - (current_price - price_change_24h))
                        total_change += value_change
        
        if current_value > 0:
            total_change_percent = (total_change / current_value) * 100
            
        return total_change, total_change_percent

    async def get_portfolio_summary(self) -> Dict:
        """Get complete portfolio summary."""
        try:
            # Fetch required data
            balances = await self.get_balances()
            market_data = await self.get_market_data()
            active_orders = await self.get_active_orders()
            
            # Calculate portfolio metrics
            portfolio_value = self.calculate_portfolio_value(balances, market_data)
            change_24h, change_percent_24h = self.calculate_24h_change(balances, market_data)
            
            return {
                'portfolio_value': portfolio_value,
                'change_24h': change_24h,
                'change_percent_24h': change_percent_24h,
                'active_orders_count': len(active_orders),
                'balances': balances,
                'market_data': market_data.to_dict('records'),
                'active_orders': active_orders
            }
        except Exception as e:
            raise Exception(f"Failed to get portfolio summary: {str(e)}") 