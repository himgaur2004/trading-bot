from typing import Dict, List, Optional, Callable
import asyncio
import json
from datetime import datetime
import pandas as pd
import aiohttp
import hmac
import hashlib
import time
from loguru import logger

class MarketDataService:
    def __init__(self, api_key: str, api_secret: str, symbols: List[str]):
        """
        Initialize market data service for CoinDCX.
        
        Args:
            api_key: CoinDCX API key
            api_secret: CoinDCX API secret
            symbols: List of trading pairs
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.callbacks: List[Callable] = []
        self.running = False
        self.last_data: Dict[str, Dict] = {}
        self.base_url = "https://api.coindcx.com"
        self.session = None
        self.connector = None
        
    def _generate_signature(self, body: Dict) -> str:
        """Generate signature for private API calls."""
        serialized_data = json.dumps(body, separators=(',', ':'))
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            serialized_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    async def _init_session(self):
        """Initialize aiohttp session with proper configuration."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.connector = aiohttp.TCPConnector(ssl=False, force_close=True)
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout
            )
        
    async def _make_request(self, endpoint: str, method: str = 'GET', body: Dict = None) -> Dict:
        """
        Make authenticated HTTP request to CoinDCX API.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            body: Request body for POST requests
            
        Returns:
            API response data
        """
        headers = {'Content-Type': 'application/json'}
        
        # Add authentication headers for private endpoints
        if body is not None:
            timestamp = int(time.time() * 1000)
            body['timestamp'] = timestamp
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                json.dumps(body, separators=(',', ':')).encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            headers.update({
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            })
            
        # Initialize session if needed
        await self._init_session()
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}{endpoint}"
                if method == 'GET':
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', retry_delay))
                            logger.warning(f"Rate limit hit, waiting {retry_after} seconds...")
                            await asyncio.sleep(retry_after)
                            continue
                            
                        data = await response.json()
                        if response.status != 200:
                            logger.error(f"API error: {data}")
                            raise Exception(f"API error: {data}")
                            
                        return data
                else:
                    async with self.session.post(url, headers=headers, json=body) as response:
                        if response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', retry_delay))
                            logger.warning(f"Rate limit hit, waiting {retry_after} seconds...")
                            await asyncio.sleep(retry_after)
                            continue
                            
                        data = await response.json()
                        if response.status != 200:
                            logger.error(f"API error: {data}")
                            raise Exception(f"API error: {data}")
                            
                        return data
                        
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Network error after {max_retries} attempts: {e}")
                    raise
                    
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Network error, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error making request: {e}")
                raise
            
    async def get_all_futures_pairs(self) -> List[str]:
        """Fetch all available USDT futures trading pairs from CoinDCX."""
        try:
            # Fetch all markets
            markets = await self._make_request('/exchange/v1/markets_details')
            
            # Filter and collect USDT futures markets
            futures_markets = [
                market for market in markets 
                if ('USDT' in market.get('target_currency', '').upper() or 
                    'USDT' in market.get('market', '').upper()) and
                market.get('coindcx_name', '').endswith('FUTURS')
            ]
            
            # If no futures pairs found, try spot pairs
            if not futures_markets:
                spot_markets = [
                    market for market in markets
                    if 'USDT' in market.get('pair', '').upper()
                ]
                logger.info(f"\nNo futures pairs found, using {len(spot_markets)} USDT spot pairs:")
                for market in spot_markets[:5]:  # Show first 5 pairs
                    logger.info(f"  • {market['pair']}")
                return [market['pair'] for market in spot_markets]
            
            # Log detailed information about found pairs
            logger.info(f"\nFound {len(futures_markets)} USDT futures pairs:")
            for market in futures_markets:
                logger.info(f"\n  • {market['pair']}:")
                logger.info(f"    - Base Currency: {market.get('base_currency', 'N/A')}")
                logger.info(f"    - Target Currency: {market.get('target_currency', 'N/A')}")
                logger.info(f"    - Last Price: {market.get('last_price', 'N/A')}")
                logger.info(f"    - Volume (24h): {market.get('volume', 'N/A')}")
                logger.info(f"    - Change (24h): {market.get('change_24_hour', 'N/A')}%")
                logger.info(f"    - High (24h): {market.get('high', 'N/A')}")
                logger.info(f"    - Low (24h): {market.get('low', 'N/A')}")
            
            return [market['pair'] for market in futures_markets]
            
        except Exception as e:
            logger.error(f"Error fetching futures pairs: {e}")
            raise
            
    async def initialize(self):
        """Initialize market data service."""
        try:
            # Test API connection
            await self._make_request('/exchange/v1/markets_details')
            logger.info("Successfully connected to CoinDCX API")
            
            # If no symbols specified, get all futures pairs
            if not self.symbols:
                self.symbols = await self.get_all_futures_pairs()
                logger.info(f"Scanning {len(self.symbols)} futures pairs: {', '.join(self.symbols)}")
                
        except Exception as e:
            logger.error(f"Failed to initialize CoinDCX connection: {e}")
            raise
            
    async def fetch_market_data(self, symbol: str) -> Dict:
        """
        Fetch market data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary containing market data
        """
        try:
            # Initialize default values
            price_change = 0
            volume_change = 0
            bid_ask_ratio = 1.0
            buy_sell_ratio = 1.0
            spread = 0
            
            # Fetch ticker data
            ticker = await self._make_request('/exchange/v1/markets_details')
            ticker_data = next((t for t in ticker if t['pair'] == symbol), None)
            
            if ticker_data:
                # Calculate price changes
                current_price = float(ticker_data.get('last_price', 0))
                prev_price = float(self.last_data.get(symbol, {}).get('ticker', {}).get('last_price', current_price))
                price_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                
                # Calculate volume changes
                current_volume = float(ticker_data.get('volume', 0))
                prev_volume = float(self.last_data.get(symbol, {}).get('ticker', {}).get('volume', current_volume))
                volume_change = ((current_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0
                
                logger.info(f"\nMarket Update for {symbol}:")
                logger.info(f"  Price Information:")
                logger.info(f"    • Current Price: {current_price}")
                logger.info(f"    • Price Change: {price_change:.2f}%")
                logger.info(f"    • 24h High: {ticker_data.get('high', 'N/A')}")
                logger.info(f"    • 24h Low: {ticker_data.get('low', 'N/A')}")
                logger.info(f"  Volume Information:")
                logger.info(f"    • Current Volume: {current_volume}")
                logger.info(f"    • Volume Change: {volume_change:.2f}%")
                logger.info(f"    • 24h Volume: {ticker_data.get('volume', 'N/A')}")
            else:
                logger.warning(f"  No ticker data available for {symbol}")
            
            # Fetch order book
            orderbook = await self._make_request(f'/market_data/orderbook?pair={symbol}')
            if orderbook:
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                # Calculate spread and volumes only if we have both bids and asks
                if bids and asks:
                    spread = float(asks[0][0]) - float(bids[0][0])
                    bid_volume = sum(float(bid[1]) for bid in bids[:10])
                    ask_volume = sum(float(ask[1]) for ask in asks[:10])
                    bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
                    
                    logger.info(f"  Order Book Analysis:")
                    logger.info(f"    • Spread: {spread:.8f}")
                    logger.info(f"    • Bid/Ask Ratio: {bid_ask_ratio:.2f}")
                    logger.info(f"    • Top 10 Levels:")
                    logger.info(f"      Bids:")
                    for i, bid in enumerate(bids[:5]):
                        logger.info(f"        {i+1}. Price: {bid[0]}, Volume: {bid[1]}")
                    logger.info(f"      Asks:")
                    for i, ask in enumerate(asks[:5]):
                        logger.info(f"        {i+1}. Price: {ask[0]}, Volume: {ask[1]}")
                else:
                    logger.warning(f"  Empty order book for {symbol}")
            else:
                logger.warning(f"  No order book data available for {symbol}")
            
            # Fetch trades
            trades_response = await self._make_request(f'/market_data/trade_history?pair={symbol}')
            trades = trades_response if isinstance(trades_response, list) else []
            
            if trades:
                try:
                    recent_trades = trades[:10] if len(trades) > 0 else []
                    buy_volume = sum(float(trade.get('quantity', 0)) for trade in recent_trades if trade.get('type') == 'buy')
                    sell_volume = sum(float(trade.get('quantity', 0)) for trade in recent_trades if trade.get('type') == 'sell')
                    buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 1.0
                    
                    logger.info(f"  Recent Trades Analysis:")
                    logger.info(f"    • Buy Volume: {buy_volume:.8f}")
                    logger.info(f"    • Sell Volume: {sell_volume:.8f}")
                    logger.info(f"    • Buy/Sell Ratio: {buy_sell_ratio:.2f}")
                    logger.info(f"    • Last 5 Trades:")
                    for i, trade in enumerate(recent_trades[:5]):
                        logger.info(f"      {i+1}. Price: {trade.get('price', 'N/A')}, "
                                  f"Size: {trade.get('quantity', 'N/A')}, "
                                  f"Side: {trade.get('type', 'N/A')}, "
                                  f"Time: {trade.get('timestamp', 'N/A')}")
                except Exception as e:
                    logger.error(f"Error processing trades data for {symbol}: {e}")
                    trades = []
            else:
                logger.warning(f"  No recent trades for {symbol}")
            
            # Format data
            market_data = {
                'ticker': ticker_data,
                'orderbook': orderbook,
                'trades': trades,
                'timestamp': datetime.now(),
                'analysis': {
                    'price_change': price_change,
                    'volume_change': volume_change,
                    'bid_ask_ratio': bid_ask_ratio,
                    'buy_sell_ratio': buy_sell_ratio,
                    'spread': spread
                }
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            raise
            
    async def start(self):
        """Start market data service."""
        self.running = True
        while self.running:
            for symbol in self.symbols:
                try:
                    market_data = await self.fetch_market_data(symbol)
                    self.last_data[symbol] = market_data
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        await callback(symbol, market_data)
                        
                except Exception as e:
                    logger.error(f"Error in market data loop: {e}")
                    
            await asyncio.sleep(1)  # Rate limit compliance
            
    async def stop(self):
        """Stop market data service."""
        self.running = False
        if self.session and not self.session.closed:
            await self.session.close()
        if self.connector and not self.connector.closed:
            await self.connector.close()
            
    def add_callback(self, callback: Callable):
        """Add callback for market data updates."""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback: Callable):
        """Remove callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def get_last_data(self, symbol: str) -> Optional[Dict]:
        """
        Get last known market data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Last known market data or None
        """
        return self.last_data.get(symbol)
        
    async def get_historical_data(self,
                                symbol: str,
                                timeframe: str = '1h',
                                limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval
            limit: Number of candles
            
        Returns:
            DataFrame with historical data
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            ).set_index('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
            
    async def get_order_book_snapshot(self,
                                    symbol: str,
                                    depth: int = 20) -> Dict:
        """
        Get current order book snapshot.
        
        Args:
            symbol: Trading pair symbol
            depth: Order book depth
            
        Returns:
            Order book data
        """
        try:
            orderbook = await self.exchange.fetch_order_book(
                symbol,
                limit=depth
            )
            return orderbook
            
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return {'bids': [], 'asks': []}
            
    async def __aenter__(self):
        """Async context manager enter."""
        await self._init_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop() 