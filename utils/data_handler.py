import requests
import pandas as pd
from typing import Optional

class BinanceFuturesDataHandler:
    """Handler for Binance Futures public data"""
    BASE_URL = "https://fapi.binance.com"

    def get_candles(self, symbol: str, interval: str = '15m', limit: int = 100) -> pd.DataFrame:
        """Fetch candlestick data from Binance Futures public API"""
        endpoint = f"/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # Parse kline data into DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df.set_index('close_time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']] 