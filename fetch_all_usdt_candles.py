#!/usr/bin/env python3
"""
Script to fetch candlestick data for USDT pairs from Binance Futures public API
"""
import pandas as pd
from utils.data_handler import BinanceFuturesDataHandler

def fetch_binance_usdt_candles(symbol: str, interval: str = '15m', limit: int = 100):
    handler = BinanceFuturesDataHandler()
    df = handler.get_candles(symbol, interval, limit)
    print(df.tail())

if __name__ == "__main__":
    # Example: Fetch BTCUSDT 15m candles
    fetch_binance_usdt_candles('BTCUSDT', '15m', 100) 