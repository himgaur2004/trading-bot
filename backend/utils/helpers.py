from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_volatility(prices: pd.Series, window: int = 20) -> float:
    """
    Calculate price volatility.
    
    Args:
        prices: Price series
        window: Rolling window size
        
    Returns:
        Volatility value
    """
    returns = prices.pct_change()
    return returns.std()
    
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        df: OHLC DataFrame
        period: ATR period
        
    Returns:
        ATR series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr
    
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI series
    """
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
    
def calculate_macd(prices: pd.Series,
                  fast_period: int = 12,
                  slow_period: int = 26,
                  signal_period: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD indicator.
    
    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Dictionary with MACD line and signal line
    """
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': macd_line - signal_line
    }
    
def calculate_bollinger_bands(prices: pd.Series,
                            period: int = 20,
                            std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price series
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Dictionary with upper, middle, and lower bands
    """
    middle_band = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }
    
def calculate_volume_profile(df: pd.DataFrame,
                           price_levels: int = 100) -> Dict[float, float]:
    """
    Calculate volume profile.
    
    Args:
        df: OHLCV DataFrame
        price_levels: Number of price levels
        
    Returns:
        Dictionary of price levels and volumes
    """
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_step = (price_max - price_min) / price_levels
    
    volume_profile = {}
    
    for i in range(price_levels):
        price_level = price_min + (i * price_step)
        mask = (df['low'] <= price_level) & (df['high'] >= price_level)
        volume = df.loc[mask, 'volume'].sum()
        volume_profile[price_level] = volume
        
    return volume_profile
    
def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        VWAP series
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap
    
def detect_divergence(price: pd.Series,
                     indicator: pd.Series,
                     window: int = 20) -> List[Dict]:
    """
    Detect price and indicator divergences.
    
    Args:
        price: Price series
        indicator: Indicator series
        window: Look-back window
        
    Returns:
        List of divergence events
    """
    divergences = []
    
    for i in range(window, len(price)):
        price_window = price[i-window:i]
        indicator_window = indicator[i-window:i]
        
        price_trend = 1 if price_window.iloc[-1] > price_window.iloc[0] else -1
        indicator_trend = 1 if indicator_window.iloc[-1] > indicator_window.iloc[0] else -1
        
        if price_trend != indicator_trend:
            divergences.append({
                'timestamp': price.index[i],
                'price': price.iloc[i],
                'indicator': indicator.iloc[i],
                'type': 'bullish' if indicator_trend > price_trend else 'bearish'
            })
            
    return divergences
    
def calculate_support_resistance(df: pd.DataFrame,
                               window: int = 20,
                               threshold: float = 0.02) -> Dict[str, List[float]]:
    """
    Calculate support and resistance levels.
    
    Args:
        df: OHLC DataFrame
        window: Look-back window
        threshold: Price level threshold
        
    Returns:
        Dictionary with support and resistance levels
    """
    levels = {'support': [], 'resistance': []}
    
    for i in range(window, len(df)):
        high_window = df['high'][i-window:i]
        low_window = df['low'][i-window:i]
        
        # Find local maxima
        if df['high'].iloc[i-1] == high_window.max():
            resistance = df['high'].iloc[i-1]
            if not any(abs(r - resistance) / r < threshold for r in levels['resistance']):
                levels['resistance'].append(resistance)
                
        # Find local minima
        if df['low'].iloc[i-1] == low_window.min():
            support = df['low'].iloc[i-1]
            if not any(abs(s - support) / s < threshold for s in levels['support']):
                levels['support'].append(support)
                
    return levels 