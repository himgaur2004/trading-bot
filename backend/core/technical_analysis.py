import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class TAParameters:
    """Technical Analysis Parameters"""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_periods: List[int] = None
    vwap_period: int = 14
    
    def __post_init__(self):
        if self.ema_periods is None:
            self.ema_periods = [9, 21, 50, 200]

class TechnicalAnalysis:
    def __init__(self, params: Optional[TAParameters] = None):
        """
        Initialize Technical Analysis engine.
        
        Args:
            params: Technical analysis parameters
        """
        self.params = params or TAParameters()
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Basic price indicators
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['hlc3'] = df['typical_price']
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Momentum indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.params.rsi_period)
        
        macd, signal, hist = talib.MACD(
            df['close'],
            fastperiod=self.params.macd_fast,
            slowperiod=self.params.macd_slow,
            signalperiod=self.params.macd_signal
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Trend indicators
        for period in self.params.ema_periods:
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            
        # Volatility indicators
        upper, middle, lower = talib.BBANDS(
            df['close'],
            timeperiod=self.params.bb_period,
            nbdevup=self.params.bb_std,
            nbdevdn=self.params.bb_std
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        
        # Volume indicators
        df['vwap'] = self._calculate_vwap(df)
        df['volume_ema'] = talib.EMA(df['volume'], timeperiod=self.params.vwap_period)
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=self.params.vwap_period)
        
        # Additional indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP indicator."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=self.params.vwap_period).sum() / \
               df['volume'].rolling(window=self.params.vwap_period).sum()
        return vwap
    
    def detect_divergence(self, 
                         df: pd.DataFrame, 
                         price_col: str = 'close',
                         indicator_col: str = 'rsi',
                         window: int = 20) -> pd.Series:
        """
        Detect regular and hidden divergences.
        
        Args:
            df: DataFrame with price and indicator data
            price_col: Column name for price data
            indicator_col: Column name for indicator data
            window: Lookback window for divergence detection
            
        Returns:
            Series with divergence signals (1: bullish, -1: bearish, 0: none)
        """
        signals = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i+1]
            
            # Find local extrema
            price_min_idx = window_data[price_col].idxmin()
            price_max_idx = window_data[price_col].idxmax()
            ind_min_idx = window_data[indicator_col].idxmin()
            ind_max_idx = window_data[indicator_col].idxmax()
            
            # Regular bullish divergence
            if (price_min_idx == window_data.index[-1] and 
                ind_min_idx != window_data.index[-1] and
                window_data[indicator_col].iloc[-1] > window_data[indicator_col][ind_min_idx]):
                signals.iloc[i] = 1
                
            # Regular bearish divergence
            elif (price_max_idx == window_data.index[-1] and
                  ind_max_idx != window_data.index[-1] and
                  window_data[indicator_col].iloc[-1] < window_data[indicator_col][ind_max_idx]):
                signals.iloc[i] = -1
                
        return signals
    
    def get_market_condition(self, df: pd.DataFrame) -> str:
        """
        Determine market condition (trending/ranging/volatile).
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Market condition string
        """
        latest = df.iloc[-1]
        
        # Check for strong trend
        adx_strong = latest['adx'] > 25
        ema_aligned = (latest['ema_9'] > latest['ema_21'] > latest['ema_50']) or \
                     (latest['ema_9'] < latest['ema_21'] < latest['ema_50'])
        
        # Check for high volatility
        high_bb_width = latest['bb_width'] > df['bb_width'].mean() + df['bb_width'].std()
        high_atr = latest['atr'] > df['atr'].mean() + df['atr'].std()
        
        if high_bb_width and high_atr:
            return 'volatile'
        elif adx_strong and ema_aligned:
            return 'trending'
        else:
            return 'ranging'
    
    def calculate_support_resistance(self, 
                                   df: pd.DataFrame,
                                   window: int = 20,
                                   num_points: int = 3) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            window: Lookback window
            num_points: Number of S/R points to return
            
        Returns:
            Dict with support and resistance levels
        """
        levels = {'support': [], 'resistance': []}
        
        # Find local minima and maxima
        for i in range(window, len(df)-window):
            if all(df['low'].iloc[i] <= df['low'].iloc[i-window:i]) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+1:i+window]):
                levels['support'].append(df['low'].iloc[i])
                
            if all(df['high'].iloc[i] >= df['high'].iloc[i-window:i]) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+1:i+window]):
                levels['resistance'].append(df['high'].iloc[i])
                
        # Cluster nearby levels
        levels['support'] = self._cluster_levels(levels['support'])[:num_points]
        levels['resistance'] = self._cluster_levels(levels['resistance'])[:num_points]
        
        return levels
    
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.002) -> List[float]:
        """Cluster nearby price levels."""
        if not levels:
            return []
            
        levels = sorted(levels)
        clusters = [[levels[0]]]
        
        for level in levels[1:]:
            if abs(level - clusters[-1][0]) / clusters[-1][0] <= tolerance:
                clusters[-1].append(level)
            else:
                clusters.append([level])
                
        return [sum(cluster)/len(cluster) for cluster in clusters] 