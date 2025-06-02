from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase

@dataclass
class MomentumParams:
    """Momentum Strategy Parameters"""
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_ma_period: int = 20
    min_volume_factor: float = 1.5
    trend_ema_period: int = 100
    
class MomentumStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Momentum",
                 params: Optional[MomentumParams] = None,
                 **kwargs):
        """
        Momentum trading strategy combining RSI, MACD, and volume analysis.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or MomentumParams()
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on momentum indicators."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Get trend direction using EMA
        if f'ema_{self.params.trend_ema_period}' not in df.columns:
            df[f'ema_{self.params.trend_ema_period}'] = df['close'].ewm(
                span=self.params.trend_ema_period
            ).mean()
        
        trend_direction = np.where(
            df['close'] > df[f'ema_{self.params.trend_ema_period}'],
            1,  # Uptrend
            -1  # Downtrend
        )
        
        # Volume confirmation
        volume_ma = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = df['volume'] > volume_ma * self.params.min_volume_factor
        
        # Generate buy signals
        buy_conditions = (
            (df['rsi'] < self.params.rsi_oversold) &  # Oversold
            (df['macd'] > df['macd_signal']) &  # MACD bullish crossover
            (trend_direction == 1) &  # Uptrend
            volume_confirmed  # Volume confirmation
        )
        
        # Generate sell signals
        sell_conditions = (
            (df['rsi'] > self.params.rsi_overbought) &  # Overbought
            (df['macd'] < df['macd_signal']) &  # MACD bearish crossover
            (trend_direction == -1) &  # Downtrend
            volume_confirmed  # Volume confirmation
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength
        df['signal_strength'] = abs(df['macd'] - df['macd_signal']) / df['close']
        df.loc[df['signal'] == 0, 'signal_strength'] = 0
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on volatility and signal strength.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Current market volatility
            
        Returns:
            Position size in base currency
        """
        # Get latest signal strength
        signal_strength = self.current_signal_strength if hasattr(self, 'current_signal_strength') else 0.5
        
        # Base position size from risk parameters
        base_size = account_balance * self.risk_per_trade
        
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility)
        
        # Adjust for signal strength
        signal_factor = 0.5 + (signal_strength * 0.5)  # Scale between 0.5 and 1.0
        
        # Calculate final position size
        position_size = base_size * vol_factor * signal_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size 