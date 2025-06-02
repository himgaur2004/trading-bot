from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase

@dataclass
class TrendFollowingParams:
    """Trend Following Strategy Parameters"""
    ema_periods: List[int] = None
    adx_period: int = 14
    adx_threshold: float = 25.0
    volume_ma_period: int = 20
    min_volume_factor: float = 1.5
    atr_period: int = 14
    atr_multiplier: float = 2.0
    
    def __post_init__(self):
        if self.ema_periods is None:
            self.ema_periods = [10, 20, 50, 200]  # Multiple EMAs for trend confirmation
            
class TrendFollowingStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Trend_Following",
                 params: Optional[TrendFollowingParams] = None,
                 **kwargs):
        """
        Trend following strategy using multiple EMAs, ADX, and volume profile.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or TrendFollowingParams()
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on trend indicators."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate EMAs if not present
        for period in self.params.ema_periods:
            if f'ema_{period}' not in df.columns:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Check EMA alignment for trend confirmation
        ema_aligned_bullish = True
        ema_aligned_bearish = True
        
        for i in range(len(self.params.ema_periods)-1):
            fast_ema = df[f'ema_{self.params.ema_periods[i]}']
            slow_ema = df[f'ema_{self.params.ema_periods[i+1]}']
            
            ema_aligned_bullish &= (fast_ema > slow_ema)
            ema_aligned_bearish &= (fast_ema < slow_ema)
        
        # Volume confirmation
        volume_ma = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = df['volume'] > volume_ma * self.params.min_volume_factor
        
        # Strong trend confirmation
        strong_trend = df['adx'] > self.params.adx_threshold
        
        # Generate buy signals
        buy_conditions = (
            ema_aligned_bullish &  # EMAs aligned bullishly
            strong_trend &  # Strong trend
            volume_confirmed  # Volume confirmation
        )
        
        # Generate sell signals
        sell_conditions = (
            ema_aligned_bearish &  # EMAs aligned bearishly
            strong_trend &  # Strong trend
            volume_confirmed  # Volume confirmation
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength based on ADX and volume
        df['signal_strength'] = (
            (df['adx'] / 100) * 0.7 +  # ADX contribution
            (df['volume'] / volume_ma) * 0.3  # Volume contribution
        ).clip(0, 1)
        
        df.loc[df['signal'] == 0, 'signal_strength'] = 0
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on trend strength and volatility.
        
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
        
        # Adjust for trend strength
        trend_factor = 0.5 + (signal_strength * 0.5)
        
        # Calculate final position size
        position_size = base_size * vol_factor * trend_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on ATR and trend direction.
        
        Args:
            current_price: Current asset price
            position_data: Current position information
            
        Returns:
            Dict with updated stop levels
        """
        if not position_data:
            return {}
            
        entry_price = position_data['entry_price']
        side = position_data['side']
        current_stop = position_data.get('stop_loss')
        
        # Get current ATR if available
        atr = self.market_data['atr'].iloc[-1] if hasattr(self, 'market_data') else None
        if atr is None:
            return {}
            
        updates = {}
        
        # Update trailing stop based on ATR
        if side == 'buy':
            new_stop = current_price - (atr * self.params.atr_multiplier)
            if current_stop is None or new_stop > current_stop:
                updates['stop_loss'] = new_stop
                
            # Update take profit to be 2x the stop distance
            stop_distance = current_price - new_stop
            updates['take_profit'] = current_price + (stop_distance * 2)
            
        else:  # sell position
            new_stop = current_price + (atr * self.params.atr_multiplier)
            if current_stop is None or new_stop < current_stop:
                updates['stop_loss'] = new_stop
                
            # Update take profit to be 2x the stop distance
            stop_distance = new_stop - current_price
            updates['take_profit'] = current_price - (stop_distance * 2)
            
        return updates 