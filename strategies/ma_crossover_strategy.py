from typing import Dict, Optional
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase
from dataclasses import dataclass

@dataclass
class MACrossoverParams:
    """Moving Average Crossover Strategy Parameters"""
    fast_ma: int = 9
    slow_ma: int = 21
    signal_ma: int = 5
    min_trend_strength: float = 25.0  # Minimum ADX value for trend
    volume_factor: float = 1.5  # Minimum volume increase for confirmation
    
class MACrossoverStrategy(StrategyBase):
    def __init__(self,
                 name: str = "MA_Crossover",
                 params: Optional[MACrossoverParams] = None,
                 **kwargs):
        """
        Moving Average Crossover Strategy.
        
        Args:
            name: Strategy name
            params: Strategy-specific parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or MACrossoverParams()
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MA crossover.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with signals
        """
        df = data.copy()
        
        # Calculate moving averages if not present
        if f'ema_{self.params.fast_ma}' not in df.columns:
            df[f'ema_{self.params.fast_ma}'] = df['close'].ewm(span=self.params.fast_ma).mean()
        if f'ema_{self.params.slow_ma}' not in df.columns:
            df[f'ema_{self.params.slow_ma}'] = df['close'].ewm(span=self.params.slow_ma).mean()
            
        fast_ma = df[f'ema_{self.params.fast_ma}']
        slow_ma = df[f'ema_{self.params.slow_ma}']
        
        # Calculate crossover signals
        df['ma_diff'] = fast_ma - slow_ma
        df['ma_diff_prev'] = df['ma_diff'].shift(1)
        
        # Initialize signals
        df['signal'] = 0
        
        # Detect crossovers
        df.loc[(df['ma_diff'] > 0) & (df['ma_diff_prev'] <= 0), 'signal'] = 1  # Buy signal
        df.loc[(df['ma_diff'] < 0) & (df['ma_diff_prev'] >= 0), 'signal'] = -1  # Sell signal
        
        # Apply filters
        if 'adx' in df.columns:
            df.loc[df['adx'] < self.params.min_trend_strength, 'signal'] = 0
            
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(window=20).mean()
            df.loc[df['volume'] < volume_ma * self.params.volume_factor, 'signal'] = 0
            
        # Calculate signal strength (0 to 1)
        df['signal_strength'] = abs(df['ma_diff']) / df['close']
        
        # Apply signal smoothing
        if self.params.signal_ma > 1:
            df['signal_strength'] = df['signal_strength'].rolling(
                window=self.params.signal_ma
            ).mean()
            
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
        max_size = account_balance * self.max_position_size / current_price
        position_size = min(position_size, max_size)
        
        return position_size
    
    def should_update_stops(self, 
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Check if stop loss/take profit should be updated.
        
        Args:
            current_price: Current asset price
            position_data: Current position information
            
        Returns:
            Dict with updated stop levels or empty dict if no update needed
        """
        if not position_data:
            return {}
            
        entry_price = position_data['entry_price']
        side = position_data['side']
        current_stop = position_data.get('stop_loss')
        current_tp = position_data.get('take_profit')
        
        updates = {}
        
        # Trail stop loss
        if side == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct > 0.02:  # Start trailing after 2% profit
                new_stop = current_price * (1 - self.stop_loss * 0.8)  # Tighter stop
                if not current_stop or new_stop > current_stop:
                    updates['stop_loss'] = new_stop
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct > 0.02:
                new_stop = current_price * (1 + self.stop_loss * 0.8)
                if not current_stop or new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        # Update take profit based on volatility
        if hasattr(self, 'current_volatility'):
            vol_factor = 1 + (self.current_volatility * 2)
            if side == 'buy':
                new_tp = entry_price * (1 + self.take_profit * vol_factor)
                if not current_tp or new_tp > current_tp:
                    updates['take_profit'] = new_tp
            else:
                new_tp = entry_price * (1 - self.take_profit * vol_factor)
                if not current_tp or new_tp < current_tp:
                    updates['take_profit'] = new_tp
                    
        return updates 