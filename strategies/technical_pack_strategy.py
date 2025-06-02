from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase

@dataclass
class TechnicalPackParams:
    """Technical Pack Strategy Parameters"""
    # MACD Parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # ADX Parameters
    adx_period: int = 14
    adx_threshold: float = 25.0
    
    # Triple MA Parameters
    fast_ma: int = 5
    medium_ma: int = 21
    slow_ma: int = 50
    
    # Supertrend Parameters
    atr_period: int = 10
    atr_multiplier: float = 3.0
    
    # Volume Parameters
    volume_ma_period: int = 20
    min_volume_factor: float = 1.5
    
    # Signal Parameters
    min_confirmation_signals: int = 3  # Minimum signals needed for trade
    
class TechnicalPackStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Technical_Pack",
                 params: Optional[TechnicalPackParams] = None,
                 **kwargs):
        """
        Technical Pack strategy combining multiple technical indicators.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or TechnicalPackParams()
        
    def _calculate_supertrend(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Supertrend indicator."""
        df = data.copy()
        
        # Calculate ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.params.atr_period).mean()
        
        # Calculate Supertrend
        upperband = ((df['high'] + df['low']) / 2) + (self.params.atr_multiplier * atr)
        lowerband = ((df['high'] + df['low']) / 2) - (self.params.atr_multiplier * atr)
        
        supertrend = pd.Series(index=df.index)
        direction = pd.Series(index=df.index)
        
        for i in range(1, len(df)):
            if df['close'][i] > upperband[i-1]:
                supertrend[i] = lowerband[i]
                direction[i] = 1
            elif df['close'][i] < lowerband[i-1]:
                supertrend[i] = upperband[i]
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1]
                
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        
        return df
        
    def _calculate_triple_ma(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Triple Moving Average signals."""
        df = data.copy()
        
        # Calculate MAs
        df['fast_ma'] = df['close'].ewm(span=self.params.fast_ma).mean()
        df['medium_ma'] = df['close'].ewm(span=self.params.medium_ma).mean()
        df['slow_ma'] = df['close'].ewm(span=self.params.slow_ma).mean()
        
        # MA Alignment
        df['triple_ma_bullish'] = (
            (df['fast_ma'] > df['medium_ma']) &
            (df['medium_ma'] > df['slow_ma'])
        )
        
        df['triple_ma_bearish'] = (
            (df['fast_ma'] < df['medium_ma']) &
            (df['medium_ma'] < df['slow_ma'])
        )
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on multiple technical indicators."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate Supertrend
        df = self._calculate_supertrend(df)
        
        # Calculate Triple MA
        df = self._calculate_triple_ma(df)
        
        # Volume confirmation
        volume_ma = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = df['volume'] > volume_ma * self.params.min_volume_factor
        
        # Count bullish signals
        bullish_signals = (
            (df['macd'] > df['macd_signal']).astype(int) +  # MACD bullish
            (df['adx'] > self.params.adx_threshold).astype(int) +  # Strong trend
            (df['supertrend_direction'] == 1).astype(int) +  # Supertrend bullish
            df['triple_ma_bullish'].astype(int)  # Triple MA bullish
        )
        
        # Count bearish signals
        bearish_signals = (
            (df['macd'] < df['macd_signal']).astype(int) +  # MACD bearish
            (df['adx'] > self.params.adx_threshold).astype(int) +  # Strong trend
            (df['supertrend_direction'] == -1).astype(int) +  # Supertrend bearish
            df['triple_ma_bearish'].astype(int)  # Triple MA bearish
        )
        
        # Generate buy signals
        buy_conditions = (
            (bullish_signals >= self.params.min_confirmation_signals) &
            volume_confirmed
        )
        
        # Generate sell signals
        sell_conditions = (
            (bearish_signals >= self.params.min_confirmation_signals) &
            volume_confirmed
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength based on number of confirming signals
        df.loc[buy_conditions, 'signal_strength'] = bullish_signals / 4
        df.loc[sell_conditions, 'signal_strength'] = bearish_signals / 4
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on signal strength and trend strength.
        
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
        signal_factor = 0.5 + (signal_strength * 0.5)
        
        # Calculate final position size
        position_size = base_size * vol_factor * signal_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on Supertrend and ATR.
        
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
        
        # Get current market data
        if not hasattr(self, 'market_data'):
            return {}
            
        supertrend = self.market_data['supertrend'].iloc[-1]
        atr = self.market_data['atr'].iloc[-1]
        
        updates = {}
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = min(
                    entry_price - (atr * self.params.atr_multiplier),
                    supertrend
                )
                updates['take_profit'] = entry_price + (atr * self.params.atr_multiplier * 2)
            else:
                # Trail stop using Supertrend
                new_stop = supertrend
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = max(
                    entry_price + (atr * self.params.atr_multiplier),
                    supertrend
                )
                updates['take_profit'] = entry_price - (atr * self.params.atr_multiplier * 2)
            else:
                # Trail stop using Supertrend
                new_stop = supertrend
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates 