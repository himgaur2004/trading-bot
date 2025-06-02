from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase

@dataclass
class BreakoutParams:
    """Breakout Strategy Parameters"""
    support_resistance_period: int = 20
    volume_ma_period: int = 20
    min_volume_factor: float = 2.0  # Higher volume requirement for breakouts
    atr_period: int = 14
    atr_multiplier: float = 1.5
    rsi_period: int = 14
    rsi_threshold: float = 50  # Momentum confirmation
    min_consolidation_periods: int = 10
    max_consolidation_range: float = 0.03  # 3% range for consolidation
    
class BreakoutStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Breakout",
                 params: Optional[BreakoutParams] = None,
                 **kwargs):
        """
        Breakout strategy using support/resistance levels and volume confirmation.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or BreakoutParams()
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        
    def _find_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels."""
        highs = data['high'].values
        lows = data['low'].values
        
        support = []
        resistance = []
        
        # Look for swing highs and lows
        for i in range(2, len(data)-2):
            # Support
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support.append(lows[i])
                
            # Resistance
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance.append(highs[i])
                
        # Cluster nearby levels
        support = self._cluster_levels(support)
        resistance = self._cluster_levels(resistance)
        
        return support, resistance
        
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.005) -> List[float]:
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
        
    def _is_consolidating(self, data: pd.DataFrame) -> bool:
        """Check if price is consolidating."""
        recent_data = data.tail(self.params.min_consolidation_periods)
        price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['low'].min()
        return price_range <= self.params.max_consolidation_range
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on breakout conditions."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Find support and resistance levels
        support, resistance = self._find_support_resistance(
            df.tail(self.params.support_resistance_period)
        )
        self.support_levels = support
        self.resistance_levels = resistance
        
        # Get current price and indicators
        current_price = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        # Volume confirmation
        volume_ma = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = df['volume'] > volume_ma * self.params.min_volume_factor
        
        # Volatility (ATR)
        atr = df['atr'].iloc[-1]
        
        # Check for consolidation
        is_consolidating = self._is_consolidating(df)
        
        # RSI for momentum confirmation
        rsi_confirmed = df['rsi'].iloc[-1] > self.params.rsi_threshold
        
        # Generate signals
        for resistance_level in resistance:
            # Bullish breakout
            if (current_high > resistance_level and
                is_consolidating and
                volume_confirmed and
                rsi_confirmed):
                
                df.iloc[-1, df.columns.get_loc('signal')] = 1
                # Signal strength based on breakout size and volume
                breakout_size = (current_high - resistance_level) / atr
                volume_factor = df['volume'].iloc[-1] / volume_ma.iloc[-1]
                df.iloc[-1, df.columns.get_loc('signal_strength')] = min(
                    1.0,
                    (breakout_size * 0.7 + volume_factor * 0.3)
                )
                break
                
        for support_level in support:
            # Bearish breakout
            if (current_low < support_level and
                is_consolidating and
                volume_confirmed and
                not rsi_confirmed):  # RSI below threshold for bearish
                
                df.iloc[-1, df.columns.get_loc('signal')] = -1
                # Signal strength based on breakout size and volume
                breakout_size = (support_level - current_low) / atr
                volume_factor = df['volume'].iloc[-1] / volume_ma.iloc[-1]
                df.iloc[-1, df.columns.get_loc('signal_strength')] = min(
                    1.0,
                    (breakout_size * 0.7 + volume_factor * 0.3)
                )
                break
                
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on breakout strength and volatility.
        
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
        
        # Adjust for volatility (reduce size in high volatility)
        vol_factor = 1 / (1 + volatility * 2)
        
        # Adjust for breakout strength
        breakout_factor = 0.5 + (signal_strength * 0.5)
        
        # Calculate final position size
        position_size = base_size * vol_factor * breakout_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on ATR and support/resistance levels.
        
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
        
        if side == 'buy':
            # Initial stop below the breakout level
            if not current_stop:
                # Find nearest support level
                lower_levels = [s for s in self.support_levels if s < entry_price]
                if lower_levels:
                    updates['stop_loss'] = max(lower_levels) - (atr * self.params.atr_multiplier)
                else:
                    updates['stop_loss'] = entry_price - (atr * self.params.atr_multiplier)
            else:
                # Trail stop as price moves up
                new_stop = current_price - (atr * self.params.atr_multiplier)
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop above the breakout level
            if not current_stop:
                # Find nearest resistance level
                upper_levels = [r for r in self.resistance_levels if r > entry_price]
                if upper_levels:
                    updates['stop_loss'] = min(upper_levels) + (atr * self.params.atr_multiplier)
                else:
                    updates['stop_loss'] = entry_price + (atr * self.params.atr_multiplier)
            else:
                # Trail stop as price moves down
                new_stop = current_price + (atr * self.params.atr_multiplier)
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates 