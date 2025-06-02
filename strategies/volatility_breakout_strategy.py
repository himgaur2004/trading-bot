from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase

@dataclass
class VolatilityBreakoutParams:
    """Volatility Breakout Strategy Parameters"""
    # Volatility Parameters
    atr_period: int = 14
    volatility_ma_period: int = 20
    volatility_threshold: float = 1.5  # Minimum volatility expansion
    
    # Breakout Parameters
    breakout_period: int = 20
    breakout_std: float = 2.0  # Standard deviations for breakout
    
    # Volume Parameters
    volume_ma_period: int = 20
    min_volume_factor: float = 2.0  # Higher volume requirement for breakouts
    
    # Trend Parameters
    trend_ema_period: int = 100
    min_trend_strength: float = 0.02  # 2% minimum trend strength
    
    # Volatility Ranking Parameters
    ranking_period: int = 100
    min_volatility_rank: float = 0.7  # Top 30% volatility
    
    # Consolidation Parameters
    consolidation_period: int = 10
    max_consolidation_range: float = 0.02  # Maximum range for consolidation
    
class VolatilityBreakoutStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Volatility_Breakout",
                 params: Optional[VolatilityBreakoutParams] = None,
                 **kwargs):
        """
        Volatility Breakout strategy with advanced volatility analysis.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or VolatilityBreakoutParams()
        
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility metrics."""
        df = data.copy()
        
        # Calculate ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.params.atr_period).mean()
        
        # Calculate volatility ratio
        df['volatility'] = df['atr'] / df['close']
        df['volatility_ma'] = df['volatility'].rolling(window=self.params.volatility_ma_period).mean()
        df['volatility_ratio'] = df['volatility'] / df['volatility_ma']
        
        # Calculate volatility rank
        df['volatility_rank'] = df['volatility'].rolling(
            window=self.params.ranking_period
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        return df
        
    def _calculate_breakout_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout levels."""
        df = data.copy()
        
        # Calculate price channels
        df['upper_channel'] = df['high'].rolling(window=self.params.breakout_period).max()
        df['lower_channel'] = df['low'].rolling(window=self.params.breakout_period).min()
        
        # Calculate Bollinger Bands
        df['price_ma'] = df['close'].rolling(window=self.params.breakout_period).mean()
        df['price_std'] = df['close'].rolling(window=self.params.breakout_period).std()
        df['upper_band'] = df['price_ma'] + (df['price_std'] * self.params.breakout_std)
        df['lower_band'] = df['price_ma'] - (df['price_std'] * self.params.breakout_std)
        
        # Calculate consolidation
        df['price_range'] = (df['high'].rolling(window=self.params.consolidation_period).max() -
                           df['low'].rolling(window=self.params.consolidation_period).min()) / df['close']
        df['is_consolidating'] = df['price_range'] < self.params.max_consolidation_range
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on volatility breakouts."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate volatility metrics
        df = self._calculate_volatility_metrics(df)
        
        # Calculate breakout levels
        df = self._calculate_breakout_levels(df)
        
        # Calculate trend
        df['ema'] = df['close'].ewm(span=self.params.trend_ema_period).mean()
        df['trend_strength'] = (df['close'] - df['ema']) / df['close']
        
        # Volume confirmation
        volume_ma = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = df['volume'] > volume_ma * self.params.min_volume_factor
        
        # Volatility expansion filter
        volatility_expansion = df['volatility_ratio'] > self.params.volatility_threshold
        
        # High volatility environment
        high_volatility = df['volatility_rank'] > self.params.min_volatility_rank
        
        # Generate buy signals
        buy_conditions = (
            # Price breaks above upper band
            (df['close'] > df['upper_band']) &
            # Confirmed by channel breakout
            (df['close'] > df['upper_channel']) &
            # Coming from consolidation
            df['is_consolidating'].shift(1) &
            # Strong uptrend
            (df['trend_strength'] > self.params.min_trend_strength) &
            # Volatility conditions
            volatility_expansion &
            high_volatility &
            # Volume confirmation
            volume_confirmed
        )
        
        # Generate sell signals
        sell_conditions = (
            # Price breaks below lower band
            (df['close'] < df['lower_band']) &
            # Confirmed by channel breakout
            (df['close'] < df['lower_channel']) &
            # Coming from consolidation
            df['is_consolidating'].shift(1) &
            # Strong downtrend
            (df['trend_strength'] < -self.params.min_trend_strength) &
            # Volatility conditions
            volatility_expansion &
            high_volatility &
            # Volume confirmation
            volume_confirmed
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength
        strength_factors = {
            'volatility': df['volatility_ratio'] / self.params.volatility_threshold,
            'trend': abs(df['trend_strength']) / self.params.min_trend_strength,
            'volume': df['volume'] / volume_ma,
            'breakout': abs(df['close'] - df['price_ma']) / df['price_std']
        }
        
        # Weighted signal strength
        weights = {'volatility': 0.4, 'trend': 0.2, 'volume': 0.2, 'breakout': 0.2}
        signal_strength = sum(
            strength_factors[factor] * weight
            for factor, weight in weights.items()
        ).clip(0, 1)
        
        df.loc[buy_conditions | sell_conditions, 'signal_strength'] = signal_strength[
            buy_conditions | sell_conditions
        ]
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on volatility metrics.
        
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
        
        # Adjust for volatility (smaller size in high volatility)
        vol_factor = 1 / (1 + volatility * 2)
        
        # Adjust for signal strength
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
        Update stops based on volatility and breakout levels.
        
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
            
        atr = self.market_data['atr'].iloc[-1]
        volatility_ratio = self.market_data['volatility_ratio'].iloc[-1]
        
        updates = {}
        
        # Adjust stop distance based on volatility
        stop_distance = atr * max(2, volatility_ratio)
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price - stop_distance
                updates['take_profit'] = entry_price + (stop_distance * 2)
            else:
                # Trail stop using ATR
                new_stop = current_price - stop_distance
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price + stop_distance
                updates['take_profit'] = entry_price - (stop_distance * 2)
            else:
                # Trail stop using ATR
                new_stop = current_price + stop_distance
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates 