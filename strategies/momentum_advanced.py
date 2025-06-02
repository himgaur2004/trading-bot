from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase

@dataclass
class MomentumAdvancedParams:
    """Advanced Momentum Strategy Parameters"""
    # ROC Parameters
    roc_period: int = 10
    roc_threshold: float = 0.02  # 2% minimum rate of change
    
    # Momentum Parameters
    momentum_period: int = 14
    momentum_ma_period: int = 10
    
    # Volume Parameters
    volume_ma_period: int = 20
    min_volume_factor: float = 2.0  # Higher volume requirement for momentum
    volume_roc_period: int = 5
    min_volume_roc: float = 0.5  # 50% minimum volume increase
    
    # Trend Filter Parameters
    trend_ema_period: int = 100
    min_trend_strength: float = 0.02  # 2% minimum trend strength
    
    # Volatility Parameters
    atr_period: int = 14
    min_volatility_percentile: float = 60  # Minimum volatility percentile
    
    # Momentum Ranking Parameters
    ranking_period: int = 20
    min_momentum_rank: float = 0.8  # Top 20% momentum
    
class MomentumAdvancedStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Momentum_Advanced",
                 params: Optional[MomentumAdvancedParams] = None,
                 **kwargs):
        """
        Advanced Momentum strategy using multiple indicators and filters.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or MomentumAdvancedParams()
        
    def _calculate_momentum_rank(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum ranking."""
        # Calculate returns over different periods
        returns = {}
        for period in [5, 10, 20, 60]:
            returns[period] = data['close'].pct_change(period)
            
        # Combine returns with weights (more weight to recent periods)
        weights = {5: 0.4, 10: 0.3, 20: 0.2, 60: 0.1}
        momentum_score = sum(returns[period] * weights[period] for period in weights)
        
        # Calculate percentile rank
        rank = momentum_score.rolling(window=self.params.ranking_period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        return rank
        
    def _calculate_volume_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume profile indicators."""
        df = data.copy()
        
        # Volume ROC
        df['volume_roc'] = df['volume'].pct_change(self.params.volume_roc_period)
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        
        # Volume strength
        df['volume_strength'] = df['volume'] / df['volume_ma']
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on momentum indicators."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate ROC
        df['roc'] = df['close'].pct_change(self.params.roc_period)
        
        # Calculate Momentum
        df['momentum'] = df['close'] - df['close'].shift(self.params.momentum_period)
        df['momentum_ma'] = df['momentum'].rolling(window=self.params.momentum_ma_period).mean()
        
        # Calculate trend strength
        df['trend_ema'] = df['close'].ewm(span=self.params.trend_ema_period).mean()
        df['trend_strength'] = (df['close'] - df['trend_ema']) / df['trend_ema']
        
        # Calculate momentum rank
        df['momentum_rank'] = self._calculate_momentum_rank(df)
        
        # Calculate volume profile
        df = self._calculate_volume_profile(df)
        
        # Calculate volatility percentile
        df['volatility'] = df['atr'] / df['close']
        df['volatility_rank'] = df['volatility'].rolling(
            window=self.params.ranking_period
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Volume confirmation
        volume_confirmed = (
            (df['volume'] > df['volume_ma'] * self.params.min_volume_factor) &
            (df['volume_roc'] > self.params.min_volume_roc)
        )
        
        # Generate buy signals
        buy_conditions = (
            # Strong momentum
            (df['roc'] > self.params.roc_threshold) &
            (df['momentum'] > df['momentum_ma']) &
            # High momentum rank
            (df['momentum_rank'] > self.params.min_momentum_rank) &
            # Strong trend
            (df['trend_strength'] > self.params.min_trend_strength) &
            # Sufficient volatility
            (df['volatility_rank'] > self.params.min_volatility_percentile / 100) &
            # Volume confirmation
            volume_confirmed
        )
        
        # Generate sell signals
        sell_conditions = (
            # Weak momentum
            (df['roc'] < -self.params.roc_threshold) &
            (df['momentum'] < df['momentum_ma']) &
            # Low momentum rank
            (df['momentum_rank'] < (1 - self.params.min_momentum_rank)) &
            # Weak trend
            (df['trend_strength'] < -self.params.min_trend_strength) &
            # Sufficient volatility
            (df['volatility_rank'] > self.params.min_volatility_percentile / 100) &
            # Volume confirmation
            volume_confirmed
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength
        strength_factors = {
            'momentum': df['momentum_rank'],
            'trend': abs(df['trend_strength']) / self.params.min_trend_strength,
            'volume': df['volume_strength'],
            'volatility': df['volatility_rank']
        }
        
        # Weighted signal strength
        weights = {'momentum': 0.4, 'trend': 0.3, 'volume': 0.2, 'volatility': 0.1}
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
        Calculate position size based on momentum strength and volatility.
        
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
        
        # Adjust for volatility (larger size in high momentum)
        vol_factor = (1 + volatility) if volatility > 0 else 1
        
        # Adjust for momentum strength
        momentum_factor = 0.5 + (signal_strength * 0.5)
        
        # Calculate final position size
        position_size = base_size * vol_factor * momentum_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on momentum and volatility.
        
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
        momentum_rank = self.market_data['momentum_rank'].iloc[-1]
        
        updates = {}
        
        # Adjust stop distance based on momentum strength
        stop_multiplier = 2 * (1 + momentum_rank)  # Wider stops for stronger momentum
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                stop_distance = atr * stop_multiplier
                updates['stop_loss'] = entry_price - stop_distance
                updates['take_profit'] = entry_price + (stop_distance * 2)
            else:
                # Trail stop based on ATR and momentum
                new_stop = current_price - (atr * stop_multiplier)
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                stop_distance = atr * stop_multiplier
                updates['stop_loss'] = entry_price + stop_distance
                updates['take_profit'] = entry_price - (stop_distance * 2)
            else:
                # Trail stop based on ATR and momentum
                new_stop = current_price + (atr * stop_multiplier)
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates 