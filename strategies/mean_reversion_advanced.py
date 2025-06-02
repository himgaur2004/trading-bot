from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from backend.core.strategy_base import StrategyBase

@dataclass
class MeanReversionAdvancedParams:
    """Advanced Mean Reversion Strategy Parameters"""
    # Bollinger Bands Parameters
    bb_period: int = 20
    bb_std: float = 2.0
    
    # RSI Parameters
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # Statistical Parameters
    zscore_period: int = 20
    zscore_threshold: float = 2.0
    std_dev_period: int = 50
    std_dev_threshold: float = 2.5
    
    # Volume Parameters
    volume_ma_period: int = 20
    min_volume_factor: float = 1.5
    
    # Price Deviation Parameters
    price_ma_period: int = 50
    max_deviation: float = 0.1  # 10% max deviation from MA
    
    # Divergence Parameters
    divergence_lookback: int = 10
    min_divergence_strength: float = 0.5
    
class MeanReversionAdvancedStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Mean_Reversion_Advanced",
                 params: Optional[MeanReversionAdvancedParams] = None,
                 **kwargs):
        """
        Advanced Mean Reversion strategy using multiple indicators and statistical measures.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or MeanReversionAdvancedParams()
        
    def _calculate_zscore(self, data: pd.Series) -> float:
        """Calculate z-score for latest value."""
        mean = data.rolling(window=self.params.zscore_period).mean()
        std = data.rolling(window=self.params.zscore_period).std()
        zscore = (data - mean) / std
        return zscore
        
    def _detect_divergence(self, price: pd.Series, indicator: pd.Series) -> Tuple[bool, bool]:
        """Detect bullish and bearish divergences."""
        lookback = self.params.divergence_lookback
        
        # Get local extrema
        price_min = price.rolling(window=lookback, center=True).min()
        price_max = price.rolling(window=lookback, center=True).max()
        ind_min = indicator.rolling(window=lookback, center=True).min()
        ind_max = indicator.rolling(window=lookback, center=True).max()
        
        # Detect divergences
        bullish_div = (
            (price.iloc[-1] < price.iloc[-lookback]) and  # Price making lower lows
            (indicator.iloc[-1] > indicator.iloc[-lookback])  # Indicator making higher lows
        )
        
        bearish_div = (
            (price.iloc[-1] > price.iloc[-lookback]) and  # Price making higher highs
            (indicator.iloc[-1] < indicator.iloc[-lookback])  # Indicator making lower highs
        )
        
        return bullish_div, bearish_div
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on mean reversion indicators."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate price MA and deviation
        df['price_ma'] = df['close'].rolling(window=self.params.price_ma_period).mean()
        df['price_deviation'] = (df['close'] - df['price_ma']) / df['price_ma']
        
        # Calculate z-score
        df['zscore'] = self._calculate_zscore(df['close'])
        
        # Calculate standard deviation
        df['rolling_std'] = df['close'].rolling(window=self.params.std_dev_period).std()
        df['std_dev_ratio'] = df['rolling_std'] / df['close']
        
        # Volume confirmation
        volume_ma = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = df['volume'] > volume_ma * self.params.min_volume_factor
        
        # Detect RSI divergences
        bullish_div, bearish_div = self._detect_divergence(df['close'], df['rsi'])
        
        # Generate buy signals
        buy_conditions = (
            # Oversold conditions
            (df['rsi'] < self.params.rsi_oversold) &
            (df['zscore'] < -self.params.zscore_threshold) &
            # Price deviation
            (df['price_deviation'] < -self.params.max_deviation) &
            # Volume confirmation
            volume_confirmed &
            # Additional filters
            (df['std_dev_ratio'] > self.params.std_dev_threshold) &
            bullish_div
        )
        
        # Generate sell signals
        sell_conditions = (
            # Overbought conditions
            (df['rsi'] > self.params.rsi_overbought) &
            (df['zscore'] > self.params.zscore_threshold) &
            # Price deviation
            (df['price_deviation'] > self.params.max_deviation) &
            # Volume confirmation
            volume_confirmed &
            # Additional filters
            (df['std_dev_ratio'] > self.params.std_dev_threshold) &
            bearish_div
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength
        df['signal_strength'] = 0.0
        
        # For buy signals
        buy_strength = (
            (abs(df['zscore']) / self.params.zscore_threshold) * 0.4 +
            (abs(df['price_deviation']) / self.params.max_deviation) * 0.3 +
            (df['std_dev_ratio'] / self.params.std_dev_threshold) * 0.3
        ).clip(0, 1)
        
        df.loc[buy_conditions, 'signal_strength'] = buy_strength[buy_conditions]
        df.loc[sell_conditions, 'signal_strength'] = buy_strength[sell_conditions]
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on mean reversion signals and volatility.
        
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
        
        # Adjust for volatility (larger size in lower volatility)
        vol_factor = 1 / (1 + volatility * 2)
        
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
        Update stops based on mean reversion levels and volatility.
        
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
            
        price_ma = self.market_data['price_ma'].iloc[-1]
        rolling_std = self.market_data['rolling_std'].iloc[-1]
        
        updates = {}
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                stop_distance = rolling_std * self.params.std_dev_threshold
                updates['stop_loss'] = entry_price - stop_distance
                updates['take_profit'] = price_ma  # Target the mean
            else:
                # Trail stop as price moves toward mean
                new_stop = current_price - (rolling_std * self.params.std_dev_threshold)
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                stop_distance = rolling_std * self.params.std_dev_threshold
                updates['stop_loss'] = entry_price + stop_distance
                updates['take_profit'] = price_ma  # Target the mean
            else:
                # Trail stop as price moves toward mean
                new_stop = current_price + (rolling_std * self.params.std_dev_threshold)
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates 