from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase

@dataclass
class DonchianParams:
    """Donchian Channel Strategy Parameters"""
    # Channel Parameters
    channel_period: int = 20
    breakout_period: int = 55  # Longer period for major breakouts
    atr_period: int = 14
    
    # Trend Filter Parameters
    ema_period: int = 200
    min_trend_strength: float = 0.02  # 2% minimum trend strength
    
    # Volume Parameters
    volume_ma_period: int = 20
    min_volume_factor: float = 2.0  # Higher volume requirement for breakouts
    
    # Volatility Parameters
    volatility_lookback: int = 100
    min_volatility_percentile: float = 30  # Minimum volatility percentile
    
    # Channel Parameters
    min_channel_width: float = 0.02  # Minimum channel width as % of price
    
class DonchianStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Donchian_Channel",
                 params: Optional[DonchianParams] = None,
                 **kwargs):
        """
        Donchian Channel strategy with trend filtering and volatility adaptation.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or DonchianParams()
        
    def _calculate_donchian(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian Channels."""
        df = data.copy()
        
        # Calculate standard channels
        df['upper_channel'] = df['high'].rolling(window=self.params.channel_period).max()
        df['lower_channel'] = df['low'].rolling(window=self.params.channel_period).min()
        df['middle_channel'] = (df['upper_channel'] + df['lower_channel']) / 2
        
        # Calculate longer-term breakout levels
        df['breakout_high'] = df['high'].rolling(window=self.params.breakout_period).max()
        df['breakout_low'] = df['low'].rolling(window=self.params.breakout_period).min()
        
        # Calculate channel metrics
        df['channel_width'] = (df['upper_channel'] - df['lower_channel']) / df['close']
        
        return df
        
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility metrics."""
        df = data.copy()
        
        # Calculate ATR-based volatility
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.params.atr_period).mean()
        
        # Calculate volatility percentile
        df['volatility'] = df['atr'] / df['close']
        df['volatility_rank'] = df['volatility'].rolling(
            window=self.params.volatility_lookback
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Donchian Channels."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate Donchian Channels
        df = self._calculate_donchian(df)
        
        # Calculate volatility metrics
        df = self._calculate_volatility_metrics(df)
        
        # Calculate trend
        df['ema'] = df['close'].ewm(span=self.params.ema_period).mean()
        df['trend_strength'] = (df['close'] - df['ema']) / df['close']
        
        # Volume confirmation
        volume_ma = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = df['volume'] > volume_ma * self.params.min_volume_factor
        
        # Channel width filter
        wide_channel = df['channel_width'] > self.params.min_channel_width
        
        # Volatility filter
        good_volatility = df['volatility_rank'] > self.params.min_volatility_percentile / 100
        
        # Generate buy signals
        buy_conditions = (
            # Price breaks above channel
            (df['close'] > df['upper_channel']) &
            # Confirmed by breakout level
            (df['close'] > df['breakout_high']) &
            # Strong uptrend
            (df['trend_strength'] > self.params.min_trend_strength) &
            # Channel is wide enough
            wide_channel &
            # Good volatility environment
            good_volatility &
            # Volume confirmation
            volume_confirmed
        )
        
        # Generate sell signals
        sell_conditions = (
            # Price breaks below channel
            (df['close'] < df['lower_channel']) &
            # Confirmed by breakout level
            (df['close'] < df['breakout_low']) &
            # Strong downtrend
            (df['trend_strength'] < -self.params.min_trend_strength) &
            # Channel is wide enough
            wide_channel &
            # Good volatility environment
            good_volatility &
            # Volume confirmation
            volume_confirmed
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength
        strength_factors = {
            'trend': abs(df['trend_strength']) / self.params.min_trend_strength,
            'channel': df['channel_width'] / self.params.min_channel_width,
            'volume': df['volume'] / volume_ma,
            'volatility': df['volatility_rank']
        }
        
        # Weighted signal strength
        weights = {'trend': 0.3, 'channel': 0.3, 'volume': 0.2, 'volatility': 0.2}
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
        Calculate position size based on channel width and volatility.
        
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
        channel_factor = 0.5 + (signal_strength * 0.5)
        
        # Calculate final position size
        position_size = base_size * vol_factor * channel_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on Donchian Channels and ATR.
        
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
            
        lower_channel = self.market_data['lower_channel'].iloc[-1]
        upper_channel = self.market_data['upper_channel'].iloc[-1]
        atr = self.market_data['atr'].iloc[-1]
        
        updates = {}
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                stop_distance = max(
                    atr * 2,  # ATR-based stop
                    entry_price - lower_channel  # Channel-based stop
                )
                updates['stop_loss'] = entry_price - stop_distance
                updates['take_profit'] = entry_price + (stop_distance * 2)
            else:
                # Trail stop using lower channel
                new_stop = lower_channel
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                stop_distance = max(
                    atr * 2,  # ATR-based stop
                    upper_channel - entry_price  # Channel-based stop
                )
                updates['stop_loss'] = entry_price + stop_distance
                updates['take_profit'] = entry_price - (stop_distance * 2)
            else:
                # Trail stop using upper channel
                new_stop = upper_channel
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates 