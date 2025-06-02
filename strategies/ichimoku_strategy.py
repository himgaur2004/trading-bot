from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase

@dataclass
class IchimokuParams:
    """Ichimoku Strategy Parameters"""
    # Ichimoku Parameters
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_span_b_period: int = 52
    displacement: int = 26
    
    # Volume Parameters
    volume_ma_period: int = 20
    min_volume_factor: float = 1.5
    
    # Trend Strength Parameters
    min_cloud_thickness: float = 0.002  # Minimum cloud thickness as % of price
    min_trend_strength: float = 0.01  # Minimum trend strength
    
    # Confirmation Parameters
    chikou_confirmation: bool = True  # Use Chikou span confirmation
    price_confirmation: bool = True  # Require price above/below cloud
    
class IchimokuStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Ichimoku_Cloud",
                 params: Optional[IchimokuParams] = None,
                 **kwargs):
        """
        Ichimoku Cloud strategy with advanced confirmation signals.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or IchimokuParams()
        
    def _calculate_ichimoku(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components."""
        df = data.copy()
        high = df['high']
        low = df['low']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=self.params.tenkan_period).max()
        tenkan_low = low.rolling(window=self.params.tenkan_period).min()
        df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=self.params.kijun_period).max()
        kijun_low = low.rolling(window=self.params.kijun_period).min()
        df['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.params.displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=self.params.senkou_span_b_period).max()
        senkou_low = low.rolling(window=self.params.senkou_span_b_period).min()
        df['senkou_span_b'] = ((senkou_high + senkou_low) / 2).shift(self.params.displacement)
        
        # Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-self.params.displacement)
        
        # Calculate cloud metrics
        df['cloud_top'] = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        df['cloud_bottom'] = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        df['cloud_thickness'] = (df['cloud_top'] - df['cloud_bottom']) / df['close']
        
        # Calculate trend strength
        df['trend_strength'] = (df['close'] - df['kijun_sen']) / df['close']
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Ichimoku Cloud."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate Ichimoku components
        df = self._calculate_ichimoku(df)
        
        # Volume confirmation
        volume_ma = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = df['volume'] > volume_ma * self.params.min_volume_factor
        
        # Cloud thickness filter
        thick_cloud = df['cloud_thickness'] > self.params.min_cloud_thickness
        
        # Generate buy signals
        buy_conditions = (
            # Price above cloud
            (df['close'] > df['cloud_top']) &
            # Conversion line above base line
            (df['tenkan_sen'] > df['kijun_sen']) &
            # Strong trend
            (df['trend_strength'] > self.params.min_trend_strength) &
            # Cloud is thick enough
            thick_cloud &
            # Volume confirmation
            volume_confirmed
        )
        
        if self.params.chikou_confirmation:
            buy_conditions &= df['chikou_span'] > df['close'].shift(self.params.displacement)
            
        if self.params.price_confirmation:
            buy_conditions &= df['close'] > df['senkou_span_a']
        
        # Generate sell signals
        sell_conditions = (
            # Price below cloud
            (df['close'] < df['cloud_bottom']) &
            # Conversion line below base line
            (df['tenkan_sen'] < df['kijun_sen']) &
            # Strong trend
            (df['trend_strength'] < -self.params.min_trend_strength) &
            # Cloud is thick enough
            thick_cloud &
            # Volume confirmation
            volume_confirmed
        )
        
        if self.params.chikou_confirmation:
            sell_conditions &= df['chikou_span'] < df['close'].shift(self.params.displacement)
            
        if self.params.price_confirmation:
            sell_conditions &= df['close'] < df['senkou_span_b']
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength
        strength_factors = {
            'trend': abs(df['trend_strength']) / self.params.min_trend_strength,
            'cloud': df['cloud_thickness'] / self.params.min_cloud_thickness,
            'volume': df['volume'] / volume_ma
        }
        
        # Weighted signal strength
        weights = {'trend': 0.4, 'cloud': 0.4, 'volume': 0.2}
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
        Calculate position size based on cloud metrics and trend strength.
        
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
        
        # Adjust for cloud metrics and trend strength
        cloud_factor = 0.5 + (signal_strength * 0.5)
        
        # Calculate final position size
        position_size = base_size * vol_factor * cloud_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on Ichimoku levels.
        
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
            
        kijun = self.market_data['kijun_sen'].iloc[-1]
        cloud_bottom = self.market_data['cloud_bottom'].iloc[-1]
        cloud_top = self.market_data['cloud_top'].iloc[-1]
        
        updates = {}
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = min(kijun, cloud_bottom)
                updates['take_profit'] = entry_price + (entry_price - updates['stop_loss']) * 2
            else:
                # Trail stop using Kijun-sen and cloud bottom
                new_stop = min(kijun, cloud_bottom)
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = max(kijun, cloud_top)
                updates['take_profit'] = entry_price - (updates['stop_loss'] - entry_price) * 2
            else:
                # Trail stop using Kijun-sen and cloud top
                new_stop = max(kijun, cloud_top)
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates 