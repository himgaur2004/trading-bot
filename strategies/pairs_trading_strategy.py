from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
from backend.core.strategy_base import StrategyBase

@dataclass
class PairsParams:
    """Pairs Trading Strategy Parameters"""
    # Cointegration Parameters
    lookback_period: int = 252  # One year of data
    coint_pvalue_threshold: float = 0.05
    min_half_life: int = 5
    max_half_life: int = 100
    
    # Z-Score Parameters
    zscore_entry: float = 2.0
    zscore_exit: float = 0.0
    zscore_stop: float = 4.0
    
    # Hedge Ratio Parameters
    hedge_lookback: int = 60
    min_hedge_ratio: float = 0.1
    max_hedge_ratio: float = 10.0
    
    # Volume Parameters
    volume_ma_period: int = 20
    min_volume_factor: float = 1.5
    
    # Position Parameters
    max_position_value: float = 0.2  # Maximum 20% of portfolio in a pair
    position_timeout: int = 20  # Maximum days in position
    
class PairsStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Pairs_Trading",
                 params: Optional[PairsParams] = None,
                 **kwargs):
        """
        Statistical arbitrage pairs trading strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or PairsParams()
        self.hedge_ratio = None
        self.zscore_mean = None
        self.zscore_std = None
        
    def _test_cointegration(self,
                           price1: pd.Series,
                           price2: pd.Series) -> Tuple[bool, float, float]:
        """
        Test for cointegration between two price series.
        
        Returns:
            Tuple of (is_cointegrated, p_value, half_life)
        """
        # Run cointegration test
        score, pvalue, _ = coint(price1, price2)
        
        if pvalue > self.params.coint_pvalue_threshold:
            return False, pvalue, np.inf
            
        # Calculate spread
        spread = self._calculate_spread(price1, price2)
        
        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)
        
        is_valid = (
            half_life > self.params.min_half_life and
            half_life < self.params.max_half_life
        )
        
        return is_valid, pvalue, half_life
        
    def _calculate_hedge_ratio(self,
                             price1: pd.Series,
                             price2: pd.Series) -> float:
        """Calculate dynamic hedge ratio using rolling regression."""
        # Prepare data
        X = price2.values.reshape(-1, 1)
        y = price1.values.reshape(-1, 1)
        
        # Fit regression
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        hedge_ratio = model.coef_[0][0]
        
        # Clip hedge ratio to valid range
        hedge_ratio = np.clip(
            hedge_ratio,
            self.params.min_hedge_ratio,
            self.params.max_hedge_ratio
        )
        
        return hedge_ratio
        
    def _calculate_spread(self,
                         price1: pd.Series,
                         price2: pd.Series) -> pd.Series:
        """Calculate normalized price spread."""
        if self.hedge_ratio is None:
            self.hedge_ratio = self._calculate_hedge_ratio(
                price1.iloc[-self.params.hedge_lookback:],
                price2.iloc[-self.params.hedge_lookback:]
            )
            
        spread = price1 - (self.hedge_ratio * price2)
        return spread
        
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion."""
        # Calculate lag-1 spread
        lag_spread = spread.shift(1)
        lag_spread = lag_spread.dropna()
        spread = spread[1:]
        
        # Run OLS regression
        X = lag_spread.values.reshape(-1, 1)
        y = (spread - lag_spread).values.reshape(-1, 1)
        
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        # Calculate half-life
        lambda_param = -model.coef_[0][0]
        half_life = np.log(2) / lambda_param if lambda_param > 0 else np.inf
        
        return half_life
        
    def _calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate z-score of spread."""
        if self.zscore_mean is None or self.zscore_std is None:
            self.zscore_mean = spread.mean()
            self.zscore_std = spread.std()
            
        zscore = (spread - self.zscore_mean) / self.zscore_std
        return zscore
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals based on pairs analysis.
        
        Args:
            data: Dictionary containing DataFrames for each asset in the pair
        """
        # Ensure we have data for both assets
        if len(data) != 2:
            raise ValueError("Pairs trading requires exactly two assets")
            
        # Get price series
        symbols = list(data.keys())
        price1 = data[symbols[0]]['close']
        price2 = data[symbols[1]]['close']
        
        # Initialize output DataFrame
        df = pd.DataFrame(index=price1.index)
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Test cointegration
        is_cointegrated, _, half_life = self._test_cointegration(
            price1.iloc[-self.params.lookback_period:],
            price2.iloc[-self.params.lookback_period:]
        )
        
        if not is_cointegrated:
            return df
            
        # Calculate spread and z-score
        spread = self._calculate_spread(price1, price2)
        zscore = self._calculate_zscore(spread)
        
        # Volume confirmation
        volume1 = data[symbols[0]]['volume']
        volume2 = data[symbols[1]]['volume']
        
        volume1_ma = volume1.rolling(window=self.params.volume_ma_period).mean()
        volume2_ma = volume2.rolling(window=self.params.volume_ma_period).mean()
        
        volume_confirmed = (
            (volume1 > volume1_ma * self.params.min_volume_factor) &
            (volume2 > volume2_ma * self.params.min_volume_factor)
        )
        
        # Generate long spread signals (long asset1, short asset2)
        long_spread_conditions = (
            (zscore < -self.params.zscore_entry) &
            volume_confirmed
        )
        
        # Generate short spread signals (short asset1, long asset2)
        short_spread_conditions = (
            (zscore > self.params.zscore_entry) &
            volume_confirmed
        )
        
        # Exit conditions
        exit_conditions = (
            (abs(zscore) < self.params.zscore_exit) |
            (abs(zscore) > self.params.zscore_stop)
        )
        
        # Set signals
        df.loc[long_spread_conditions, 'signal'] = 1
        df.loc[short_spread_conditions, 'signal'] = -1
        df.loc[exit_conditions, 'signal'] = 0
        
        # Calculate signal strength based on z-score
        df['signal_strength'] = abs(zscore / self.params.zscore_entry).clip(0, 1)
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size for each leg of the pair.
        
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
        max_position = account_balance * self.params.max_position_value
        base_size = max_position * signal_strength
        
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility)
        
        # Calculate final position size
        position_size = base_size * vol_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on spread z-score.
        
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
            
        # Get current z-score
        zscore = self._calculate_zscore(
            self._calculate_spread(
                self.market_data[self.symbols[0]]['close'],
                self.market_data[self.symbols[1]]['close']
            )
        ).iloc[-1]
        
        updates = {}
        
        if side == 'buy':  # Long spread position
            # Stop out if spread widens too much
            if zscore < -self.params.zscore_stop:
                updates['stop_loss'] = current_price * 0.99  # Force exit
            # Take profit if spread normalizes
            elif zscore > -self.params.zscore_exit:
                updates['take_profit'] = current_price * 1.01  # Force exit
                
        else:  # Short spread position
            # Stop out if spread widens too much
            if zscore > self.params.zscore_stop:
                updates['stop_loss'] = current_price * 1.01  # Force exit
            # Take profit if spread normalizes
            elif zscore < self.params.zscore_exit:
                updates['take_profit'] = current_price * 0.99  # Force exit
                
        return updates
        
    def update_pair_metrics(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Update pair trading metrics (hedge ratio, z-score parameters).
        
        Args:
            data: Dictionary containing DataFrames for each asset in the pair
        """
        symbols = list(data.keys())
        price1 = data[symbols[0]]['close']
        price2 = data[symbols[1]]['close']
        
        # Update hedge ratio
        self.hedge_ratio = self._calculate_hedge_ratio(
            price1.iloc[-self.params.hedge_lookback:],
            price2.iloc[-self.params.hedge_lookback:]
        )
        
        # Update z-score parameters
        spread = self._calculate_spread(price1, price2)
        self.zscore_mean = spread.mean()
        self.zscore_std = spread.std() 