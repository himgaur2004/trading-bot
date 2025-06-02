from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase
from backend.utils.helpers import calculate_atr, calculate_support_resistance

@dataclass
class GridParams:
    """Grid Trading Strategy Parameters"""
    # Grid Parameters
    num_grids: int = 10  # Number of price levels
    grid_size_atr_factor: float = 0.5  # Grid size as factor of ATR
    max_active_positions: int = 5  # Maximum concurrent positions
    
    # Range Parameters
    range_period: int = 100  # Period for range calculation
    range_std_dev: float = 2.0  # Standard deviations for range
    min_range_size: float = 0.05  # Minimum range size as % of price
    
    # Volatility Parameters
    atr_period: int = 14
    volatility_ma_period: int = 20
    min_volatility_percentile: float = 30  # Minimum volatility percentile
    
    # Volume Parameters
    volume_ma_period: int = 20
    min_volume_factor: float = 1.2
    
    # Position Parameters
    position_size_factor: float = 0.1  # Base position size as % of account
    max_grid_exposure: float = 0.5  # Maximum total exposure in grid
    
class GridTradingStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Grid_Trading",
                 params: Optional[GridParams] = None,
                 **kwargs):
        """
        Grid Trading strategy with dynamic grid sizing.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or GridParams()
        self.grid_levels: List[float] = []
        self.active_positions: Dict[float, Dict] = {}
        self.grids: Dict[str, List[Dict]] = {}
        
    def _calculate_grid_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate grid trading metrics."""
        df = data.copy()
        
        # Calculate ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.params.atr_period).mean()
        
        # Calculate volatility percentile
        df['volatility'] = df['atr'] / df['close']
        df['volatility_ma'] = df['volatility'].rolling(window=self.params.volatility_ma_period).mean()
        df['volatility_rank'] = df['volatility'].rolling(
            window=self.params.range_period
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Calculate range boundaries
        df['price_ma'] = df['close'].rolling(window=self.params.range_period).mean()
        df['price_std'] = df['close'].rolling(window=self.params.range_period).std()
        df['upper_range'] = df['price_ma'] + (df['price_std'] * self.params.range_std_dev)
        df['lower_range'] = df['price_ma'] - (df['price_std'] * self.params.range_std_dev)
        
        # Calculate range size
        df['range_size'] = (df['upper_range'] - df['lower_range']) / df['close']
        
        return df
        
    def _calculate_grid_levels(self, data: pd.DataFrame) -> List[float]:
        """Calculate grid price levels."""
        current_price = data['close'].iloc[-1]
        atr = data['atr'].iloc[-1]
        upper_range = data['upper_range'].iloc[-1]
        lower_range = data['lower_range'].iloc[-1]
        
        # Adjust range if too small
        range_size = (upper_range - lower_range) / current_price
        if range_size < self.params.min_range_size:
            half_min_range = current_price * self.params.min_range_size / 2
            upper_range = current_price + half_min_range
            lower_range = current_price - half_min_range
            
        # Calculate grid size based on ATR
        grid_size = max(
            atr * self.params.grid_size_atr_factor,
            (upper_range - lower_range) / self.params.num_grids
        )
        
        # Generate grid levels
        grid_levels = []
        current_level = lower_range
        while current_level <= upper_range:
            grid_levels.append(current_level)
            current_level += grid_size
            
        return sorted(grid_levels)
        
    def _find_active_grid_levels(self,
                                current_price: float,
                                grid_levels: List[float]) -> Tuple[float, float]:
        """Find the active grid levels around current price."""
        for i in range(len(grid_levels) - 1):
            if grid_levels[i] <= current_price <= grid_levels[i + 1]:
                return grid_levels[i], grid_levels[i + 1]
        return None, None
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on grid levels."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate grid metrics
        df = self._calculate_grid_metrics(df)
        
        # Volume confirmation
        volume_ma = df['volume'].rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = df['volume'] > volume_ma * self.params.min_volume_factor
        
        # Volatility filter
        good_volatility = df['volatility_rank'] > self.params.min_volatility_percentile / 100
        
        # Check if range is valid
        valid_range = df['range_size'] >= self.params.min_range_size
        
        # Generate grid levels
        self.grid_levels = self._calculate_grid_levels(df.iloc[-1:])
        
        # Get current price and find active grid levels
        current_price = df['close'].iloc[-1]
        lower_grid, upper_grid = self._find_active_grid_levels(current_price, self.grid_levels)
        
        if lower_grid is None or upper_grid is None:
            return df
            
        # Generate buy signals (at lower grid)
        buy_conditions = (
            (df['close'] <= lower_grid) &
            volume_confirmed &
            good_volatility &
            valid_range &
            (len(self.active_positions) < self.params.max_active_positions)
        )
        
        # Generate sell signals (at upper grid)
        sell_conditions = (
            (df['close'] >= upper_grid) &
            volume_confirmed &
            good_volatility &
            valid_range &
            (len(self.active_positions) < self.params.max_active_positions)
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength based on position in grid
        grid_range = upper_grid - lower_grid
        price_position = (current_price - lower_grid) / grid_range
        
        # Stronger signals when price is closer to grid levels
        df['signal_strength'] = np.where(
            df['signal'] == 1,
            1 - price_position,  # Buy strength
            price_position  # Sell strength
        ).clip(0, 1)
        
        # Store current grid
        symbol = data.get('symbol', 'unknown')
        self.grids[symbol] = self.grid_levels
        
        # Update current signal
        self.current_signal = df['signal'].iloc[-1]
        self.current_signal_strength = df['signal_strength'].iloc[-1]
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size for grid trades.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Current market volatility
            
        Returns:
            Position size in base currency
        """
        # Calculate total grid exposure
        total_exposure = sum(
            pos.get('position_value', 0)
            for pos in self.active_positions.values()
        )
        
        # Calculate remaining available exposure
        max_exposure = account_balance * self.params.max_grid_exposure
        available_exposure = max_exposure - total_exposure
        
        if available_exposure <= 0:
            return 0
            
        # Base position size
        base_size = account_balance * self.params.position_size_factor
        
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility)
        
        # Calculate final position size
        position_size = min(
            base_size * vol_factor,
            available_exposure
        ) / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        # Adjust size based on signal strength
        adjusted_size = position_size * self.current_signal_strength
        
        # Scale size based on grid level
        if self.current_signal != 0:
            grid_factor = 1 + (abs(self.current_signal_strength) * 0.5)
            adjusted_size *= grid_factor
            
        return adjusted_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on grid levels.
        
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
            
        # Get ATR for stop calculation
        atr = self.market_data['atr'].iloc[-1]
        grid_size = atr * self.params.grid_size_atr_factor
        
        updates = {}
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price - grid_size
                updates['take_profit'] = entry_price + grid_size
            else:
                # Trail stop to next grid level below
                new_stop = current_price - grid_size
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price + grid_size
                updates['take_profit'] = entry_price - grid_size
            else:
                # Trail stop to next grid level above
                new_stop = current_price + grid_size
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates
        
    def update_active_positions(self,
                              position_id: str,
                              position_data: Dict) -> None:
        """
        Update active positions dictionary.
        
        Args:
            position_id: Unique position identifier
            position_data: Position information
        """
        if position_data.get('status') == 'closed':
            self.active_positions.pop(position_id, None)
        else:
            self.active_positions[position_id] = position_data 