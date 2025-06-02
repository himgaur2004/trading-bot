from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase
from backend.utils.helpers import calculate_volatility, calculate_vwap

@dataclass
class MarketMakingParams:
    """Market Making Strategy Parameters"""
    # Spread Parameters
    min_spread: float = 0.001  # Minimum 0.1% spread
    max_spread: float = 0.01  # Maximum 1% spread
    spread_multiplier: float = 1.5  # Multiplier for base spread
    
    # Inventory Parameters
    target_inventory: float = 0.0  # Target inventory position
    max_inventory: float = 1.0  # Maximum inventory as multiple of position size
    inventory_range: float = 0.5  # Range for inventory management
    
    # Order Parameters
    order_refresh_time: int = 30  # Seconds between order updates
    min_order_size: float = 100  # Minimum order size in USD
    max_order_size: float = 10000  # Maximum order size in USD
    
    # Risk Parameters
    max_position_value: float = 0.2  # Maximum 20% of portfolio per side
    max_open_orders: int = 10
    
    # Market Parameters
    volatility_window: int = 100
    volatility_threshold: float = 0.02  # Maximum acceptable volatility
    
    # Execution Parameters
    cancel_threshold: float = 0.002  # Cancel orders if price moves 0.2%
    min_profitability: float = 0.0005  # Minimum 0.05% profitability
    
class MarketMakingStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Market_Making",
                 params: Optional[MarketMakingParams] = None,
                 **kwargs):
        """
        Market Making strategy providing liquidity.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or MarketMakingParams()
        self.active_orders: Dict[str, Dict] = {}
        self.current_inventory: float = 0.0
        self.last_order_refresh: float = 0.0
        
    def _calculate_market_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market metrics for spread adjustment."""
        metrics = {}
        
        # Calculate volatility
        returns = data['close'].pct_change()
        metrics['volatility'] = returns.rolling(window=self.params.volatility_window).std()
        
        # Calculate volume metrics
        metrics['volume_ma'] = data['volume'].rolling(window=20).mean()
        metrics['volume_std'] = data['volume'].rolling(window=20).std()
        
        # Calculate price metrics
        metrics['price_ma'] = data['close'].rolling(window=20).mean()
        metrics['price_std'] = data['close'].rolling(window=20).std()
        
        return metrics
        
    def _calculate_optimal_spread(self,
                                current_price: float,
                                metrics: Dict[str, float]) -> Tuple[float, float]:
        """Calculate optimal bid-ask spread."""
        # Base spread from volatility
        base_spread = max(
            self.params.min_spread,
            metrics['volatility'] * self.params.spread_multiplier
        )
        
        # Adjust spread based on inventory
        inventory_factor = abs(self.current_inventory) / self.params.max_inventory
        inventory_spread = base_spread * (1 + inventory_factor)
        
        # Ensure spread is within limits
        final_spread = min(
            max(inventory_spread, self.params.min_spread),
            self.params.max_spread
        )
        
        # Calculate bid and ask prices
        half_spread = final_spread / 2
        bid_price = current_price * (1 - half_spread)
        ask_price = current_price * (1 + half_spread)
        
        return bid_price, ask_price
        
    def _calculate_order_sizes(self,
                             account_balance: float,
                             current_price: float) -> Tuple[float, float]:
        """Calculate optimal order sizes."""
        # Base order size
        base_size = account_balance * self.params.max_position_value
        
        # Adjust for inventory management
        inventory_ratio = self.current_inventory / self.params.max_inventory
        
        # Calculate bid size (larger when inventory is low)
        bid_size = base_size * (1 - inventory_ratio)
        bid_size = max(
            min(bid_size, self.params.max_order_size),
            self.params.min_order_size
        )
        
        # Calculate ask size (larger when inventory is high)
        ask_size = base_size * (1 + inventory_ratio)
        ask_size = max(
            min(ask_size, self.params.max_order_size),
            self.params.min_order_size
        )
        
        return bid_size / current_price, ask_size / current_price
        
    def _should_cancel_orders(self,
                            current_price: float,
                            order_book: Dict) -> bool:
        """Determine if orders should be cancelled."""
        if not self.active_orders:
            return False
            
        # Check price movement
        for order_id, order in self.active_orders.items():
            price_diff = abs(order['price'] - current_price) / current_price
            if price_diff > self.params.cancel_threshold:
                return True
                
        return False
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate market making signals."""
        # Initialize output DataFrame
        df = pd.DataFrame(index=data['market'].index)
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate market metrics
        metrics = self._calculate_market_metrics(data['market'])
        
        # Get current price and check volatility
        current_price = data['market']['close'].iloc[-1]
        current_volatility = metrics['volatility'].iloc[-1]
        
        if current_volatility > self.params.volatility_threshold:
            return df
            
        # Calculate optimal spread
        bid_price, ask_price = self._calculate_optimal_spread(current_price, metrics)
        
        # Check if spread is profitable
        spread = (ask_price - bid_price) / current_price
        if spread < self.params.min_profitability:
            return df
            
        # Generate buy signals (place bid)
        buy_conditions = (
            # Inventory not too high
            (self.current_inventory < self.params.max_inventory) &
            # Not too many open orders
            (len(self.active_orders) < self.params.max_open_orders)
        )
        
        # Generate sell signals (place ask)
        sell_conditions = (
            # Inventory not too low
            (self.current_inventory > -self.params.max_inventory) &
            # Not too many open orders
            (len(self.active_orders) < self.params.max_open_orders)
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength based on spread
        normalized_spread = (spread - self.params.min_spread) / (
            self.params.max_spread - self.params.min_spread
        )
        df['signal_strength'] = normalized_spread.clip(0, 1)
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size for market making.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Current market volatility
            
        Returns:
            Position size in base currency
        """
        # Calculate bid/ask sizes
        bid_size, ask_size = self._calculate_order_sizes(account_balance, current_price)
        
        # Use appropriate size based on signal
        if hasattr(self, 'current_signal') and self.current_signal == 1:
            position_size = bid_size
        elif hasattr(self, 'current_signal') and self.current_signal == -1:
            position_size = ask_size
        else:
            position_size = (bid_size + ask_size) / 2
            
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility * 2)
        position_size *= vol_factor
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops for market making positions.
        
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
            
        # Calculate market metrics
        metrics = self._calculate_market_metrics(self.market_data)
        current_volatility = metrics['volatility'].iloc[-1]
        
        updates = {}
        
        # Calculate stop distance based on volatility and spread
        stop_distance = current_price * max(
            self.params.min_spread * 2,
            current_volatility * self.params.spread_multiplier
        )
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price - stop_distance
                updates['take_profit'] = entry_price + (stop_distance * 2)
            else:
                # Trail stop to protect profits
                new_stop = current_price - stop_distance
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price + stop_distance
                updates['take_profit'] = entry_price - (stop_distance * 2)
            else:
                # Trail stop to protect profits
                new_stop = current_price + stop_distance
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates
        
    def update_inventory(self, trade_data: Dict) -> None:
        """
        Update current inventory based on executed trades.
        
        Args:
            trade_data: Trade execution information
        """
        if trade_data['side'] == 'buy':
            self.current_inventory += trade_data['size']
        else:
            self.current_inventory -= trade_data['size']
            
        # Clip inventory to limits
        self.current_inventory = np.clip(
            self.current_inventory,
            -self.params.max_inventory,
            self.params.max_inventory
        )
        
    def update_active_orders(self,
                           order_id: str,
                           order_data: Dict) -> None:
        """
        Update active orders dictionary.
        
        Args:
            order_id: Unique order identifier
            order_data: Order information
        """
        if order_data.get('status') in ['filled', 'cancelled']:
            self.active_orders.pop(order_id, None)
        else:
            self.active_orders[order_id] = order_data 