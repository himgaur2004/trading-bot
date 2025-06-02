from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase
from backend.utils.helpers import calculate_volume_profile

@dataclass
class OrderFlowParams:
    """Order Flow Strategy Parameters"""
    # Order Book Parameters
    book_depth: int = 20  # Levels of order book to analyze
    min_book_imbalance: float = 0.2  # Minimum order book imbalance
    min_book_depth_usd: float = 100000  # Minimum depth in USD
    
    # Trade Flow Parameters
    trade_window: int = 100  # Number of trades to analyze
    min_trade_size: float = 1000  # Minimum trade size in USD
    large_trade_factor: float = 5  # Factor for large trade detection
    
    # Volume Profile Parameters
    volume_levels: int = 50  # Number of volume profile levels
    volume_threshold: float = 0.7  # Volume concentration threshold
    
    # Time and Sales Parameters
    tick_window: int = 1000  # Number of ticks to analyze
    aggressive_trade_threshold: float = 0.6  # Threshold for aggressive trades
    
    # Signal Parameters
    min_confidence: float = 0.6  # Minimum confidence for signals
    max_position_value: float = 0.2  # Maximum 20% of portfolio per trade
    
class OrderFlowStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Order_Flow",
                 params: Optional[OrderFlowParams] = None,
                 **kwargs):
        """
        Order Flow strategy analyzing market microstructure.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or OrderFlowParams()
        self.active_positions: Dict[str, Dict] = {}
        
    def _calculate_order_book_metrics(self,
                                    bids: List[Dict],
                                    asks: List[Dict]) -> Dict[str, float]:
        """Calculate order book metrics."""
        metrics = {}
        
        # Calculate book depth
        bid_depth = sum(level['size'] * level['price'] for level in bids[:self.params.book_depth])
        ask_depth = sum(level['size'] * level['price'] for level in asks[:self.params.book_depth])
        
        # Calculate imbalance
        total_depth = bid_depth + ask_depth
        metrics['book_imbalance'] = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        # Calculate weighted average prices
        bid_wap = sum(level['size'] * level['price']**2 for level in bids[:self.params.book_depth])
        ask_wap = sum(level['size'] * level['price']**2 for level in asks[:self.params.book_depth])
        metrics['bid_wap'] = bid_wap / bid_depth if bid_depth > 0 else 0
        metrics['ask_wap'] = ask_wap / ask_depth if ask_depth > 0 else 0
        
        # Calculate spread and depth metrics
        metrics['spread'] = asks[0]['price'] - bids[0]['price']
        metrics['relative_spread'] = metrics['spread'] / asks[0]['price']
        metrics['total_depth'] = total_depth
        
        return metrics
        
    def _analyze_trade_flow(self, trades: List[Dict]) -> Dict[str, float]:
        """Analyze recent trade flow."""
        metrics = {}
        
        # Filter recent trades
        recent_trades = trades[-self.params.trade_window:]
        
        # Calculate buy/sell pressure
        buy_volume = sum(trade['size'] * trade['price'] for trade in recent_trades if trade['side'] == 'buy')
        sell_volume = sum(trade['size'] * trade['price'] for trade in recent_trades if trade['side'] == 'sell')
        
        total_volume = buy_volume + sell_volume
        metrics['buy_pressure'] = buy_volume / total_volume if total_volume > 0 else 0
        
        # Detect large trades
        trade_sizes = [trade['size'] * trade['price'] for trade in recent_trades]
        avg_size = np.mean(trade_sizes) if trade_sizes else 0
        large_trades = [size for size in trade_sizes if size > avg_size * self.params.large_trade_factor]
        
        metrics['large_trade_ratio'] = len(large_trades) / len(trade_sizes) if trade_sizes else 0
        metrics['avg_trade_size'] = avg_size
        
        return metrics
        
    def _calculate_volume_profile(self,
                                price_levels: np.ndarray,
                                volumes: np.ndarray) -> Dict[str, float]:
        """Calculate volume profile metrics."""
        metrics = {}
        
        # Create volume profile
        hist, bins = np.histogram(price_levels, bins=self.params.volume_levels, weights=volumes)
        
        # Calculate volume concentration
        total_volume = np.sum(hist)
        sorted_hist = np.sort(hist)[::-1]
        cumsum_volume = np.cumsum(sorted_hist)
        
        # Find price levels with highest volume concentration
        concentration_idx = np.where(cumsum_volume >= total_volume * self.params.volume_threshold)[0][0]
        metrics['volume_concentration'] = concentration_idx / self.params.volume_levels
        
        # Calculate volume weighted average price (VWAP)
        metrics['vwap'] = np.sum(price_levels * volumes) / np.sum(volumes)
        
        return metrics
        
    def _analyze_tick_data(self, ticks: List[Dict]) -> Dict[str, float]:
        """Analyze tick data for aggressive trades."""
        metrics = {}
        
        # Filter recent ticks
        recent_ticks = ticks[-self.params.tick_window:]
        
        # Calculate aggressive trade metrics
        aggressive_buys = sum(1 for tick in recent_ticks if tick.get('aggressive') and tick['side'] == 'buy')
        aggressive_sells = sum(1 for tick in recent_ticks if tick.get('aggressive') and tick['side'] == 'sell')
        
        total_trades = len(recent_ticks)
        if total_trades > 0:
            metrics['aggressive_buy_ratio'] = aggressive_buys / total_trades
            metrics['aggressive_sell_ratio'] = aggressive_sells / total_trades
            metrics['aggressive_ratio'] = (aggressive_buys + aggressive_sells) / total_trades
        else:
            metrics['aggressive_buy_ratio'] = 0
            metrics['aggressive_sell_ratio'] = 0
            metrics['aggressive_ratio'] = 0
            
        return metrics
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate trading signals based on order flow analysis."""
        # Initialize output DataFrame
        df = pd.DataFrame(index=data['market'].index)
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Get order book data
        order_book = data.get('order_book', {'bids': [], 'asks': []})
        book_metrics = self._calculate_order_book_metrics(
            order_book.get('bids', []),
            order_book.get('asks', [])
        )
        
        # Get trade flow data
        trade_metrics = self._analyze_trade_flow(data.get('trades', []))
        
        # Calculate volume profile
        price_data = data['market']['close'].values
        volume_data = data['market']['volume'].values
        volume_metrics = self._calculate_volume_profile(price_data, volume_data)
        
        # Analyze tick data
        tick_metrics = self._analyze_tick_data(data.get('ticks', []))
        
        # Generate buy signals
        buy_conditions = (
            # Strong buying pressure
            (trade_metrics['buy_pressure'] > self.params.aggressive_trade_threshold) &
            # Positive order book imbalance
            (book_metrics['book_imbalance'] > self.params.min_book_imbalance) &
            # Sufficient market depth
            (book_metrics['total_depth'] > self.params.min_book_depth_usd) &
            # High aggressive buying
            (tick_metrics['aggressive_buy_ratio'] > tick_metrics['aggressive_sell_ratio'])
        )
        
        # Generate sell signals
        sell_conditions = (
            # Strong selling pressure
            (trade_metrics['buy_pressure'] < (1 - self.params.aggressive_trade_threshold)) &
            # Negative order book imbalance
            (book_metrics['book_imbalance'] < -self.params.min_book_imbalance) &
            # Sufficient market depth
            (book_metrics['total_depth'] > self.params.min_book_depth_usd) &
            # High aggressive selling
            (tick_metrics['aggressive_sell_ratio'] > tick_metrics['aggressive_buy_ratio'])
        )
        
        # Set signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate signal strength
        df['signal_strength'] = np.where(
            df['signal'] == 1,
            trade_metrics['buy_pressure'],
            np.where(
                df['signal'] == -1,
                1 - trade_metrics['buy_pressure'],
                0
            )
        ).clip(0, 1)
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on order flow metrics.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Current market volatility
            
        Returns:
            Position size in base currency
        """
        # Get latest signal strength
        signal_strength = self.current_signal_strength if hasattr(self, 'current_signal_strength') else 0.5
        
        # Base position size
        base_size = account_balance * self.params.max_position_value
        
        # Adjust for order flow strength
        flow_factor = 0.5 + (signal_strength * 0.5)
        
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility)
        
        # Calculate final position size
        position_size = base_size * flow_factor * vol_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on order flow metrics.
        
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
            
        # Get order book metrics for stop calculation
        order_book = getattr(self, 'order_book', {'bids': [], 'asks': []})
        book_metrics = self._calculate_order_book_metrics(
            order_book.get('bids', []),
            order_book.get('asks', [])
        )
        
        updates = {}
        
        # Calculate stop distance based on order book depth
        stop_distance = current_price * max(
            0.01,  # Minimum 1% stop
            book_metrics['relative_spread'] * 2  # 2x the spread
        )
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price - stop_distance
                updates['take_profit'] = entry_price + (stop_distance * 2)
            else:
                # Trail stop using order book metrics
                new_stop = current_price - stop_distance
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price + stop_distance
                updates['take_profit'] = entry_price - (stop_distance * 2)
            else:
                # Trail stop using order book metrics
                new_stop = current_price + stop_distance
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