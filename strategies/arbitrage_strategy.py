from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.core.strategy_base import StrategyBase

@dataclass
class ArbitrageParams:
    """Arbitrage Strategy Parameters"""
    # Price Difference Parameters
    min_price_diff: float = 0.002  # Minimum 0.2% price difference
    max_price_diff: float = 0.05  # Maximum 5% price difference
    
    # Execution Parameters
    max_execution_time: int = 5  # Maximum seconds for execution
    min_profit_after_fees: float = 0.001  # Minimum 0.1% profit after fees
    
    # Volume Parameters
    min_volume_usd: float = 10000  # Minimum volume in USD
    volume_ma_period: int = 20
    min_volume_factor: float = 1.5
    
    # Risk Parameters
    max_position_value: float = 0.2  # Maximum 20% of portfolio per trade
    max_open_trades: int = 3
    
    # Liquidity Parameters
    min_order_book_depth: float = 0.8  # Minimum 80% of intended trade size
    slippage_factor: float = 1.002  # Expected 0.2% slippage
    
class ArbitrageStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Arbitrage",
                 params: Optional[ArbitrageParams] = None,
                 **kwargs):
        """
        Arbitrage strategy for exploiting price differences.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or ArbitrageParams()
        self.active_trades: Dict[str, Dict] = {}
        
    def _calculate_arbitrage_metrics(self,
                                   data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate arbitrage opportunities across exchanges."""
        # Initialize metrics DataFrame
        metrics = pd.DataFrame()
        
        # Get exchange data
        exchanges = list(data.keys())
        base_exchange = exchanges[0]
        
        # Calculate metrics for base exchange
        metrics['base_price'] = data[base_exchange]['close']
        metrics['base_volume'] = data[base_exchange]['volume']
        metrics['base_volume_ma'] = metrics['base_volume'].rolling(
            window=self.params.volume_ma_period
        ).mean()
        
        # Calculate price differences and opportunities
        for exchange in exchanges[1:]:
            # Price metrics
            metrics[f'{exchange}_price'] = data[exchange]['close']
            metrics[f'{exchange}_volume'] = data[exchange]['volume']
            metrics[f'{exchange}_volume_ma'] = metrics[f'{exchange}_volume'].rolling(
                window=self.params.volume_ma_period
            ).mean()
            
            # Calculate price difference
            metrics[f'{exchange}_price_diff'] = (
                metrics[f'{exchange}_price'] - metrics['base_price']
            ) / metrics['base_price']
            
            # Calculate volume-weighted opportunity score
            volume_ratio = metrics[f'{exchange}_volume'] / metrics[f'{exchange}_volume_ma']
            metrics[f'{exchange}_opportunity'] = metrics[f'{exchange}_price_diff'] * volume_ratio
            
        return metrics
        
    def _validate_opportunity(self,
                            price_diff: float,
                            base_volume: float,
                            target_volume: float,
                            base_depth: Dict,
                            target_depth: Dict) -> bool:
        """Validate if arbitrage opportunity is executable."""
        # Check price difference thresholds
        if not (self.params.min_price_diff <= abs(price_diff) <= self.params.max_price_diff):
            return False
            
        # Check volume thresholds
        if min(base_volume, target_volume) < self.params.min_volume_usd:
            return False
            
        # Check order book depth
        base_liquidity = sum(level['size'] for level in base_depth['bids'])
        target_liquidity = sum(level['size'] for level in target_depth['asks'])
        
        min_required_depth = self.params.min_order_book_depth * min(base_volume, target_volume)
        if min(base_liquidity, target_liquidity) < min_required_depth:
            return False
            
        # Calculate expected slippage
        expected_slippage = price_diff * self.params.slippage_factor
        
        # Check if profitable after fees and slippage
        return expected_slippage > self.params.min_profit_after_fees
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate trading signals based on arbitrage opportunities."""
        # Initialize output DataFrame
        df = pd.DataFrame(index=data[list(data.keys())[0]].index)
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate arbitrage metrics
        metrics = self._calculate_arbitrage_metrics(data)
        
        # Get exchanges
        exchanges = [ex for ex in data.keys() if ex != list(data.keys())[0]]
        
        for exchange in exchanges:
            # Get price difference and volumes
            price_diff = metrics[f'{exchange}_price_diff']
            base_volume = metrics['base_volume']
            target_volume = metrics[f'{exchange}_volume']
            
            # Volume confirmation
            volume_confirmed = (
                (base_volume > metrics['base_volume_ma'] * self.params.min_volume_factor) &
                (target_volume > metrics[f'{exchange}_volume_ma'] * self.params.min_volume_factor)
            )
            
            # Generate long signals (buy base, sell target)
            long_conditions = (
                (price_diff < -self.params.min_price_diff) &
                volume_confirmed &
                (len(self.active_trades) < self.params.max_open_trades)
            )
            
            # Generate short signals (sell base, buy target)
            short_conditions = (
                (price_diff > self.params.min_price_diff) &
                volume_confirmed &
                (len(self.active_trades) < self.params.max_open_trades)
            )
            
            # Update signals
            df.loc[long_conditions, 'signal'] = 1
            df.loc[short_conditions, 'signal'] = -1
            
            # Calculate signal strength based on price difference
            df.loc[long_conditions | short_conditions, 'signal_strength'] = abs(
                price_diff / self.params.max_price_diff
            ).clip(0, 1)
            
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size for arbitrage trades.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Current market volatility
            
        Returns:
            Position size in base currency
        """
        # Calculate total exposure
        total_exposure = sum(
            trade.get('position_value', 0)
            for trade in self.active_trades.values()
        )
        
        # Calculate remaining available exposure
        max_exposure = account_balance * self.params.max_position_value
        available_exposure = max_exposure - total_exposure
        
        if available_exposure <= 0:
            return 0
            
        # Get latest signal strength
        signal_strength = self.current_signal_strength if hasattr(self, 'current_signal_strength') else 0.5
        
        # Calculate position size based on opportunity size
        position_size = (available_exposure * signal_strength) / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on arbitrage parameters.
        
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
        
        updates = {}
        
        # Calculate stop distance based on min profit requirement
        stop_distance = entry_price * (
            self.params.min_profit_after_fees + self.params.slippage_factor
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
        
    def update_active_trades(self,
                           trade_id: str,
                           trade_data: Dict) -> None:
        """
        Update active trades dictionary.
        
        Args:
            trade_id: Unique trade identifier
            trade_data: Trade information
        """
        if trade_data.get('status') == 'closed':
            self.active_trades.pop(trade_id, None)
        else:
            self.active_trades[trade_id] = trade_data 