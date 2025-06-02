from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

@dataclass
class RiskParameters:
    """Risk Management Parameters"""
    max_position_size: float = 0.1  # Maximum position size as % of account
    max_daily_drawdown: float = 0.02  # Maximum daily drawdown allowed
    max_open_positions: int = 3  # Maximum number of concurrent positions
    position_sizing_type: str = 'fixed'  # 'fixed', 'kelly', 'volatility'
    risk_per_trade: float = 0.01  # Risk per trade as % of account
    max_leverage: float = 1.0  # Maximum allowed leverage
    trade_cooldown: int = 300  # Seconds between trades
    correlation_threshold: float = 0.7  # Maximum correlation between positions

class RiskManager:
    def __init__(self, params: Optional[RiskParameters] = None):
        """
        Initialize Risk Manager.
        
        Args:
            params: Risk management parameters
        """
        self.params = params or RiskParameters()
        self.open_positions: List[Dict] = []
        self.daily_pnl: List[float] = []
        self.last_trade_time: Optional[datetime] = None
        self.position_correlations: Dict[str, pd.Series] = {}
        
    def can_open_position(self, 
                         account_balance: float,
                         symbol: str,
                         price_data: pd.DataFrame) -> bool:
        """
        Check if a new position can be opened.
        
        Args:
            account_balance: Current account balance
            symbol: Trading symbol
            price_data: Historical price data
            
        Returns:
            Boolean indicating if position can be opened
        """
        # Check number of open positions
        if len(self.open_positions) >= self.params.max_open_positions:
            return False
            
        # Check daily drawdown
        if self.get_daily_drawdown() <= -self.params.max_daily_drawdown:
            return False
            
        # Check trade cooldown
        if self.last_trade_time and \
           datetime.now() - self.last_trade_time < timedelta(seconds=self.params.trade_cooldown):
            return False
            
        # Check correlation with existing positions
        if not self._check_correlation(symbol, price_data):
            return False
            
        return True
    
    def calculate_position_size(self,
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              win_rate: float = 0.5,
                              volatility: float = None) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            win_rate: Historical win rate
            volatility: Market volatility (for volatility-based sizing)
            
        Returns:
            Position size in base currency
        """
        risk_amount = account_balance * self.params.risk_per_trade
        
        if self.params.position_sizing_type == 'fixed':
            position_size = risk_amount / abs(entry_price - stop_loss)
            
        elif self.params.position_sizing_type == 'kelly':
            win_ratio = win_rate
            loss_ratio = 1 - win_rate
            avg_win_loss_ratio = abs(entry_price - stop_loss)  # Simplified
            kelly_fraction = (win_ratio * avg_win_loss_ratio - loss_ratio) / avg_win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, self.params.max_position_size))
            position_size = (account_balance * kelly_fraction) / entry_price
            
        elif self.params.position_sizing_type == 'volatility':
            if volatility is None:
                raise ValueError("Volatility required for volatility-based sizing")
            vol_adjusted_size = self.params.max_position_size * (1 / volatility)
            position_size = min(vol_adjusted_size, self.params.max_position_size) * account_balance / entry_price
            
        else:
            raise ValueError(f"Unknown position sizing type: {self.params.position_sizing_type}")
            
        # Apply maximum position size limit
        max_size = account_balance * self.params.max_position_size / entry_price
        position_size = min(position_size, max_size)
        
        return position_size
    
    def add_position(self, position: Dict):
        """
        Add a new position to tracking.
        
        Args:
            position: Position details dictionary
        """
        self.open_positions.append(position)
        self.last_trade_time = datetime.now()
        
    def close_position(self, position: Dict, pnl: float):
        """
        Close a position and update metrics.
        
        Args:
            position: Position to close
            pnl: Realized PnL
        """
        self.open_positions.remove(position)
        self.daily_pnl.append(pnl)
        
    def get_daily_drawdown(self) -> float:
        """Calculate current daily drawdown."""
        if not self.daily_pnl:
            return 0
        return sum(self.daily_pnl) / max(1, abs(max(self.daily_pnl)))
    
    def _check_correlation(self, symbol: str, price_data: pd.DataFrame) -> bool:
        """
        Check if new position would be too correlated with existing ones.
        
        Args:
            symbol: Symbol to check
            price_data: Price data for correlation calculation
            
        Returns:
            Boolean indicating if correlation is acceptable
        """
        if not self.open_positions:
            return True
            
        # Update correlation matrix
        returns = price_data['close'].pct_change()
        self.position_correlations[symbol] = returns
        
        # Check correlation with existing positions
        for pos in self.open_positions:
            if pos['symbol'] in self.position_correlations:
                corr = returns.corr(self.position_correlations[pos['symbol']])
                if abs(corr) > self.params.correlation_threshold:
                    return False
                    
        return True
    
    def adjust_risk_parameters(self, market_condition: str, volatility: float):
        """
        Adjust risk parameters based on market conditions.
        
        Args:
            market_condition: Current market condition
            volatility: Current market volatility
        """
        base_risk = self.params.risk_per_trade
        
        if market_condition == 'volatile':
            self.params.risk_per_trade = base_risk * 0.7
            self.params.max_position_size *= 0.8
            self.params.trade_cooldown *= 1.5
        elif market_condition == 'trending':
            self.params.risk_per_trade = base_risk * 1.2
            self.params.max_position_size *= 1.1
        else:  # ranging
            self.params.risk_per_trade = base_risk
            self.params.max_position_size = self.params.max_position_size
            
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        return {
            'open_positions': len(self.open_positions),
            'daily_drawdown': self.get_daily_drawdown(),
            'risk_per_trade': self.params.risk_per_trade,
            'max_position_size': self.params.max_position_size,
            'trade_cooldown': self.params.trade_cooldown
        } 