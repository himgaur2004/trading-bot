from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

class StrategyBase(ABC):
    def __init__(self, 
                 name: str,
                 timeframe: str = '1h',
                 risk_per_trade: float = 0.02,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.04):
        """
        Base class for all trading strategies.
        
        Args:
            name: Strategy name
            timeframe: Trading timeframe (e.g., '1m', '5m', '1h', '1d')
            risk_per_trade: Percentage of account to risk per trade
            stop_loss: Default stop loss percentage
            take_profit: Default take profit percentage
        """
        self.name = name
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Performance metrics
        self.trades: List[Dict] = []
        self.current_position: Optional[Dict] = None
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        
        Args:
            data: OHLCV data with technical indicators
            
        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, 
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Current market volatility
            
        Returns:
            Position size in base currency
        """
        pass
    
    def update_parameters(self, market_condition: str, volatility: float):
        """
        Update strategy parameters based on market conditions.
        
        Args:
            market_condition: Current market condition (trend/ranging/volatile)
            volatility: Current market volatility
        """
        if market_condition == 'volatile':
            self.risk_per_trade *= 0.8
            self.stop_loss *= 1.2
        elif market_condition == 'trending':
            self.take_profit *= 1.2
        
    def add_trade(self, 
                  entry_price: float,
                  position_size: float,
                  side: str,
                  timestamp: datetime):
        """
        Record a new trade.
        
        Args:
            entry_price: Entry price
            position_size: Position size
            side: Trade side (buy/sell)
            timestamp: Trade timestamp
        """
        trade = {
            'entry_price': entry_price,
            'position_size': position_size,
            'side': side,
            'entry_time': timestamp,
            'stop_loss': entry_price * (1 - self.stop_loss) if side == 'buy' 
                        else entry_price * (1 + self.stop_loss),
            'take_profit': entry_price * (1 + self.take_profit) if side == 'buy'
                         else entry_price * (1 - self.take_profit)
        }
        self.current_position = trade
        self.trades.append(trade)
        
    def close_trade(self, 
                    exit_price: float,
                    timestamp: datetime,
                    reason: str = 'signal'):
        """
        Close current trade and record results.
        
        Args:
            exit_price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing (signal/sl/tp)
        """
        if not self.current_position:
            return
            
        trade = self.current_position
        trade['exit_price'] = exit_price
        trade['exit_time'] = timestamp
        trade['duration'] = timestamp - trade['entry_time']
        trade['pnl'] = (exit_price - trade['entry_price']) * trade['position_size']
        if trade['side'] == 'sell':
            trade['pnl'] *= -1
        trade['pnl_percentage'] = trade['pnl'] / (trade['entry_price'] * trade['position_size'])
        trade['exit_reason'] = reason
        
        self.current_position = None
        
    def get_performance_metrics(self) -> Dict:
        """
        Calculate strategy performance metrics.
        
        Returns:
            Dict containing performance metrics
        """
        if not self.trades:
            return {}
            
        pnls = [t['pnl'] for t in self.trades if 'pnl' in t]
        win_trades = [p for p in pnls if p > 0]
        loss_trades = [p for p in pnls if p <= 0]
        
        metrics = {
            'total_trades': len(self.trades),
            'win_rate': len(win_trades) / len(pnls) if pnls else 0,
            'avg_win': np.mean(win_trades) if win_trades else 0,
            'avg_loss': np.mean(loss_trades) if loss_trades else 0,
            'largest_win': max(win_trades) if win_trades else 0,
            'largest_loss': min(loss_trades) if loss_trades else 0,
            'total_pnl': sum(pnls),
            'sharpe_ratio': self._calculate_sharpe_ratio(pnls),
            'max_drawdown': self._calculate_max_drawdown(pnls)
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, pnls: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from PnL list."""
        if not pnls:
            return 0
        returns = pd.Series(pnls)
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from PnL list."""
        if not pnls:
            return 0
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(min(drawdown)) if drawdown.size > 0 else 0 