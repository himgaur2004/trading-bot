from typing import Dict, List, Optional, Type, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from backend.core.strategy_base import StrategyBase
from backend.core.technical_analysis import TechnicalAnalysis, TAParameters

@dataclass
class BacktestParameters:
    """Backtesting Parameters"""
    initial_capital: float = 10000.0
    trading_fee: float = 0.001  # 0.1% per trade
    slippage: float = 0.001  # 0.1% slippage
    enable_fractional: bool = True
    leverage: float = 1.0
    margin_type: str = 'CROSSED'  # or 'ISOLATED'
    
class BacktestResult:
    def __init__(self,
                 trades: List[Dict],
                 equity_curve: pd.Series,
                 parameters: Dict):
        """
        Backtest result container.
        
        Args:
            trades: List of executed trades
            equity_curve: Portfolio value over time
            parameters: Strategy parameters used
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.parameters = parameters
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            self.metrics = {}
            return
            
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        self.metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'losing_trades': len(trades_df[trades_df['pnl'] <= 0]),
            'total_pnl': trades_df['pnl'].sum(),
            'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df),
            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0,
            'avg_loss': trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] <= 0]) > 0 else 0,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'avg_trade_duration': trades_df['duration'].mean(),
            'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) \
                           if trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
        }
        
        # Calculate returns
        returns = self.equity_curve.pct_change().dropna()
        
        # Risk metrics
        self.metrics.update({
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(self.equity_curve),
            'max_drawdown_duration': self._calculate_max_drawdown_duration(self.equity_curve),
            'volatility': returns.std() * np.sqrt(252),
            'calmar_ratio': self.metrics['total_pnl'] / abs(self._calculate_max_drawdown(self.equity_curve)) \
                          if self._calculate_max_drawdown(self.equity_curve) != 0 else float('inf')
        })
        
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = returns - risk_free_rate/252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sortino ratio."""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf')
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        rolling_max = equity.expanding().max()
        drawdowns = equity/rolling_max - 1
        return abs(drawdowns.min())
        
    def _calculate_max_drawdown_duration(self, equity: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        rolling_max = equity.expanding().max()
        drawdowns = equity/rolling_max - 1
        
        # Find drawdown periods
        is_drawdown = drawdowns < 0
        
        if not is_drawdown.any():
            return 0
            
        # Calculate durations
        drawdown_starts = is_drawdown.shift(1).fillna(False) & is_drawdown
        drawdown_ends = is_drawdown & ~is_drawdown.shift(-1).fillna(False)
        
        max_duration = 0
        current_start = None
        
        for idx in range(len(equity)):
            if drawdown_starts.iloc[idx]:
                current_start = idx
            elif drawdown_ends.iloc[idx] and current_start is not None:
                duration = idx - current_start
                max_duration = max(max_duration, duration)
                current_start = None
                
        return max_duration

class BacktestEngine:
    def __init__(self,
                 data: pd.DataFrame,
                 strategy: StrategyBase,
                 params: Optional[BacktestParameters] = None,
                 ta_params: Optional[TAParameters] = None):
        """
        Initialize backtesting engine.
        
        Args:
            data: OHLCV data
            strategy: Trading strategy instance
            params: Backtest parameters
            ta_params: Technical analysis parameters
        """
        self.data = data.copy()
        self.strategy = strategy
        self.params = params or BacktestParameters()
        self.ta = TechnicalAnalysis(ta_params)
        
        # Add technical indicators
        self.data = self.ta.add_indicators(self.data)
        
        # Initialize tracking variables
        self.current_position = None
        self.trades = []
        self.equity_curve = pd.Series(index=data.index, dtype=float)
        self.equity_curve.iloc[0] = self.params.initial_capital
        
    def run(self) -> BacktestResult:
        """
        Run backtest simulation.
        
        Returns:
            BacktestResult object with performance metrics
        """
        current_capital = self.params.initial_capital
        
        for i in range(1, len(self.data)):
            current_bar = self.data.iloc[i]
            prev_bar = self.data.iloc[i-1]
            
            # Update equity curve
            self.equity_curve.iloc[i] = current_capital
            
            # Generate trading signals
            signals = self.strategy.generate_signals(self.data.iloc[:i+1])
            current_signal = signals.iloc[-1]
            
            # Check for stop loss/take profit
            if self.current_position:
                if self._check_stop_loss(current_bar) or self._check_take_profit(current_bar):
                    current_capital = self._close_position(current_bar)
                    continue
                    
                # Check for stop updates
                stop_updates = self.strategy.should_update_stops(
                    current_bar['close'],
                    self.current_position
                )
                if stop_updates:
                    self.current_position.update(stop_updates)
            
            # Process signals
            if current_signal['signal'] != 0 and not self.current_position:
                # Calculate position size
                price = current_bar['close']
                volatility = current_bar['atr'] / price if 'atr' in current_bar else 0.02
                
                position_size = self.strategy.calculate_position_size(
                    current_capital,
                    price,
                    volatility
                )
                
                # Apply leverage
                position_size *= self.params.leverage
                
                # Open new position
                side = 'buy' if current_signal['signal'] > 0 else 'sell'
                entry_price = price * (1 + self.params.slippage) if side == 'buy' \
                            else price * (1 - self.params.slippage)
                            
                self.current_position = {
                    'side': side,
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'entry_time': current_bar.name,
                    'stop_loss': entry_price * (1 - self.strategy.stop_loss) if side == 'buy'
                                else entry_price * (1 + self.strategy.stop_loss),
                    'take_profit': entry_price * (1 + self.strategy.take_profit) if side == 'buy'
                                 else entry_price * (1 - self.strategy.take_profit)
                }
                
                # Deduct fees
                current_capital -= position_size * entry_price * self.params.trading_fee
                
            elif current_signal['signal'] != 0 and self.current_position:
                # Close existing position if signal is in opposite direction
                if (current_signal['signal'] > 0 and self.current_position['side'] == 'sell') or \
                   (current_signal['signal'] < 0 and self.current_position['side'] == 'buy'):
                    current_capital = self._close_position(current_bar)
                    
        # Close any remaining position
        if self.current_position:
            current_capital = self._close_position(self.data.iloc[-1])
            
        return BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            parameters=self.strategy.params.__dict__
        )
        
    def _close_position(self, bar: pd.Series) -> float:
        """
        Close current position and return updated capital.
        
        Args:
            bar: Current price bar
            
        Returns:
            Updated capital after closing position
        """
        exit_price = bar['close']
        position = self.current_position
        
        # Apply slippage
        exit_price *= (1 - self.params.slippage) if position['side'] == 'buy' \
                     else (1 + self.params.slippage)
                     
        # Calculate PnL
        if position['side'] == 'buy':
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['position_size']
            
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': bar.name,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'pnl': pnl,
            'pnl_percentage': pnl / (position['entry_price'] * position['position_size']),
            'duration': bar.name - position['entry_time']
        }
        self.trades.append(trade)
        
        # Update capital
        current_capital = self.equity_curve.iloc[-1]
        current_capital += pnl
        current_capital -= position['position_size'] * exit_price * self.params.trading_fee
        
        self.current_position = None
        return current_capital
        
    def _check_stop_loss(self, bar: pd.Series) -> bool:
        """Check if stop loss is hit."""
        if not self.current_position:
            return False
            
        if self.current_position['side'] == 'buy':
            return bar['low'] <= self.current_position['stop_loss']
        else:
            return bar['high'] >= self.current_position['stop_loss']
            
    def _check_take_profit(self, bar: pd.Series) -> bool:
        """Check if take profit is hit."""
        if not self.current_position:
            return False
            
        if self.current_position['side'] == 'buy':
            return bar['high'] >= self.current_position['take_profit']
        else:
            return bar['low'] <= self.current_position['take_profit'] 