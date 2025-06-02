import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import ta
from enum import Enum

class StrategyType(Enum):
    MOVING_AVERAGE_CROSSOVER = "Moving Average Crossover"
    RSI = "RSI Strategy"
    MACD = "MACD Strategy"
    BOLLINGER_BANDS = "Bollinger Bands"
    CUSTOM = "Custom Strategy"

class TradingStrategy:
    def __init__(self, strategy_type: StrategyType, parameters: Dict):
        self.strategy_type = strategy_type
        self.parameters = parameters
        self.signals = []
        self.performance_metrics = {}

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on the strategy type."""
        if self.strategy_type == StrategyType.MOVING_AVERAGE_CROSSOVER:
            return self._moving_average_crossover(data)
        elif self.strategy_type == StrategyType.RSI:
            return self._rsi_strategy(data)
        elif self.strategy_type == StrategyType.MACD:
            return self._macd_strategy(data)
        elif self.strategy_type == StrategyType.BOLLINGER_BANDS:
            return self._bollinger_bands_strategy(data)
        else:
            return self._custom_strategy(data)

    def _moving_average_crossover(self, data: pd.DataFrame) -> pd.DataFrame:
        """Moving Average Crossover Strategy."""
        fast_period = self.parameters.get('fast_ma', 9)
        slow_period = self.parameters.get('slow_ma', 21)
        
        # Calculate moving averages
        data['fast_ma'] = ta.trend.sma_indicator(data['close'], window=fast_period)
        data['slow_ma'] = ta.trend.sma_indicator(data['close'], window=slow_period)
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1  # Buy signal
        data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1  # Sell signal
        
        return data

    def _rsi_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """RSI Strategy."""
        rsi_period = self.parameters.get('rsi_period', 14)
        overbought = self.parameters.get('overbought', 70)
        oversold = self.parameters.get('oversold', 30)
        
        # Calculate RSI
        data['rsi'] = ta.momentum.rsi(data['close'], window=rsi_period)
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['rsi'] < oversold, 'signal'] = 1  # Buy signal
        data.loc[data['rsi'] > overbought, 'signal'] = -1  # Sell signal
        
        return data

    def _macd_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """MACD Strategy."""
        fast_period = self.parameters.get('fast_period', 12)
        slow_period = self.parameters.get('slow_period', 26)
        signal_period = self.parameters.get('signal_period', 9)
        
        # Calculate MACD
        data['macd'] = ta.trend.macd(data['close'], 
                                    window_slow=slow_period,
                                    window_fast=fast_period)
        data['macd_signal'] = ta.trend.macd_signal(data['close'],
                                                  window_slow=slow_period,
                                                  window_fast=fast_period,
                                                  window_sign=signal_period)
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['macd'] > data['macd_signal'], 'signal'] = 1  # Buy signal
        data.loc[data['macd'] < data['macd_signal'], 'signal'] = -1  # Sell signal
        
        return data

    def _bollinger_bands_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands Strategy."""
        window = self.parameters.get('window', 20)
        std_dev = self.parameters.get('std_dev', 2)
        
        # Calculate Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(close=data["close"], 
                                                   window=window, 
                                                   window_dev=std_dev)
        
        data['bb_high'] = indicator_bb.bollinger_hband()
        data['bb_low'] = indicator_bb.bollinger_lband()
        data['bb_mid'] = indicator_bb.bollinger_mavg()
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['close'] < data['bb_low'], 'signal'] = 1  # Buy signal
        data.loc[data['close'] > data['bb_high'], 'signal'] = -1  # Sell signal
        
        return data

    def _custom_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Custom Strategy combining multiple indicators."""
        # Calculate RSI
        data['rsi'] = ta.momentum.rsi(data['close'], window=14)
        
        # Calculate MACD
        data['macd'] = ta.trend.macd(data['close'])
        data['macd_signal'] = ta.trend.macd_signal(data['close'])
        
        # Calculate Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(close=data["close"])
        data['bb_high'] = bb_indicator.bollinger_hband()
        data['bb_low'] = bb_indicator.bollinger_lband()
        
        # Generate signals based on multiple conditions
        data['signal'] = 0
        
        # Buy conditions (all must be true)
        buy_condition = (
            (data['rsi'] < 40) &  # RSI oversold
            (data['macd'] > data['macd_signal']) &  # MACD crossover
            (data['close'] < data['bb_low'])  # Price below lower BB
        )
        
        # Sell conditions (all must be true)
        sell_condition = (
            (data['rsi'] > 60) &  # RSI overbought
            (data['macd'] < data['macd_signal']) &  # MACD crossunder
            (data['close'] > data['bb_high'])  # Price above upper BB
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data

    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics."""
        if 'signal' not in data.columns:
            data = self.calculate_signals(data)
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['returns'] * data['signal'].shift(1)
        
        # Calculate metrics
        total_trades = len(data[data['signal'] != 0])
        winning_trades = len(data[data['strategy_returns'] > 0])
        losing_trades = len(data[data['strategy_returns'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profits = data[data['strategy_returns'] > 0]['strategy_returns'].sum()
        gross_losses = abs(data[data['strategy_returns'] < 0]['strategy_returns'].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        # Calculate max drawdown
        cumulative_returns = (1 + data['strategy_returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_return': (cumulative_returns.iloc[-1] - 1) if len(cumulative_returns) > 0 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(data['strategy_returns']),
            'sortino_ratio': self._calculate_sortino_ratio(data['strategy_returns'])
        }
        
        return self.performance_metrics

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.01) -> float:
        """Calculate the Sharpe Ratio."""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        if len(excess_returns) == 0:
            return 0
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.01) -> float:
        """Calculate the Sortino Ratio."""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0
        return np.sqrt(252) * (excess_returns.mean() / downside_returns.std()) if downside_returns.std() != 0 else 0

class StrategyHandler:
    def __init__(self):
        self.strategies = {}
        self.active_strategy = None

    def add_strategy(self, name: str, strategy_type: StrategyType, parameters: Dict):
        """Add a new strategy to the handler."""
        self.strategies[name] = TradingStrategy(strategy_type, parameters)

    def remove_strategy(self, name: str):
        """Remove a strategy from the handler."""
        if name in self.strategies:
            del self.strategies[name]

    def set_active_strategy(self, name: str):
        """Set the active trading strategy."""
        if name in self.strategies:
            self.active_strategy = self.strategies[name]
        else:
            raise ValueError(f"Strategy '{name}' not found")

    def get_strategy_signals(self, name: str, data: pd.DataFrame) -> pd.DataFrame:
        """Get trading signals for a specific strategy."""
        if name in self.strategies:
            return self.strategies[name].calculate_signals(data)
        raise ValueError(f"Strategy '{name}' not found")

    def get_strategy_performance(self, name: str, data: pd.DataFrame) -> Dict:
        """Get performance metrics for a specific strategy."""
        if name in self.strategies:
            return self.strategies[name].calculate_performance_metrics(data)
        raise ValueError(f"Strategy '{name}' not found") 