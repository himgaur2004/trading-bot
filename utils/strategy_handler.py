from enum import Enum
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class StrategyType(Enum):
    """Available trading strategy types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    GRID_TRADING = "grid_trading"
    MOMENTUM = "momentum"

class StrategyHandler:
    """Handler for managing and executing trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.current_config = self.get_base_config()
        
    def get_base_config(self) -> Dict:
        """Get base configuration for strategies"""
        return {
            "buy_conditions": {
                "min_volume": 100000,
                "use_ema": True,
                "ema_fast": 9,
                "ema_slow": 21
            },
            "sell_conditions": {
                "take_profit": 0.03,
                "stop_loss": 0.02,
                "trailing_stop": True,
                "trailing_percentage": 0.01
            },
            "risk_settings": {
                "position_size": 0.1,
                "max_open_trades": 5
            },
            "strategy_settings": {
                "strategy_type": "trend_following",
                "indicators": ["ema", "macd", "rsi"]
            }
        }
        
    def add_strategy(self, name: str, parameters: Dict):
        """Add a new strategy with parameters"""
        self.strategies[name] = parameters
        
    def update_config(self, config: Dict):
        """Update current configuration"""
        self.current_config.update(config)
        
    def get_current_config(self) -> Dict:
        """Get current configuration"""
        return self.current_config
        
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on current strategy"""
        signals = data.copy()
        strategy_type = self.current_config["strategy_settings"]["strategy_type"]
        
        if strategy_type == StrategyType.TREND_FOLLOWING.value:
            signals = self._trend_following_signals(signals)
        elif strategy_type == StrategyType.MEAN_REVERSION.value:
            signals = self._mean_reversion_signals(signals)
        elif strategy_type == StrategyType.BREAKOUT.value:
            signals = self._breakout_signals(signals)
        elif strategy_type == StrategyType.GRID_TRADING.value:
            signals = self._grid_trading_signals(signals)
        elif strategy_type == StrategyType.MOMENTUM.value:
            signals = self._momentum_signals(signals)
            
        return signals
        
    def get_performance(self, signals: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics"""
        if 'trade_type' not in signals.columns:
            return {
                'total_return': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
            
        trades = signals[signals['trade_type'].notna()].copy()
        
        # Calculate metrics
        total_return = trades['profit_pct'].sum()
        winning_trades = trades[trades['profit'] > 0]
        losing_trades = trades[trades['profit'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
        profit_factor = (
            abs(winning_trades['profit'].sum()) / abs(losing_trades['profit'].sum())
            if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0
            else 0
        )
        
        # Calculate max drawdown
        cumulative_returns = (1 + trades['profit_pct']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) * 100 if not drawdowns.empty else 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown
        }
        
    def _trend_following_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for trend following strategy"""
        config = self.current_config["buy_conditions"]
        
        # Calculate EMAs
        data['ema_fast'] = data['close'].ewm(span=config['ema_fast']).mean()
        data['ema_slow'] = data['close'].ewm(span=config['ema_slow']).mean()
        
        # Generate signals
        data['buy'] = (data['ema_fast'] > data['ema_slow']) & (data['volume'] >= config['min_volume'])
        data['sell'] = data['ema_fast'] < data['ema_slow']
        
        return data
        
    def _mean_reversion_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for mean reversion strategy"""
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        data['buy'] = data['rsi'] < 30
        data['sell'] = data['rsi'] > 70
        
        return data
        
    def _breakout_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for breakout strategy"""
        period = 20
        
        # Calculate Donchian Channel
        data['upper'] = data['high'].rolling(period).max()
        data['lower'] = data['low'].rolling(period).min()
        
        # Generate signals
        data['buy'] = data['close'] > data['upper'].shift(1)
        data['sell'] = data['close'] < data['lower'].shift(1)
        
        return data
        
    def _grid_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for grid trading strategy"""
        config = self.current_config["buy_conditions"]
        grid_levels = 5
        grid_spacing = 0.02  # 2%
        
        # Calculate grid levels
        price_range = data['close'].max() - data['close'].min()
        grid_size = price_range / grid_levels
        
        # Generate signals
        data['buy'] = data['close'] <= data['close'].shift(1) * (1 - grid_spacing)
        data['sell'] = data['close'] >= data['close'].shift(1) * (1 + grid_spacing)
        
        return data
        
    def _momentum_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for momentum strategy"""
        # Calculate MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Generate signals
        data['buy'] = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        data['sell'] = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        
        return data 