import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from enum import Enum

class MarketCondition(Enum):
    """Market condition types"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    STABLE = "stable"

class StrategyAI:
    """AI-powered strategy selector"""
    
    def __init__(self):
        self.current_strategy = None
        self.market_condition = None
        self.strategies = {
            MarketCondition.TRENDING: {
                "name": "trend_following",
                "min_score": 0.6
            },
            MarketCondition.RANGING: {
                "name": "mean_reversion",
                "min_score": 0.6
            },
            MarketCondition.VOLATILE: {
                "name": "breakout",
                "min_score": 0.7
            },
            MarketCondition.STABLE: {
                "name": "grid_trading",
                "min_score": 0.5
            }
        }
        
    def analyze_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        """Analyze market condition using various indicators"""
        # Calculate ADX for trend strength
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        pos_dm = up_move.where(up_move > down_move, 0)
        neg_dm = down_move.where(down_move > up_move, 0)
        
        pos_di = 100 * (pos_dm.rolling(window=14).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=14).mean() / atr)
        
        # ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=14).mean()
        
        # Volatility (using ATR)
        volatility = atr / close * 100
        
        # RSI for ranging detection
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Get latest values
        latest_adx = adx.iloc[-1]
        latest_volatility = volatility.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        
        # Determine market condition
        if latest_adx > 25:  # Strong trend
            self.market_condition = MarketCondition.TRENDING
        elif latest_volatility > 2.0:  # High volatility
            self.market_condition = MarketCondition.VOLATILE
        elif 40 < latest_rsi < 60:  # Range-bound
            self.market_condition = MarketCondition.RANGING
        else:
            self.market_condition = MarketCondition.STABLE
            
        return self.market_condition
        
    def select_best_strategy(self, data: pd.DataFrame) -> Optional[str]:
        """Select best strategy based on market condition and performance metrics"""
        if self.market_condition is None:
            self.analyze_market_condition(data)
            
        strategy_info = self.strategies.get(self.market_condition)
        if strategy_info and self._evaluate_strategy_fit(data, strategy_info["min_score"]):
            self.current_strategy = strategy_info["name"]
            return self.current_strategy
            
        return None
        
    def _evaluate_strategy_fit(self, data: pd.DataFrame, min_score: float) -> bool:
        """Evaluate if the strategy fits the current market condition"""
        # This is a simplified evaluation
        # In a real implementation, this would use more sophisticated metrics
        if self.market_condition == MarketCondition.TRENDING:
            # Check trend strength
            close = data['close']
            sma = close.rolling(window=20).mean()
            trend_score = abs(close.iloc[-1] - sma.iloc[-1]) / sma.iloc[-1]
            return trend_score > min_score
            
        elif self.market_condition == MarketCondition.RANGING:
            # Check if price is moving sideways
            close = data['close']
            upper = close.rolling(window=20).max()
            lower = close.rolling(window=20).min()
            range_score = 1 - (upper.iloc[-1] - lower.iloc[-1]) / lower.iloc[-1]
            return range_score > min_score
            
        elif self.market_condition == MarketCondition.VOLATILE:
            # Check volatility
            close = data['close']
            returns = close.pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            return volatility > min_score
            
        return True  # Default to True for STABLE condition
        
    def get_current_strategy(self) -> Optional[str]:
        """Get the currently selected strategy"""
        return self.current_strategy
        
    def get_market_condition(self) -> Optional[MarketCondition]:
        """Get the current market condition"""
        return self.market_condition 