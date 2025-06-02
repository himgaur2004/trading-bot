from typing import Dict, Optional
import pandas as pd
from backend.utils.helpers import (
    calculate_volatility,
    calculate_atr,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)

class BaseStrategy:
    def __init__(self,
                 name: str,
                 max_position_size: float = 0.1,
                 **kwargs):
        """
        Base class for all trading strategies.
        
        Args:
            name: Strategy name
            max_position_size: Maximum position size as fraction of account
            **kwargs: Additional strategy parameters
        """
        self.name = name
        self.max_position_size = max_position_size
        self.current_signal = 0
        self.current_signal_strength = 0.0
        
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            DataFrame with signal and signal strength
        """
        raise NotImplementedError("Subclass must implement generate_signals")
        
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Market volatility
            
        Returns:
            Position size in base currency
        """
        # Base position size
        base_size = account_balance * self.max_position_size
        
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility * 2)
        adjusted_size = base_size * vol_factor
        
        # Convert to asset amount
        position_size = adjusted_size / current_price
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Check if stop levels should be updated.
        
        Args:
            current_price: Current asset price
            position_data: Position information
            
        Returns:
            Dictionary with updated stop levels
        """
        return {}  # Default implementation returns no updates
        
    def calculate_technical_indicators(self,
                                    df: pd.DataFrame,
                                    **kwargs) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators.
        
        Args:
            df: OHLCV DataFrame
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of technical indicators
        """
        indicators = {}
        
        # Volatility indicators
        indicators['volatility'] = calculate_volatility(df['close'])
        indicators['atr'] = calculate_atr(df)
        
        # Momentum indicators
        indicators['rsi'] = calculate_rsi(df['close'])
        
        # Trend indicators
        macd_data = calculate_macd(df['close'])
        indicators.update(macd_data)
        
        # Volatility bands
        bb_data = calculate_bollinger_bands(df['close'])
        indicators.update(bb_data)
        
        return indicators
        
    def validate_signal(self,
                       signal: float,
                       strength: float,
                       indicators: Dict[str, pd.Series]) -> bool:
        """
        Validate trading signal with confirmation indicators.
        
        Args:
            signal: Trading signal (-1 to 1)
            strength: Signal strength (0 to 1)
            indicators: Technical indicators
            
        Returns:
            True if signal is valid
        """
        if abs(signal) < 0.1:  # Minimum signal threshold
            return False
            
        # RSI confirmation
        rsi = indicators['rsi'].iloc[-1]
        if signal > 0 and rsi > 70:  # Overbought
            return False
        if signal < 0 and rsi < 30:  # Oversold
            return False
            
        # MACD confirmation
        macd = indicators['macd'].iloc[-1]
        macd_signal = indicators['signal'].iloc[-1]
        if signal > 0 and macd < macd_signal:
            return False
        if signal < 0 and macd > macd_signal:
            return False
            
        # Bollinger Bands confirmation
        price = indicators['close'].iloc[-1]
        upper_band = indicators['upper'].iloc[-1]
        lower_band = indicators['lower'].iloc[-1]
        
        if signal > 0 and price > upper_band:
            return False
        if signal < 0 and price < lower_band:
            return False
            
        return True
        
    def calculate_risk_metrics(self,
                             position_data: Dict,
                             market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate position risk metrics.
        
        Args:
            position_data: Position information
            market_data: Market data dictionary
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Current price
        current_price = market_data['ticker']['last']
        entry_price = position_data['entry_price']
        
        # Calculate unrealized PnL
        if position_data['side'] == 'buy':
            pnl = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) / entry_price
            
        metrics['unrealized_pnl'] = pnl
        
        # Calculate risk metrics
        df = market_data['ohlcv']
        volatility = calculate_volatility(df['close'])
        atr = calculate_atr(df).iloc[-1]
        
        metrics['volatility'] = volatility
        metrics['atr'] = atr
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = pnl / volatility if volatility > 0 else 0
        metrics['risk_reward_ratio'] = abs(pnl) / (atr / current_price)
        
        return metrics 