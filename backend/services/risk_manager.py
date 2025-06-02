from typing import Dict, List
from loguru import logger
import math

class RiskManager:
    def __init__(self, 
                 max_position_size: float = 0.02,  # Max 2% of account per position
                 max_leverage: float = 5.0,        # Max 5x leverage
                 min_risk_reward: float = 2.0,     # Minimum 2:1 risk/reward ratio
                 max_daily_drawdown: float = 0.05, # Max 5% daily drawdown
                 max_open_positions: int = 5):     # Max 5 open positions
        """
        Initialize risk manager with position sizing and risk parameters.
        
        Args:
            max_position_size: Maximum position size as percentage of account
            max_leverage: Maximum allowed leverage
            min_risk_reward: Minimum risk/reward ratio
            max_daily_drawdown: Maximum allowed daily drawdown
            max_open_positions: Maximum number of open positions
        """
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.min_risk_reward = min_risk_reward
        self.max_daily_drawdown = max_daily_drawdown
        self.max_open_positions = max_open_positions
        self.daily_pnl = 0
        self.open_positions = {}
        
    def calculate_position_size(self,
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              take_profit: float,
                              market_data: Dict) -> Dict:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            account_balance: Current account balance in USDT
            entry_price: Planned entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            market_data: Market data for volatility and liquidity checks
            
        Returns:
            Dictionary with position details
        """
        try:
            # Calculate risk/reward ratio
            risk_per_unit = abs(entry_price - stop_loss)
            reward_per_unit = abs(take_profit - entry_price)
            risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
            
            # Check if risk/reward meets minimum requirement
            if risk_reward_ratio < self.min_risk_reward:
                logger.warning(f"Risk/Reward ratio {risk_reward_ratio:.2f} below minimum {self.min_risk_reward}")
                return None
                
            # Calculate maximum position value based on account risk
            max_risk_amount = account_balance * self.max_position_size
            max_position_value = (max_risk_amount / risk_per_unit) * entry_price
            
            # Adjust for leverage
            max_leveraged_value = max_position_value * self.max_leverage
            
            # Check market liquidity
            if market_data and 'volume' in market_data:
                avg_daily_volume = float(market_data['volume'])
                max_volume_based = avg_daily_volume * 0.01  # Max 1% of daily volume
                max_position_value = min(max_leveraged_value, max_volume_based)
            
            # Calculate final position size in base currency
            position_size = max_position_value / entry_price
            
            # Round to appropriate precision
            position_size = math.floor(position_size * 1e8) / 1e8
            
            logger.info(f"\nPosition Size Calculation:")
            logger.info(f"  • Account Balance: {account_balance:.2f} USDT")
            logger.info(f"  • Risk Amount: {max_risk_amount:.2f} USDT")
            logger.info(f"  • Position Value: {max_position_value:.2f} USDT")
            logger.info(f"  • Position Size: {position_size:.8f}")
            logger.info(f"  • Risk/Reward: {risk_reward_ratio:.2f}")
            logger.info(f"  • Leverage: {self.max_leverage}x")
            
            return {
                'size': position_size,
                'value': max_position_value,
                'leverage': self.max_leverage,
                'risk_reward': risk_reward_ratio,
                'risk_amount': max_risk_amount
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None
            
    def validate_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Validate trading signals against risk parameters.
        
        Args:
            signals: List of trading signals
            
        Returns:
            List of validated signals
        """
        validated_signals = []
        
        try:
            # Check if we can take more positions
            if len(self.open_positions) >= self.max_open_positions:
                logger.warning(f"Maximum open positions ({self.max_open_positions}) reached")
                return validated_signals
                
            # Check daily drawdown
            if abs(self.daily_pnl) >= self.max_daily_drawdown:
                logger.warning(f"Daily drawdown limit ({self.max_daily_drawdown*100}%) reached")
                return validated_signals
                
            for signal in signals:
                # Validate signal has required risk parameters
                if not all(k in signal for k in ['entry_price', 'stop_loss', 'take_profit']):
                    logger.warning(f"Signal missing required risk parameters")
                    continue
                    
                # Calculate position size
                position = self.calculate_position_size(
                    account_balance=signal.get('account_balance', 0),
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    market_data=signal.get('market_data', {})
                )
                
                if position:
                    signal['position'] = position
                    validated_signals.append(signal)
                    
            return validated_signals
            
        except Exception as e:
            logger.error(f"Error validating signals: {e}")
            return []
            
    def update_position(self, symbol: str, pnl: float, is_closed: bool = False):
        """Update position tracking and daily PnL."""
        if is_closed:
            if symbol in self.open_positions:
                del self.open_positions[symbol]
        else:
            self.open_positions[symbol] = pnl
            
        self.daily_pnl += pnl 