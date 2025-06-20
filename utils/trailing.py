from typing import Dict, Optional
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class TrailingConfig:
    """Configuration for trailing orders"""
    activation_percentage: float  # Percentage from entry to activate trailing
    callback_rate: float  # How far price can move against position before triggering
    step_size: float  # Minimum price movement increment
    arm_price: Optional[float] = None  # Price at which trailing becomes active

class TrailingOrderManager:
    """Manager for trailing order types (stop-loss, take-profit, buy)"""
    
    def __init__(self, config: TrailingConfig):
        self.config = config
        self.highest_price = None
        self.lowest_price = None
        self.trailing_price = None
        self.is_active = False
        
    def update_trailing_stop(self, current_price: float, position_type: str = "long") -> Optional[float]:
        """Update trailing stop price based on current market price"""
        current_price = Decimal(str(current_price))
        
        if not self.is_active:
            if self.config.arm_price is None:
                self.is_active = True
            else:
                # Check if activation price is reached
                if position_type == "long" and current_price >= Decimal(str(self.config.arm_price)):
                    self.is_active = True
                elif position_type == "short" and current_price <= Decimal(str(self.config.arm_price)):
                    self.is_active = True
                else:
                    return None
                    
        if position_type == "long":
            # Update highest seen price
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                # Calculate new trailing stop
                callback = self.highest_price * (1 - Decimal(str(self.config.callback_rate)))
                # Round to step size
                step_size = Decimal(str(self.config.step_size))
                self.trailing_price = (callback / step_size).quantize(Decimal('1')) * step_size
                
        else:  # Short position
            # Update lowest seen price
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
                # Calculate new trailing stop
                callback = self.lowest_price * (1 + Decimal(str(self.config.callback_rate)))
                # Round to step size
                step_size = Decimal(str(self.config.step_size))
                self.trailing_price = (callback / step_size).quantize(Decimal('1')) * step_size
                
        return float(self.trailing_price) if self.trailing_price is not None else None
        
    def check_stop_triggered(self, current_price: float, position_type: str = "long") -> bool:
        """Check if trailing stop is triggered"""
        if not self.is_active or self.trailing_price is None:
            return False
            
        current_price = Decimal(str(current_price))
        
        if position_type == "long":
            return current_price <= self.trailing_price
        else:  # Short position
            return current_price >= self.trailing_price
            
    def reset(self):
        """Reset trailing stop"""
        self.highest_price = None
        self.lowest_price = None
        self.trailing_price = None
        self.is_active = False
        
    def get_current_stop(self) -> Optional[float]:
        """Get current trailing stop price"""
        return float(self.trailing_price) if self.trailing_price is not None else None
        
    def update_trailing_take_profit(self, current_price: float,
                                  entry_price: float) -> Optional[float]:
        """Update trailing take-profit price
        
        Similar to stop-loss but triggers when price falls below trailing level
        after reaching take-profit target
        """
        if self.highest_price is None:
            self.highest_price = entry_price
            
        if current_price > self.highest_price:
            self.highest_price = current_price
            
        if not self.is_active:
            activation_price = entry_price * (1 + self.config.activation_percentage)
            if current_price >= activation_price:
                self.is_active = True
                self.trailing_price = current_price * (1 - self.config.callback_rate)
                
        if self.is_active:
            new_profit = current_price * (1 - self.config.callback_rate)
            if new_profit > self.trailing_price:
                self.trailing_price = new_profit
                
            if current_price <= self.trailing_price:
                return current_price
                
        return None
        
    def update_trailing_buy(self, current_price: float, 
                          target_price: float) -> Optional[float]:
        """Update trailing buy price
        
        Trails price down to get better entry when market is falling
        """
        if self.lowest_price is None:
            self.lowest_price = current_price
            
        if current_price < self.lowest_price:
            self.lowest_price = current_price
            
        if not self.is_active:
            if self.config.arm_price and current_price <= self.config.arm_price:
                self.is_active = True
                self.trailing_price = current_price * (1 + self.config.callback_rate)
            elif current_price <= target_price:
                self.is_active = True
                self.trailing_price = current_price * (1 + self.config.callback_rate)
                
        if self.is_active:
            new_buy = current_price * (1 + self.config.callback_rate)
            if new_buy < self.trailing_price:
                self.trailing_price = new_buy
                
            if current_price >= self.trailing_price:
                return current_price
                
        return None 