from typing import Dict, List, Optional
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class PoolConfig:
    """Configuration for a specific market condition or asset"""
    buy_conditions: Dict
    sell_conditions: Dict
    risk_settings: Dict
    strategy_settings: Dict
    
class ConfigPool:
    """Manager for different trading configurations"""
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
        self.pools: Dict[str, PoolConfig] = {}
        
    def add_pool(self, name: str, config: PoolConfig):
        """Add a new configuration pool"""
        self.pools[name] = config
        
    def get_config(self, pool_name: str = None) -> Dict:
        """Get configuration for specified pool or base config if not found"""
        if not pool_name or pool_name not in self.pools:
            return deepcopy(self.base_config)
            
        # Start with base config
        config = deepcopy(self.base_config)
        pool_config = self.pools[pool_name]
        
        # Update with pool-specific settings
        config.update({
            "buy_conditions": pool_config.buy_conditions,
            "sell_conditions": pool_config.sell_conditions,
            "risk_settings": pool_config.risk_settings,
            "strategy_settings": pool_config.strategy_settings
        })
        
        return config
        
class MarketConfigPool:
    """Manager for different trading configurations"""
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
        self.pools: Dict[str, PoolConfig] = {}
        
        # Initialize default pools
        self._init_default_pools()
        
    def _init_default_pools(self):
        """Initialize default configuration pools for different market conditions"""
        # Trending market configuration
        self.add_pool("trending", PoolConfig(
            buy_conditions={
                "min_volume": 100000,
                "use_ema": True,
                "ema_fast": 9,
                "ema_slow": 21
            },
            sell_conditions={
                "take_profit": 0.03,
                "stop_loss": 0.02,
                "trailing_stop": True,
                "trailing_percentage": 0.01
            },
            risk_settings={
                "position_size": 0.1,
                "max_open_trades": 5
            },
            strategy_settings={
                "strategy_type": "trend_following",
                "indicators": ["ema", "macd", "rsi"]
            }
        ))
        
        # Ranging market configuration
        self.add_pool("ranging", PoolConfig(
            buy_conditions={
                "min_volume": 50000,
                "use_rsi": True,
                "rsi_period": 14,
                "rsi_oversold": 30
            },
            sell_conditions={
                "take_profit": 0.02,
                "stop_loss": 0.01,
                "trailing_stop": False,
                "rsi_overbought": 70
            },
            risk_settings={
                "position_size": 0.05,
                "max_open_trades": 3
            },
            strategy_settings={
                "strategy_type": "mean_reversion",
                "indicators": ["rsi", "bollinger_bands"]
            }
        ))
        
        # Volatile market configuration
        self.add_pool("volatile", PoolConfig(
            buy_conditions={
                "min_volume": 200000,
                "use_breakout": True,
                "breakout_period": 20,
                "volume_factor": 2.0
            },
            sell_conditions={
                "take_profit": 0.04,
                "stop_loss": 0.02,
                "trailing_stop": True,
                "trailing_percentage": 0.015
            },
            risk_settings={
                "position_size": 0.03,
                "max_open_trades": 2
            },
            strategy_settings={
                "strategy_type": "breakout",
                "indicators": ["atr", "volume"]
            }
        ))
        
    def add_pool(self, name: str, config: PoolConfig):
        """Add a new configuration pool"""
        self.pools[name] = config
        
    def get_config(self, pool_name: str = None) -> Dict:
        """Get configuration for specified pool or base config"""
        if pool_name is None or pool_name not in self.pools:
            return self.base_config
            
        pool_config = self.pools[pool_name]
        return {
            "buy_conditions": pool_config.buy_conditions,
            "sell_conditions": pool_config.sell_conditions,
            "risk_settings": pool_config.risk_settings,
            "strategy_settings": pool_config.strategy_settings
        }
        
    def update_pool(self, name: str, config: PoolConfig):
        """Update an existing configuration pool"""
        if name in self.pools:
            self.pools[name] = config
            
    def remove_pool(self, name: str):
        """Remove a configuration pool"""
        if name in self.pools:
            del self.pools[name]
            
    def get_pool_names(self) -> List[str]:
        """Get list of available pool names"""
        return list(self.pools.keys()) 