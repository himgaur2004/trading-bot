from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ExchangeConfig:
    """Exchange configuration settings"""
    name: str = "binance"
    testnet: bool = True  # Use testnet by default for safety
    api_key: str = ""  # Will be loaded from environment variables
    api_secret: str = ""  # Will be loaded from environment variables
    
@dataclass
class TradingConfig:
    """Trading configuration settings"""
    # Trading pairs
    trading_pairs: list = None
    
    # Account settings
    max_position_size: float = 0.1  # Maximum 10% of portfolio per position
    base_order_size: float = 0.01  # Base order size in BTC
    
    # Risk management
    max_open_positions: int = 3
    max_daily_drawdown: float = 0.05  # Maximum 5% daily drawdown
    
    # Strategy weights for ensemble
    strategy_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.trading_pairs is None:
            self.trading_pairs = ["BTC/USDT", "ETH/USDT"]
        if self.strategy_weights is None:
            self.strategy_weights = {
                "grid_trading": 0.2,
                "arbitrage": 0.2,
                "sentiment": 0.2,
                "order_flow": 0.2,
                "market_making": 0.2
            }

@dataclass
class LogConfig:
    """Logging configuration settings"""
    log_level: str = "INFO"
    log_file: str = "trading_bot.log"
    enable_telegram: bool = False
    telegram_token: str = ""
    telegram_chat_id: str = ""

class Config:
    """Main configuration class"""
    def __init__(self):
        self.exchange = ExchangeConfig()
        self.trading = TradingConfig()
        self.logging = LogConfig()
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        config = cls()
        
        if 'exchange' in config_dict:
            for key, value in config_dict['exchange'].items():
                setattr(config.exchange, key, value)
                
        if 'trading' in config_dict:
            for key, value in config_dict['trading'].items():
                setattr(config.trading, key, value)
                
        if 'logging' in config_dict:
            for key, value in config_dict['logging'].items():
                setattr(config.logging, key, value)
                
        return config 

"""
Trading bot configuration parameters.
"""

# Risk Management Parameters
MAX_POSITION_SIZE = 0.02  # Maximum position size as percentage of account balance
MAX_LEVERAGE = 5.0        # Maximum allowed leverage
MIN_RISK_REWARD = 2.0     # Minimum risk/reward ratio for trades
MAX_DAILY_DRAWDOWN = 0.05 # Maximum allowed daily drawdown (5%)
MAX_OPEN_POSITIONS = 5    # Maximum number of concurrent open positions

# Trading Parameters
MIN_VOLUME_24H = 100000   # Minimum 24h volume in USDT
MIN_PRICE = 0.00001      # Minimum price to consider for trading
MAX_SPREAD = 0.002       # Maximum allowed spread (0.2%)
MIN_TICK_SIZE = 0.00001  # Minimum price movement

# Technical Analysis Parameters
FAST_MA = 9              # Fast moving average period
SLOW_MA = 21             # Slow moving average period
RSI_PERIOD = 14          # RSI calculation period
RSI_OVERBOUGHT = 70      # RSI overbought threshold
RSI_OVERSOLD = 30        # RSI oversold threshold
VOLUME_MA_PERIOD = 20    # Volume moving average period
MIN_VOLUME_RATIO = 1.5   # Minimum volume ratio compared to average

# Order Parameters
ORDER_TYPE = 'LIMIT'     # Default order type
POST_ONLY = True         # Use post-only orders to avoid taker fees
TIME_IN_FORCE = 'GTC'    # Good till cancelled
DEFAULT_RETRY = 3        # Number of retries for failed orders
RETRY_DELAY = 1          # Delay between retries in seconds

# API Rate Limiting
MAX_REQUESTS_PER_MIN = 60  # Maximum API requests per minute
RATE_LIMIT_BUFFER = 0.8    # Buffer to stay under rate limit (80%)

# Logging Configuration
LOG_LEVEL = 'INFO'       # Logging level
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Backtesting Parameters
BACKTEST_DAYS = 30       # Number of days to backtest
COMMISSION_RATE = 0.001  # Trading commission (0.1%)

# Exchange Specific
EXCHANGE_ID = 'coindcx'
QUOTE_CURRENCY = 'USDT'
SUPPORTED_QUOTE_CURRENCIES = ['USDT', 'USDC', 'DAI']

# Error Handling
MAX_ERRORS = 3           # Maximum consecutive errors before stopping
ERROR_COOLDOWN = 300     # Cooldown period after max errors (5 minutes)

# Performance Monitoring
PROFIT_TARGET = 0.02     # Daily profit target (2%)
STOP_LOSS = 0.05        # Daily stop loss (5%)
TRAILING_STOP = 0.01     # Trailing stop loss (1%)

# Notification Settings
ENABLE_NOTIFICATIONS = True
NOTIFICATION_EVENTS = [
    'trade_executed',
    'order_filled',
    'stop_loss_hit',
    'take_profit_hit',
    'error_occurred',
    'daily_summary'
] 