import os
import time
import pandas as pd
from typing import Dict, List
from datetime import datetime
import ccxt
from loguru import logger
from config import Config
from strategies.grid_trading_strategy import GridTradingStrategy
from strategies.arbitrage_strategy import ArbitrageStrategy
from strategies.sentiment_strategy import SentimentStrategy
from strategies.order_flow_strategy import OrderFlowStrategy
from strategies.market_making_strategy import MarketMakingStrategy

class TradingBot:
    def __init__(self, config: Config):
        """
        Initialize trading bot with configuration.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        self.setup_logging()
        self.initialize_exchange()
        self.initialize_strategies()
        self.active_positions: Dict[str, Dict] = {}
        
    def setup_logging(self):
        """Configure logging settings."""
        logger.add(
            self.config.logging.log_file,
            level=self.config.logging.log_level,
            rotation="1 day"
        )
        
    def initialize_exchange(self):
        """Initialize exchange connection."""
        exchange_class = getattr(ccxt, self.config.exchange.name)
        self.exchange = exchange_class({
            'apiKey': os.getenv('EXCHANGE_API_KEY'),
            'secret': os.getenv('EXCHANGE_API_SECRET'),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} if self.config.exchange.name == 'binance' else {}
        })
        
        if self.config.exchange.testnet:
            self.exchange.set_sandbox_mode(True)
            
    def initialize_strategies(self):
        """Initialize trading strategies."""
        self.strategies = {
            'grid_trading': GridTradingStrategy(),
            'arbitrage': ArbitrageStrategy(),
            'sentiment': SentimentStrategy(),
            'order_flow': OrderFlowStrategy(),
            'market_making': MarketMakingStrategy()
        }
        
    async def fetch_market_data(self, symbol: str) -> Dict:
        """
        Fetch market data for a trading pair.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary containing market data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=100)
            orderbook = self.exchange.fetch_order_book(symbol)
            trades = self.exchange.fetch_trades(symbol, limit=100)
            
            # Format data for strategies
            df_market = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            if not df_market.empty:
                if 'time' in df_market.columns:
                    df_market['timestamp'] = pd.to_datetime(df_market['time'], unit='ms')
                    df_market.set_index('timestamp', inplace=True)
                elif 'timestamp' in df_market.columns:
                    df_market['timestamp'] = pd.to_datetime(df_market['timestamp'], unit='ms')
                    df_market.set_index('timestamp', inplace=True)
            market_data = {
                'market': df_market,
                'order_book': orderbook,
                'trades': trades
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
            
    def calculate_position_size(self, symbol: str, side: str) -> float:
        """
        Calculate position size based on strategy signals.
        
        Args:
            symbol: Trading pair symbol
            side: Trade side ('buy' or 'sell')
            
        Returns:
            Position size in base currency
        """
        try:
            balance = self.exchange.fetch_balance()
            account_value = balance['total']['USDT']
            current_price = self.exchange.fetch_ticker(symbol)['last']
            
            # Get position sizes from each strategy
            strategy_sizes = {}
            for name, strategy in self.strategies.items():
                size = strategy.calculate_position_size(
                    account_value,
                    current_price,
                    self.calculate_volatility(symbol)
                )
                strategy_sizes[name] = size * self.config.trading.strategy_weights[name]
                
            # Combine position sizes
            total_size = sum(strategy_sizes.values())
            
            # Apply risk limits
            max_size = account_value * self.config.trading.max_position_size / current_price
            position_size = min(total_size, max_size)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
            
    def calculate_volatility(self, symbol: str) -> float:
        """Calculate current market volatility."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=24)
            prices = pd.DataFrame(ohlcv)
            returns = prices[4].pct_change()
            return returns.std()
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.01
            
    async def execute_trade(self, symbol: str, side: str, size: float) -> Dict:
        """
        Execute trade on exchange.
        
        Args:
            symbol: Trading pair symbol
            side: Trade side ('buy' or 'sell')
            size: Position size
            
        Returns:
            Trade execution details
        """
        try:
            order = self.exchange.create_order(
                symbol,
                'market',
                side,
                size,
                params={}
            )
            
            logger.info(f"Executed {side} order: {order}")
            return order
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
            
    async def manage_positions(self, symbol: str, market_data: Dict):
        """
        Manage open positions and update stops.
        
        Args:
            symbol: Trading pair symbol
            market_data: Current market data
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            
            for position in positions:
                if position['size'] > 0:
                    # Update position data
                    position_data = {
                        'symbol': symbol,
                        'side': 'buy' if position['side'] == 'long' else 'sell',
                        'size': position['size'],
                        'entry_price': position['entryPrice'],
                        'current_price': position['markPrice'],
                        'pnl': position['unrealizedPnl']
                    }
                    
                    # Get stop updates from strategies
                    stop_updates = {}
                    for strategy in self.strategies.values():
                        updates = strategy.should_update_stops(
                            position_data['current_price'],
                            position_data
                        )
                        stop_updates.update(updates)
                        
                    # Update stops if needed
                    if stop_updates:
                        self.update_position_stops(position['id'], stop_updates)
                        
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            
    def update_position_stops(self, position_id: str, stop_updates: Dict):
        """Update position stop losses and take profits."""
        try:
            if 'stop_loss' in stop_updates:
                self.exchange.create_order(
                    position_id,
                    'stop',
                    'sell',
                    None,
                    stop_updates['stop_loss'],
                    {'stopLoss': True}
                )
                
            if 'take_profit' in stop_updates:
                self.exchange.create_order(
                    position_id,
                    'limit',
                    'sell',
                    None,
                    stop_updates['take_profit'],
                    {'takeProfit': True}
                )
                
        except Exception as e:
            logger.error(f"Error updating stops: {e}")
            
    async def run_once(self):
        """Execute a single trading cycle."""
        try:
            for symbol in self.config.trading.trading_pairs:
                # Fetch market data
                market_data = await self.fetch_market_data(symbol)
                if not market_data:
                    continue
                    
                # Generate signals from strategies
                signals = {}
                for name, strategy in self.strategies.items():
                    signal_df = strategy.generate_signals(market_data)
                    signals[name] = {
                        'signal': signal_df['signal'].iloc[-1],
                        'strength': signal_df['strength'].iloc[-1]
                    }
                    
                # Combine signals using strategy weights
                combined_signal = 0
                for name, signal in signals.items():
                    weight = self.config.trading.strategy_weights.get(name, 0)
                    combined_signal += signal['signal'] * signal['strength'] * weight
                    
                # Execute trades based on combined signal
                if abs(combined_signal) > self.config.trading.signal_threshold:
                    side = 'buy' if combined_signal > 0 else 'sell'
                    size = self.calculate_position_size(symbol, side)
                    
                    if size > 0:
                        await self.execute_trade(symbol, side, size)
                        
                # Manage existing positions
                await self.manage_positions(symbol, market_data)
                
            return True
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return False

    async def run(self):
        """Main bot execution loop."""
        logger.info("Starting trading bot...")
        
        while True:
            try:
                success = await self.run_once()
                if not success:
                    await asyncio.sleep(60)  # Sleep on error before retrying
                else:
                    await asyncio.sleep(self.config.trading.update_interval)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Sleep on error before retrying 