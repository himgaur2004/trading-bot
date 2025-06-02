import asyncio
import logging
from typing import Dict, List, Optional, Type
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import numpy as np

from backend.core.strategy_base import StrategyBase
from backend.core.technical_analysis import TechnicalAnalysis, TAParameters
from backend.core.exchange import CoinDCXExchange, ExchangeConfig
from backend.core.risk_manager import RiskManager, RiskParameters
from utils.notifications import NotificationManager, NotificationConfig

@dataclass
class TradingEngineConfig:
    """Trading Engine Configuration"""
    symbols: List[str]
    timeframe: str = "1h"
    mode: str = "live"  # live or paper
    max_positions: int = 3
    update_interval: int = 60  # seconds
    risk_free_rate: float = 0.02
    
class TradingEngine:
    def __init__(self,
                 config: TradingEngineConfig,
                 strategy: StrategyBase,
                 exchange_config: Optional[ExchangeConfig] = None,
                 risk_config: Optional[RiskParameters] = None,
                 notification_config: Optional[NotificationConfig] = None):
        """
        Initialize trading engine.
        
        Args:
            config: Trading engine configuration
            strategy: Trading strategy instance
            exchange_config: Exchange configuration
            risk_config: Risk management configuration
            notification_config: Notification configuration
        """
        self.config = config
        self.strategy = strategy
        
        # Initialize components
        self.exchange = CoinDCXExchange(exchange_config)
        self.risk_manager = RiskManager(risk_config)
        self.notification_manager = NotificationManager(notification_config)
        self.ta = TechnicalAnalysis()
        
        # Initialize state
        self.is_running = False
        self.positions: Dict[str, Dict] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.performance_metrics: Dict[str, float] = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Start the trading engine."""
        self.is_running = True
        self.logger.info("Starting trading engine...")
        
        try:
            # Initialize market data
            await self._initialize_market_data()
            
            # Start main loop
            while self.is_running:
                await self._trading_loop()
                await asyncio.sleep(self.config.update_interval)
                
        except Exception as e:
            self.logger.error(f"Trading engine error: {str(e)}")
            await self.notification_manager.send_telegram_message(
                f"Trading engine error: {str(e)}",
                level="ERROR"
            )
            raise
            
    async def stop(self):
        """Stop the trading engine."""
        self.is_running = False
        self.logger.info("Stopping trading engine...")
        
        # Close all positions
        for symbol in self.positions:
            await self._close_position(symbol, "Trading engine stopped")
            
    async def _trading_loop(self):
        """Main trading loop."""
        try:
            # Update market data
            await self._update_market_data()
            
            # Process each symbol
            for symbol in self.config.symbols:
                # Skip if symbol data not available
                if symbol not in self.market_data:
                    continue
                    
                data = self.market_data[symbol]
                
                # Generate signals
                signals = self.strategy.generate_signals(data)
                current_signal = signals.iloc[-1]
                
                # Get market condition
                market_condition = self.ta.get_market_condition(data)
                
                # Update strategy parameters
                self.strategy.update_parameters(
                    market_condition,
                    data['atr'].iloc[-1] / data['close'].iloc[-1]
                )
                
                # Process signals
                await self._process_signal(symbol, current_signal, data)
                
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in trading loop: {str(e)}")
            await self.notification_manager.send_telegram_message(
                f"Trading loop error: {str(e)}",
                level="ERROR"
            )
            
    async def _initialize_market_data(self):
        """Initialize market data for all symbols."""
        for symbol in self.config.symbols:
            # Fetch historical data
            data = self.exchange.get_klines(
                symbol=symbol,
                interval=self.config.timeframe,
                limit=500
            )
            
            # Add technical indicators
            data = self.ta.add_indicators(data)
            
            self.market_data[symbol] = data
            
    async def _update_market_data(self):
        """Update market data with latest candles."""
        for symbol in self.config.symbols:
            # Get latest candle
            new_data = self.exchange.get_klines(
                symbol=symbol,
                interval=self.config.timeframe,
                limit=2
            )
            
            if len(new_data) > 0:
                # Update existing data
                self.market_data[symbol].loc[new_data.index] = new_data
                
                # Update indicators
                self.market_data[symbol] = self.ta.add_indicators(
                    self.market_data[symbol]
                )
                
    async def _process_signal(self,
                            symbol: str,
                            signal: pd.Series,
                            data: pd.DataFrame):
        """
        Process trading signal for a symbol.
        
        Args:
            symbol: Trading symbol
            signal: Current signal
            data: Market data
        """
        current_position = self.positions.get(symbol)
        
        # Check existing position
        if current_position:
            # Check for exit signals
            if (signal['signal'] < 0 and current_position['side'] == 'buy') or \
               (signal['signal'] > 0 and current_position['side'] == 'sell'):
                await self._close_position(symbol, "Signal exit")
                return
                
            # Check stop loss/take profit
            current_price = data['close'].iloc[-1]
            if current_position['side'] == 'buy':
                if current_price <= current_position['stop_loss']:
                    await self._close_position(symbol, "Stop loss")
                    return
                if current_price >= current_position['take_profit']:
                    await self._close_position(symbol, "Take profit")
                    return
            else:
                if current_price >= current_position['stop_loss']:
                    await self._close_position(symbol, "Stop loss")
                    return
                if current_price <= current_position['take_profit']:
                    await self._close_position(symbol, "Take profit")
                    return
                    
            # Update stops if needed
            stop_updates = self.strategy.should_update_stops(
                current_price,
                current_position
            )
            if stop_updates:
                self.positions[symbol].update(stop_updates)
                
        # Check for entry signals
        elif signal['signal'] != 0:
            # Check if we can open new position
            if len(self.positions) >= self.config.max_positions:
                return
                
            # Check risk parameters
            account_balance = await self._get_account_balance()
            if not self.risk_manager.can_open_position(
                account_balance,
                symbol,
                data
            ):
                return
                
            # Calculate position size
            current_price = data['close'].iloc[-1]
            volatility = data['atr'].iloc[-1] / current_price
            
            position_size = self.strategy.calculate_position_size(
                account_balance,
                current_price,
                volatility
            )
            
            # Open position
            await self._open_position(
                symbol,
                'buy' if signal['signal'] > 0 else 'sell',
                position_size,
                current_price
            )
            
    async def _open_position(self,
                           symbol: str,
                           side: str,
                           size: float,
                           price: float):
        """
        Open new position.
        
        Args:
            symbol: Trading symbol
            side: Trade side (buy/sell)
            size: Position size
            price: Entry price
        """
        try:
            # Create order
            if self.config.mode == 'live':
                order = self.exchange.create_order(
                    symbol=symbol,
                    side=side,
                    order_type='MARKET',
                    quantity=size
                )
                price = float(order['price'])
                
            # Record position
            position = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': price,
                'entry_time': datetime.now(),
                'stop_loss': price * (1 - self.strategy.stop_loss) if side == 'buy'
                            else price * (1 + self.strategy.stop_loss),
                'take_profit': price * (1 + self.strategy.take_profit) if side == 'buy'
                             else price * (1 - self.strategy.take_profit)
            }
            
            self.positions[symbol] = position
            
            # Send notifications
            await self.notification_manager.send_telegram_message(
                f"Opened {side} position in {symbol}\n" \
                f"Price: {price:.8f}\n" \
                f"Size: {size:.8f}",
                level="TRADE"
            )
            
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            await self.notification_manager.send_telegram_message(
                f"Failed to open position: {str(e)}",
                level="ERROR"
            )
            
    async def _close_position(self, symbol: str, reason: str):
        """
        Close existing position.
        
        Args:
            symbol: Trading symbol
            reason: Reason for closing
        """
        try:
            position = self.positions[symbol]
            
            # Create order
            if self.config.mode == 'live':
                order = self.exchange.create_order(
                    symbol=symbol,
                    side='sell' if position['side'] == 'buy' else 'buy',
                    order_type='MARKET',
                    quantity=position['size']
                )
                exit_price = float(order['price'])
            else:
                exit_price = self.market_data[symbol]['close'].iloc[-1]
                
            # Calculate PnL
            if position['side'] == 'buy':
                pnl = (exit_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - exit_price) * position['size']
                
            # Update metrics
            self.strategy.close_trade(exit_price, datetime.now(), reason)
            
            # Send notifications
            await self.notification_manager.send_telegram_message(
                f"Closed {position['side']} position in {symbol}\n" \
                f"Entry: {position['entry_price']:.8f}\n" \
                f"Exit: {exit_price:.8f}\n" \
                f"PnL: {pnl:.8f}\n" \
                f"Reason: {reason}",
                level="TRADE"
            )
            
            # Remove position
            del self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            await self.notification_manager.send_telegram_message(
                f"Failed to close position: {str(e)}",
                level="ERROR"
            )
            
    async def _get_account_balance(self) -> float:
        """Get current account balance."""
        if self.config.mode == 'live':
            balances = self.exchange.get_account_balance()
            # Sum up all asset values in USDT
            total_balance = 0.0
            for balance in balances:
                if float(balance['free']) > 0:
                    # Convert to USDT if needed
                    if balance['asset'] != 'USDT':
                        price = self.exchange.get_symbol_price(f"{balance['asset']}USDT")
                        total_balance += float(balance['free']) * price
                    else:
                        total_balance += float(balance['free'])
            return total_balance
        else:
            # Use initial balance adjusted by PnL for paper trading
            return 10000.0  # Example initial balance
            
    def _update_performance_metrics(self):
        """Update and log performance metrics."""
        metrics = self.strategy.get_performance_metrics()
        
        # Add risk metrics
        returns = pd.Series([t['pnl'] for t in self.strategy.trades])
        
        if len(returns) > 0:
            metrics.update({
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'sortino_ratio': self._calculate_sortino_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(returns)
            })
            
        self.performance_metrics = metrics
        
        # Log metrics periodically
        if len(self.strategy.trades) % 10 == 0:  # Every 10 trades
            self.notification_manager.notify_performance(metrics)
            
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
        
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        return np.sqrt(252) * returns.mean() / downside_returns.std()
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        return abs(drawdowns.min()) 