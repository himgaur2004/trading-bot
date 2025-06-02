from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import pandas as pd
from loguru import logger
from ..database.database import DatabaseHandler
from strategies.grid_trading_strategy import GridTradingStrategy
from strategies.arbitrage_strategy import ArbitrageStrategy
from strategies.sentiment_strategy import SentimentStrategy
from strategies.order_flow_strategy import OrderFlowStrategy
from strategies.market_making_strategy import MarketMakingStrategy

class StrategyManager:
    def __init__(self, strategy_weights: Dict[str, float] = None):
        """
        Initialize strategy manager.
        
        Args:
            strategy_weights: Dictionary of strategy weights
        """
        self.db = DatabaseHandler()
        self.initialize_strategies(strategy_weights)
        
    def initialize_strategies(self, weights: Dict[str, float] = None):
        """Initialize trading strategies with weights."""
        # Default equal weights if none provided
        if weights is None:
            weights = {
                'grid_trading': 0.2,
                'arbitrage': 0.2,
                'sentiment': 0.2,
                'order_flow': 0.2,
                'market_making': 0.2
            }
            
        self.strategy_weights = weights
        
        # Initialize strategy instances
        self.strategies = {
            'grid_trading': GridTradingStrategy(),
            'arbitrage': ArbitrageStrategy(),
            'sentiment': SentimentStrategy(),
            'order_flow': OrderFlowStrategy(),
            'market_making': MarketMakingStrategy()
        }
        
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame], symbol: str) -> List[Dict]:
        """
        Generate trading signals from all strategies.
        
        Args:
            market_data: Market data dictionary
            symbol: Trading pair symbol
            
        Returns:
            List of strategy signals
        """
        signals = []
        timestamp = datetime.now()
        
        try:
            # Format market data for strategies
            if market_data.get('ticker'):
                logger.info(f"\nGenerating signals for {symbol} at {timestamp}:")
                logger.info(f"  Market Conditions:")
                logger.info(f"    • Price Change: {market_data['analysis']['price_change']:.2f}%")
                logger.info(f"    • Volume Change: {market_data['analysis']['volume_change']:.2f}%")
                logger.info(f"    • Bid/Ask Ratio: {market_data['analysis']['bid_ask_ratio']:.2f}")
                logger.info(f"    • Buy/Sell Ratio: {market_data['analysis']['buy_sell_ratio']:.2f}")
                
                # Run strategies concurrently
                strategy_tasks = []
                for name, strategy in self.strategies.items():
                    logger.info(f"\n  Running {name} strategy:")
                    task = asyncio.create_task(
                        strategy.generate_signal(market_data, symbol)
                    )
                    strategy_tasks.append((name, task))
                
                # Wait for all strategies to complete
                for name, task in strategy_tasks:
                    try:
                        result = await task
                        if result:
                            # Add market data and risk parameters to signal
                            result.update({
                                'market_data': market_data,
                                'account_balance': 1000,  # TODO: Get from account service
                                'entry_price': float(market_data['ticker']['last_price']),
                                'stop_loss': float(market_data['ticker']['last_price']) * 0.98,  # 2% stop loss
                                'take_profit': float(market_data['ticker']['last_price']) * 1.04  # 4% take profit
                            })
                            
                            signals.append(result)
                            logger.info(f"    ✓ {name} Signal Generated:")
                            logger.info(f"      Signal Details:")
                            logger.info(f"        • Type: {result.get('type', 'N/A')}")
                            logger.info(f"        • Side: {result.get('side', 'N/A')}")
                            logger.info(f"        • Entry Price: {result.get('entry_price', 'N/A')}")
                            logger.info(f"        • Stop Loss: {result.get('stop_loss', 'N/A')}")
                            logger.info(f"        • Take Profit: {result.get('take_profit', 'N/A')}")
                            
                            # Log technical indicators
                            if 'indicators' in result:
                                logger.info(f"      Technical Indicators:")
                                for indicator, value in result['indicators'].items():
                                    logger.info(f"        • {indicator}: {value}")
                            
                            # Log entry/exit points
                            if 'entry_exit' in result:
                                logger.info(f"      Entry/Exit Points:")
                                logger.info(f"        • Entry: {result['entry_exit'].get('entry', 'N/A')}")
                                logger.info(f"        • Stop Loss: {result['entry_exit'].get('stop_loss', 'N/A')}")
                                logger.info(f"        • Take Profit: {result['entry_exit'].get('take_profit', 'N/A')}")
                            
                            # Log risk parameters
                            if 'risk' in result:
                                logger.info(f"      Risk Parameters:")
                                logger.info(f"        • Position Size: {result['risk'].get('position_size', 'N/A')}")
                                logger.info(f"        • Risk/Reward: {result['risk'].get('risk_reward', 'N/A')}")
                                logger.info(f"        • Max Loss: {result['risk'].get('max_loss', 'N/A')}")
                            
                            # Log confidence metrics
                            if 'confidence' in result:
                                logger.info(f"      Confidence Metrics:")
                                logger.info(f"        • Signal Strength: {result['confidence'].get('strength', 'N/A')}")
                                logger.info(f"        • Probability: {result['confidence'].get('probability', 'N/A')}")
                                logger.info(f"        • Confirmation Count: {result['confidence'].get('confirmations', 'N/A')}")
                        else:
                            logger.info(f"    - {name}: No signal generated")
                    except Exception as e:
                        logger.error(f"    ✗ {name} error: {e}")
                
                if signals:
                    logger.info(f"\n  Total signals for {symbol}: {len(signals)}")
                else:
                    logger.info(f"\n  No signals generated for {symbol}")
                
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            
        return signals
        
    def combine_signals(self, signals: Dict[str, Dict]) -> Dict:
        """
        Combine signals from multiple strategies.
        
        Args:
            signals: Dictionary of strategy signals
            
        Returns:
            Combined signal information
        """
        if not signals:
            return {
                'signal': 0,
                'strength': 0,
                'strategies': []
            }
            
        # Calculate weighted signals
        weighted_signals = []
        weighted_strengths = []
        active_strategies = []
        
        logger.info("\nCombining strategy signals:")
        for name, data in signals.items():
            signal = data['signal']
            strength = data['strength']
            weight = data['weight']
            
            weighted_signals.append(signal * weight)
            weighted_strengths.append(strength * weight)
            
            if abs(signal) > 0:
                active_strategies.append(name)
                logger.info(f"  • {name}:")
                logger.info(f"    - Signal: {signal}")
                logger.info(f"    - Strength: {strength}")
                logger.info(f"    - Weight: {weight}")
                logger.info(f"    - Weighted Signal: {signal * weight}")
                
        # Combine signals
        combined_signal = sum(weighted_signals)
        combined_strength = sum(weighted_strengths)
        
        logger.info("\nFinal combined signal:")
        logger.info(f"  • Signal: {combined_signal}")
        logger.info(f"  • Strength: {combined_strength}")
        logger.info(f"  • Active Strategies: {', '.join(active_strategies)}")
        
        return {
            'signal': combined_signal,
            'strength': combined_strength,
            'strategies': active_strategies
        }
        
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float,
                              signal_strength: float) -> float:
        """
        Calculate position size based on signals.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Market volatility
            signal_strength: Combined signal strength
            
        Returns:
            Position size in base currency
        """
        # Get position sizes from each strategy
        strategy_sizes = {}
        
        for name, strategy in self.strategies.items():
            size = strategy.calculate_position_size(
                account_balance,
                current_price,
                volatility
            )
            strategy_sizes[name] = size * self.strategy_weights[name]
            
        # Combine position sizes
        base_size = sum(strategy_sizes.values())
        
        # Adjust for signal strength
        adjusted_size = base_size * min(abs(signal_strength), 1.0)
        
        return adjusted_size
        
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
        stop_updates = {}
        
        for strategy in self.strategies.values():
            updates = strategy.should_update_stops(
                current_price,
                position_data
            )
            
            # Merge updates
            for level_type, price in updates.items():
                if level_type not in stop_updates:
                    stop_updates[level_type] = price
                else:
                    # Use most conservative stop
                    if position_data['side'] == 'buy':
                        stop_updates[level_type] = max(
                            stop_updates[level_type],
                            price
                        )
                    else:
                        stop_updates[level_type] = min(
                            stop_updates[level_type],
                            price
                        )
                        
        return stop_updates
        
    def get_strategy_performance(self,
                               start_time: datetime,
                               end_time: datetime = None) -> Dict[str, Dict]:
        """
        Get performance metrics for each strategy.
        
        Args:
            start_time: Start time for analysis
            end_time: End time for analysis
            
        Returns:
            Dictionary of strategy performance metrics
        """
        if end_time is None:
            end_time = datetime.now()
            
        performance = {}
        
        for name in self.strategies.keys():
            # Get strategy trades
            trades = self.db.get_trades(
                strategy=name,
                start_time=start_time,
                end_time=end_time
            )
            
            if trades:
                # Calculate metrics
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t['pnl'] > 0])
                total_pnl = sum(t['pnl'] for t in trades if t['pnl'])
                
                performance[name] = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': total_pnl
                }
            else:
                performance[name] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0
                }
                
        return performance 