# Crypto Trading Bot

A sophisticated cryptocurrency trading bot implementing multiple advanced trading strategies:

1. Grid Trading Strategy
2. Arbitrage Strategy
3. Sentiment Analysis Strategy
4. Order Flow Strategy
5. Market Making Strategy

## Features

- Multiple trading strategies with ensemble combination
- Real-time market data analysis
- Advanced risk management
- Position sizing and management
- Stop loss and take profit management
- Comprehensive logging and monitoring
- Support for multiple exchanges (via CCXT)
- Testnet/paper trading support

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd crypto-trading-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your configuration:
```bash
EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret
EXCHANGE_NAME=binance  # or other supported exchange
TRADING_PAIRS=BTC/USDT,ETH/USDT  # comma-separated pairs
```

## Usage

1. Start the bot:
```bash
python main.py
```

2. Monitor the logs:
- Trading activity: trading_bot.log
- Errors: error.log

## Configuration

The bot can be configured through:

1. Environment variables (.env file)
2. Configuration classes in config.py
3. Strategy-specific parameters in each strategy file

### Key Configuration Options

- Exchange settings (exchange name, API credentials)
- Trading pairs
- Position sizing and risk parameters
- Strategy weights for ensemble
- Logging and monitoring settings

## Trading Strategies

### 1. Grid Trading Strategy
- Dynamic grid sizing using ATR
- Adaptive range boundaries
- Volume and volatility filters

### 2. Arbitrage Strategy
- Cross-exchange price analysis
- Order book depth validation
- Fee-aware execution

### 3. Sentiment Analysis Strategy
- Multi-source sentiment analysis
- Time decay weighting
- Market indicator integration

### 4. Order Flow Strategy
- Order book analysis
- Trade flow patterns
- Volume profile analysis

### 5. Market Making Strategy
- Dynamic spread calculation
- Inventory management
- Order book presence

## Risk Management

- Position sizing based on account value
- Maximum drawdown limits
- Stop loss and take profit management
- Multiple confirmation signals
- Volume and volatility filters

## Monitoring and Logging

- Real-time trade logging
- Error tracking and reporting
- Performance metrics
- Position monitoring

## Safety Features

- Testnet/paper trading support
- Emergency stop functionality
- Error handling and recovery
- Rate limiting compliance

## Disclaimer

This bot is for educational and research purposes only. Use at your own risk. Cryptocurrency trading carries significant risks and may not be suitable for everyone. Always start with paper trading and small positions when testing new strategies.
