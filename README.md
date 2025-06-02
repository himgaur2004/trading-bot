# Crypto Trading Bot Dashboard

A comprehensive cryptocurrency trading dashboard built with Streamlit, featuring real-time market data, portfolio tracking, and multiple trading strategies.

## Features

- Real-time portfolio tracking
- Multiple trading strategies:
  - Moving Average Crossover
  - RSI Strategy
  - MACD Strategy
  - Bollinger Bands
  - Custom combined strategy
- Interactive strategy backtesting
- Performance metrics visualization
- Risk management settings

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
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

4. Set up your CoinDCX API credentials:
- Create a `.streamlit/secrets.toml` file
- Add your API credentials:
```toml
COINDCX_API_KEY = "your_api_key"
COINDCX_API_SECRET = "your_api_secret"
```

5. Run the dashboard:
```bash
streamlit run dashboard/main.py
```

## Deployment

The dashboard is deployed on Streamlit Cloud. To deploy your own instance:

1. Fork this repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy the app by selecting your forked repository
5. Add your API credentials in the Streamlit Cloud secrets management

## Configuration

The dashboard can be configured through the Settings page in the UI or by modifying the following files:
- `.streamlit/config.toml`: Streamlit configuration
- `dashboard/utils/strategy_handler.py`: Trading strategy parameters
- `dashboard/utils/data_handler.py`: Data fetching and processing settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# trading-bot
