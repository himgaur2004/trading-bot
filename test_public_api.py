import requests
import json

def test_public_api():
    """Test public CoinDCX API endpoints that don't require authentication."""
    base_url = "https://api.coindcx.com"
    
    print("Testing CoinDCX Public API...")
    print("-" * 50)
    
    try:
        # Test markets endpoint
        print("\nFetching market data...")
        markets_response = requests.get(f"{base_url}/exchange/v1/markets")
        markets_response.raise_for_status()
        markets = markets_response.json()
        print(f"✅ Successfully fetched {len(markets)} markets")
        
        # Find BTC-USDT pairs
        btc_pairs = [market for market in markets if 'BTC' in market and 'USDT' in market]
        print("\nAvailable BTC-USDT pairs:")
        for pair in btc_pairs:
            print(f"- {pair}")
        
        # Test ticker endpoint
        print("\nFetching ticker data...")
        ticker_response = requests.get(f"{base_url}/exchange/ticker")
        ticker_response.raise_for_status()
        tickers = ticker_response.json()
        print(f"✅ Successfully fetched {len(tickers)} tickers")
        
        # Find BTC-USDT tickers
        btc_tickers = [ticker for ticker in tickers if 'BTC' in ticker['market'] and 'USDT' in ticker['market']]
        print("\nBTC-USDT market tickers:")
        for ticker in btc_tickers:
            print(f"- {ticker['market']}: ${float(ticker['last_price']):,.2f}")
        
        print("\n✅ Public API test completed successfully!")
        
        # Save market data for reference
        print("\nSaving market data to market_data.json...")
        with open('market_data.json', 'w') as f:
            json.dump({
                'markets': markets,
                'tickers': tickers
            }, f, indent=2)
        print("✅ Market data saved")
        
    except Exception as e:
        print(f"\n❌ Error testing public API: {str(e)}")
        print("\nPlease check your internet connection")

if __name__ == "__main__":
    test_public_api() 