import hmac
import hashlib
import json
import time
import requests

def test_api(api_key, api_secret):
    """Simple test for CoinDCX API."""
    print("\nTesting CoinDCX API connection...")
    
    # Create request body
    body = {
        "timestamp": str(int(time.time() * 1000))
    }
    
    # Create signature
    signature = hmac.new(
        api_secret.encode('utf-8'),
        json.dumps(body, separators=(',', ':')).encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Set headers
    headers = {
        'Content-Type': 'application/json',
        'X-AUTH-APIKEY': api_key,
        'X-AUTH-SIGNATURE': signature
    }
    
    print("\nRequest details:")
    print(f"Timestamp: {body['timestamp']}")
    print(f"Signature: {signature}")
    
    # Test public endpoint
    print("\nTesting public endpoint...")
    response = requests.get('https://api.coindcx.com/exchange/v1/markets_details', verify=False)
    if response.status_code == 200:
        print("✅ Public API accessible")
        markets = response.json()
        print(f"Found {len(markets)} markets")
    else:
        print("❌ Cannot access public API")
        return
    
    # Test private endpoint
    print("\nTesting private endpoint...")
    response = requests.post(
        'https://api.coindcx.com/exchange/v1/users/balances',
        headers=headers,
        json=body,
        verify=False
    )
    
    print(f"\nResponse status: {response.status_code}")
    print(f"Response body: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ API authentication successful!")
        data = response.json()
        if isinstance(data, list):
            usdt = next((b for b in data if b.get('currency') == 'USDT'), None)
            if usdt:
                print(f"USDT Balance: {usdt.get('balance', 'N/A')}")
    else:
        print("\n❌ API authentication failed")
        print("Please check:")
        print("1. API key and secret are correct")
        print("2. API key has required permissions")
        print("3. Your IP is whitelisted")
        print("4. System time is accurate")

if __name__ == "__main__":
    # Get API credentials
    api_key = input("Enter your API key: ")
    api_secret = input("Enter your API secret: ")
    
    test_api(api_key, api_secret) 