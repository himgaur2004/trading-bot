import hmac
import hashlib
import json
import time
import requests

def test_api():
    # API credentials
    api_key = "ea2224143d465699a2269a98a7a5cd0961252b4705e87973"
    api_secret = "e3646cd3e8a59d94d41bedcbd95b20ad6cf2b4fcbe62031fe30927d258e836f0"

    # Test endpoint - get ticker data (public API)
    public_url = "https://api.coindcx.com/exchange/ticker"
    
    print("1. Testing public API...")
    try:
        response = requests.get(public_url)
        if response.status_code == 200:
            print("✅ Public API working!")
        else:
            print("❌ Public API failed!")
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Error accessing public API: {str(e)}")

    # Test private API - get balances
    private_url = "https://api.coindcx.com/exchange/v1/users/balances"
    
    print("\n2. Testing private API (with authentication)...")
    try:
        # Create timestamp and request body
        timestamp = str(int(time.time() * 1000))
        body = {"timestamp": timestamp}
        json_body = json.dumps(body)

        # Create signature
        signature = hmac.new(
            api_secret.encode('utf-8'),
            json_body.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Headers
        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': api_key,
            'X-AUTH-SIGNATURE': signature
        }

        # Make authenticated request
        response = requests.post(private_url, data=json_body, headers=headers)
        
        if response.status_code == 200:
            print("✅ Private API authentication successful!")
            print("\nFirst balance entry:")
            print(json.dumps(response.json()[0], indent=2))
        else:
            print("❌ Private API authentication failed!")
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error accessing private API: {str(e)}")

if __name__ == "__main__":
    print("🔍 Starting CoinDCX API Test...")
    print("-" * 50)
    test_api()
    print("-" * 50)
    print("Test complete!") 