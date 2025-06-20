"""
Test script to verify API configuration
"""
from dashboard.utils.data_handler import CoinDCXDataHandler
from config import API_KEY, API_SECRET

def test_config():
    print("Testing API Configuration...")
    print("-" * 50)
    
    # Verify API credentials format
    print("\nChecking API credentials format:")
    if len(API_KEY) != 40:
        print(f"❌ API Key length is incorrect: {len(API_KEY)} chars (should be 40)")
    else:
        print("✅ API Key length is correct")
        
    if len(API_SECRET) != 64:
        print(f"❌ API Secret length is incorrect: {len(API_SECRET)} chars (should be 64)")
    else:
        print("✅ API Secret length is correct")
    
    # Test API connection
    print("\nTesting API connection:")
    try:
        handler = CoinDCXDataHandler(API_KEY, API_SECRET)
        account_info = handler.get_account_info()
        
        print("✅ API connection successful!")
        print("\nAccount Information:")
        print(f"Name: {account_info.get('name', 'N/A')}")
        print(f"Email: {account_info.get('email', 'N/A')}")
        print(f"Status: {account_info.get('status', 'N/A')}")
        
        # Test balances
        print("\nFetching balances...")
        balances = handler.get_balances()
        non_zero = [b for b in balances if float(b['balance']) > 0]
        
        if non_zero:
            print(f"\nFound {len(non_zero)} non-zero balances:")
            for balance in non_zero:
                print(f"• {balance['currency']}: {balance['balance']}")
        else:
            print("No non-zero balances found")
            
    except Exception as e:
        print(f"\n❌ API connection failed: {str(e)}")
        print("\nPlease check:")
        print("1. API credentials in config.py are correct")
        print("2. API key is active on CoinDCX")
        print("3. API key has necessary permissions")

if __name__ == "__main__":
    test_config() 