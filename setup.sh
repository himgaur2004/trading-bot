#!/bin/bash

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install numpy first (required for pandas)
pip install numpy==1.24.0

# Install pandas separately
pip install pandas==2.0.3

# Install other requirements
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p dashboard/components
mkdir -p dashboard/utils
mkdir -p dashboard/pages
mkdir -p dashboard/static
mkdir -p data
mkdir -p logs

# Create .streamlit directory and config if it doesn't exist
mkdir -p .streamlit
if [ ! -f ".streamlit/config.toml" ]; then
    echo "Creating Streamlit config..."
    echo '[theme]
    primaryColor="#FF4B4B"
    backgroundColor="#0E1117"
    secondaryBackgroundColor="#262730"
    textColor="#FAFAFA"
    font="sans serif"

[server]
    runOnSave = true
    
[browser]
    gatherUsageStats = false' > .streamlit/config.toml
fi

# Create secrets.toml if it doesn't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "Creating secrets template..."
    echo '# Add your CoinDCX API credentials here
COINDCX_API_KEY = ""
COINDCX_API_SECRET = ""' > .streamlit/secrets.toml
fi

echo "Setup complete! You can now run the dashboard using:"
echo "streamlit run dashboard/main.py" 