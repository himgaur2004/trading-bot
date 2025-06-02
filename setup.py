from setuptools import setup, find_packages

setup(
    name="crypto-trading-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.29.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'plotly>=5.18.0',
        'ccxt>=4.0.0',
        'python-binance>=1.0.19',
        'ta>=0.10.0',
        'scikit-learn>=1.3.2',
    ],
    python_requires='>=3.8',
) 