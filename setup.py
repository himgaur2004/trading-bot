from setuptools import setup, find_packages

setup(
    name="crypto_trading_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.30.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",
    author="Gaurav Singh",
    description="A cryptocurrency trading bot with AI-powered strategies",
    keywords="cryptocurrency, trading, bot, AI, machine learning",
) 