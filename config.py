# Configuration file for crt_agent.py

import MetaTrader5 as mt5
import logging

# --- MT5 Connection Settings ---
# IMPORTANT: Replace with your actual MT5 credentials and server name
# Consider using environment variables or a more secure method for production

# Demo Account Options - Uncomment the account you want to use

# Deriv-Demo Account 1
# MT5_LOGIN =              # Deriv-Demo account number 1
# MT5_PASSWORD = ""   # Deriv-Demo password 1
# MT5_SERVER = "" # Deriv-Demo server

# Deriv-Demo Account 2
MT5_LOGIN =             # Deriv-Demo account number 2
MT5_PASSWORD = ""   # Deriv-Demo password 2
MT5_SERVER = "" # Deriv-Demo server

# FBS-Demo Account
# MT5_LOGIN =              # FBS-Demo account number
# MT5_PASSWORD = ""   # FBS-Demo password
# MT5_SERVER = "" # FBS-Demo server

# --- MT5 Installation ---
# Optional: Specify the full path to the MT5 terminal executable
# Useful if you have multiple terminals or it's not in the default location
# Leave empty ("") to let the library try to auto-detect the default installation.
MT5_TERMINAL_PATH = "" # e.g., r"C:\Program Files\MetaTrader 5\terminal64.exe"

# --- Trading Parameters ---
# Single symbol mode
SYMBOL = "EURUSD"                   # Trading symbol (e.g., "EURUSD", "GBPUSD", "XAUUSD")

# Multi-symbol mode - List of symbols to trade
SYMBOLS = [
    # Forex Majors
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF",
    # Forex Crosses
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY",
    # Commodities
    "XAUUSD", "XAGUSD", "XBRUSD", "XTIUSD",
    # Indices
    "US30", "US500", "USTEC", "UK100", "DE30", "JP225",
    # Cryptocurrencies
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD",
    # Deriv Synthetic Pairs
    "Volatility 10 Index", "Volatility 25 Index", "Volatility 50 Index", "Volatility 75 Index",
    "Volatility 100 Index", "Volatility 10 (1s) Index", "Volatility 25 (1s) Index", "Volatility 50 (1s) Index",
    "Volatility 75 (1s) Index", "Volatility 100 (1s) Index", "Crash 1000 Index", "Boom 1000 Index",
    "Step Index", "Jump 10 Index", "Jump 25 Index", "Jump 50 Index", "Jump 75 Index", "Jump 100 Index",
    "Range Break 100 Index", "Range Break 200 Index"
]

# Active symbols - Will be populated based on available symbols in the broker
ACTIVE_SYMBOLS = []

# Timeframes
TIMEFRAME = mt5.TIMEFRAME_H1  # Main timeframe for analysis
HIGHER_TIMEFRAME = mt5.TIMEFRAME_H4  # Higher timeframe for key levels and structure
LOWER_TIMEFRAME = mt5.TIMEFRAME_M15  # Lower timeframe for entry precision

# Multi-symbol mode
MULTI_SYMBOL_MODE = True  # Set to True to trade multiple symbols

# --- Risk Management ---
RISK_PERCENT = 0.5  # Risk per trade as a percentage of account balance (e.g., 2.0 for 2%)
MAX_TRADES = 5      # Maximum number of concurrent open trades for this symbol/agent

# --- Agent Behavior ---
CHECK_INTERVAL = 60    # Time between checks in seconds (e.g., 60 for 1 min, 300 for 5 mins)
BACKTEST_MODE = False  # Set to False to enable actual order placement (for live trading)
MAGIC_NUMBER = 678901  # Unique integer identifier for trades placed by this agent

# --- Logging ---
LOG_LEVEL = logging.INFO  # Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
LOG_FILE = "crt_agent.log" # Name of the log file

# --- Strategy Specifics ---
# Enable/Disable Killzone Time Filter
USE_KILLZONES = True # Set to False to disable the killzone time filter

# Optional: Minimum required Risk-Reward Ratio (can be used in filtering later)
# MIN_RR_RATIO = 1.5

# --- Sanity Checks (Optional) ---
if not isinstance(MT5_LOGIN, int):
    raise ValueError("MT5_LOGIN must be an integer account number.")
if not MT5_PASSWORD or MT5_PASSWORD == "YOUR_PASSWORD":
    # Use print here as logging might not be configured yet if run standalone
    print("WARNING: MT5_PASSWORD is not set or uses the default placeholder in config.py")
if not MT5_SERVER or MT5_SERVER == "YOUR_BROKER_SERVER":
    print("WARNING: MT5_SERVER is not set or uses the default placeholder in config.py")
if not isinstance(MAGIC_NUMBER, int) or MAGIC_NUMBER <= 0:
     raise ValueError("MAGIC_NUMBER must be a positive integer.")

print(f"Configuration loaded from config.py (Symbol: {SYMBOL}, Magic: {MAGIC_NUMBER})")
