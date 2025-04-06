"""
🕯️ CRT Strategy Configuration

This file contains configuration settings for the CRT strategy and agent.
"""

# --- CRT Strategy Parameters ---
CRT_CONFIG = {
    'min_range_height_pips': 10,    # Minimum height of range candle in pips
    'max_range_height_pips': 100,   # Maximum height of range candle in pips
    'min_manipulation_pips': 5,     # Minimum manipulation distance in pips
    'key_level_proximity_pips': 15, # Maximum distance to consider "near" a key level
    'volume_threshold': 1.5,        # Volume multiplier threshold for confirmation
    'use_key_levels': True,         # Whether to filter by key levels
    'use_market_structure': True,   # Whether to filter by market structure
    'use_volume_confirmation': True # Whether to use volume for confirmation
}

# --- CRT Agent Parameters ---
RUN_CRT_STANDALONE = True  # Whether to run CRT agent as standalone (execute trades)
CRT_PROVIDE_SIGNALS = True  # Whether to provide signals to trading agent

# --- CRT Risk Management ---
CRT_RISK_PERCENT = 1.0  # Risk per trade as percentage of account balance
CRT_MAX_TRADES = 2      # Maximum number of concurrent CRT trades

# --- CRT Timeframes ---
CRT_HIGHER_TIMEFRAME = '4h'  # Timeframe for CRT pattern identification
CRT_LOWER_TIMEFRAME = '1h'   # Timeframe for entry refinement

# --- CRT Check Interval ---
CRT_CHECK_INTERVAL_MINUTES = 60  # How often to run the CRT agent

# --- CRT Pattern Validation ---
CRT_MIN_CONFIDENCE = 70  # Minimum AI confidence to execute a trade
CRT_MIN_RISK_REWARD = 2  # Minimum risk-reward ratio to execute a trade
