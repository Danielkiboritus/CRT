# CRT Trading Agent for MetaTrader 5
# Based on CRT/ICT Smart Money Concepts

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, time, timedelta
import time as time_module # Rename to avoid conflict with datetime.time
import logging
import math
import pytz # For timezone handling
import config  # Import configuration

# --- Configuration (Load from config.py) ---
MT5_LOGIN = config.MT5_LOGIN
MT5_PASSWORD = config.MT5_PASSWORD
MT5_SERVER = config.MT5_SERVER
MT5_TERMINAL_PATH = config.MT5_TERMINAL_PATH
SYMBOL = config.SYMBOL
HIGHER_TIMEFRAME = config.HIGHER_TIMEFRAME
LOWER_TIMEFRAME = config.LOWER_TIMEFRAME
RISK_PERCENT = config.RISK_PERCENT
MAX_TRADES = config.MAX_TRADES
MAGIC_NUMBER = config.MAGIC_NUMBER
CHECK_INTERVAL = config.CHECK_INTERVAL
BACKTEST_MODE = config.BACKTEST_MODE
USE_KILLZONES = config.USE_KILLZONES
LOG_LEVEL = config.LOG_LEVEL
LOG_FILE = config.LOG_FILE

# --- Logging Setup ---
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # Use a specific logger

# --- MT5 Connection ---
def initialize_mt5(username, password, server, path=None):
    """Initialize connection to MT5 terminal."""
    logger.info("Initializing MT5 connection...")
    init_params = {
        "login": username,
        "password": password,
        "server": server,
    }
    # Only include path if it's provided and not empty
    if path:
        init_params["path"] = path
        logger.info(f"Using terminal path: {path}")
    else:
        logger.info("Using default terminal path detection.")

    # Attempt to initialize connection
    if not mt5.initialize(**init_params):
        logger.error(f"initialize() failed, error code = {mt5.last_error()}")
        return False

    # Display connection status
    logger.info(f"MetaTrader5 package version: {mt5.__version__}")

    # Display account info
    account_info = mt5.account_info()
    if account_info:
        logger.info(f"Connected to account: {account_info.login} on {account_info.server} [{account_info.name}]")
    else:
        logger.error(f"Failed to get account info after initialize: {mt5.last_error()}")
        mt5.shutdown()
        return False

    return True

# --- Data Retrieval ---
def get_ohlc_data(symbol, timeframe, count=500):
    """Retrieve OHLC data for the specified symbol and timeframe."""
    logger.debug(f"Fetching {count} {timeframe} candles for {symbol}") # Use timeframe constant directly
    try:
        # Request rates from MT5
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

        # Check if data was received
        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to get rates for {symbol} on {timeframe}: {mt5.last_error()}")
            return None

        # Convert to pandas DataFrame
        df = pd.DataFrame(rates)
        # Convert time in seconds into UTC datetime objects (MT5 times are usually UTC)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        # Set time as index
        df.set_index('time', inplace=True)

        logger.debug(f"Successfully fetched {len(df)} candles for {symbol} on {timeframe}.")
        return df

    except Exception as e:
        logger.error(f"Error getting OHLC data: {e}", exc_info=True)
        return None

# --- Analysis Functions ---

def find_key_levels(df, n_levels=10):
    """Identify key price levels from the data using swing highs/lows and session ranges."""
    logger.debug(f"Finding up to {n_levels} key levels...")
    if df is None or len(df) < 5:
        logger.warning("Not enough data to find key levels.")
        return []

    levels = []
    df_reset = df.reset_index() # Work with index for iloc access

    # Find swing highs and lows (5-candle pattern)
    logger.debug("Detecting swing points...")
    for i in range(2, len(df_reset)-2):
        is_swing_high = (df_reset.iloc[i]['high'] > df_reset.iloc[i-1]['high'] and
                         df_reset.iloc[i]['high'] > df_reset.iloc[i-2]['high'] and
                         df_reset.iloc[i]['high'] > df_reset.iloc[i+1]['high'] and
                         df_reset.iloc[i]['high'] > df_reset.iloc[i+2]['high'])
        if is_swing_high:
            levels.append(df_reset.iloc[i]['high'])

        is_swing_low = (df_reset.iloc[i]['low'] < df_reset.iloc[i-1]['low'] and
                        df_reset.iloc[i]['low'] < df_reset.iloc[i-2]['low'] and
                        df_reset.iloc[i]['low'] < df_reset.iloc[i+1]['low'] and
                        df_reset.iloc[i]['low'] < df_reset.iloc[i+2]['low'])
        if is_swing_low:
            levels.append(df_reset.iloc[i]['low'])

    # Add recent high/low as potential levels
    if not df.empty:
        levels.append(df['high'].iloc[-1])
        levels.append(df['low'].iloc[-1])

    # --- Add Session Highs/Lows ---
    logger.debug("Calculating session ranges...")
    try:
        # Ensure 'time' column is datetime and UTC
        if not pd.api.types.is_datetime64_any_dtype(df_reset['time']):
            df_reset['time'] = pd.to_datetime(df_reset['time'], utc=True)
        elif df_reset['time'].dt.tz is None:
            df_reset['time'] = df_reset['time'].dt.tz_localize('UTC')
        elif df_reset['time'].dt.tz.zone != 'UTC':
            df_reset['time'] = df_reset['time'].dt.tz_convert('UTC')

        # Define session hours in UTC (Adjust these based on your analysis/broker)
        # Example: Roughly Asian (00-08), London (07-16), NY (13-21) UTC
        asian_start_hour, asian_end_hour = 0, 8
        london_start_hour, london_end_hour = 7, 16
        ny_start_hour, ny_end_hour = 13, 21

        # Filter data for each session based on UTC hour
        # Consider only the last few days for session levels to be relevant
        days_lookback = 5 # Look back 5 days for session levels
        cutoff_date = df_reset['time'].max() - timedelta(days=days_lookback)
        recent_df = df_reset[df_reset['time'] >= cutoff_date]

        if not recent_df.empty:
            # Asian Session
            asian_session = recent_df[
                (recent_df['time'].dt.hour >= asian_start_hour) &
                (recent_df['time'].dt.hour < asian_end_hour)
            ]
            if not asian_session.empty:
                levels.append(asian_session['high'].max())
                levels.append(asian_session['low'].min())
                logger.debug(f"Recent Asian Range (approx): Low={levels[-1]:.5f}, High={levels[-2]:.5f}")

            # London Session
            london_session = recent_df[
                (recent_df['time'].dt.hour >= london_start_hour) &
                (recent_df['time'].dt.hour < london_end_hour)
            ]
            if not london_session.empty:
                levels.append(london_session['high'].max())
                levels.append(london_session['low'].min())
                logger.debug(f"Recent London Range (approx): Low={levels[-1]:.5f}, High={levels[-2]:.5f}")

            # New York Session
            ny_session = recent_df[
                (recent_df['time'].dt.hour >= ny_start_hour) &
                (recent_df['time'].dt.hour < ny_end_hour)
            ]
            if not ny_session.empty:
                levels.append(ny_session['high'].max())
                levels.append(ny_session['low'].min())
                logger.debug(f"Recent NY Range (approx): Low={levels[-1]:.5f}, High={levels[-2]:.5f}")

    except Exception as e:
        logger.warning(f"Could not calculate session levels: {e}", exc_info=True)

    # Get symbol digits for rounding
    symbol_info = mt5.symbol_info(SYMBOL)
    digits = symbol_info.digits if symbol_info else 5

    # Round levels
    levels = [round(level, digits) for level in levels if pd.notna(level)] # Handle potential NaNs

    # Remove duplicates and sort
    levels = sorted(list(set(levels)))
    logger.debug(f"Raw levels identified ({len(levels)}): {levels}")

    # Simplify by grouping nearby levels
    if len(levels) > 1:
        simplified_levels = [levels[0]]
        # Use a dynamic threshold based on price magnitude (e.g., 5 pips)
        point_value = symbol_info.point if symbol_info else (10**-digits)
        threshold_points = 50 # 5 pips for 5-digit, adjust as needed
        threshold_price = threshold_points * point_value

        for level in levels[1:]:
            if abs(level - simplified_levels[-1]) < threshold_price:
                # logger.debug(f"Simplifying: Skipping {level:.{digits}f} (close to {simplified_levels[-1]:.{digits}f})")
                continue
            simplified_levels.append(level)
        levels = simplified_levels
        logger.debug(f"Simplified levels ({len(levels)}): {levels}")

    # Return the most relevant n_levels (e.g., those closest to current price)
    # For now, just return the simplified list, capped at n_levels
    # A better approach might be to find levels closest to the current price
    current_price = mt5.symbol_info_tick(SYMBOL).bid if mt5.symbol_info_tick(SYMBOL) else (df_reset['close'].iloc[-1] if not df_reset.empty else 0)
    if current_price > 0 and levels:
         levels.sort(key=lambda x: abs(x - current_price)) # Sort by proximity to current price
         final_levels = sorted(levels[:n_levels]) # Take N closest and sort them by price
    else:
         final_levels = levels[-n_levels:] # Fallback: Take highest/lowest N

    logger.info(f"Identified key levels (closest {n_levels}): {final_levels}")
    return final_levels


def analyze_market_structure(df):
    """Analyze market structure to identify trends, swings, and potential reversals."""
    logger.debug("Analyzing market structure...")
    if df is None or len(df) < 5:
        logger.warning("Not enough data for market structure analysis.")
        return {'current_structure': 'unknown', 'swing_highs': [], 'swing_lows': []}

    df_copy = df.copy()
    df_copy['swing_high'] = False
    df_copy['swing_low'] = False

    logger.debug("Detecting swing points for structure analysis...")
    for i in range(2, len(df_copy)-2):
        if (df_copy.iloc[i]['high'] > df_copy.iloc[i-1]['high'] and
            df_copy.iloc[i]['high'] > df_copy.iloc[i-2]['high'] and
            df_copy.iloc[i]['high'] > df_copy.iloc[i+1]['high'] and
            df_copy.iloc[i]['high'] > df_copy.iloc[i+2]['high']):
            df_copy.loc[df_copy.index[i], 'swing_high'] = True

        if (df_copy.iloc[i]['low'] < df_copy.iloc[i-1]['low'] and
            df_copy.iloc[i]['low'] < df_copy.iloc[i-2]['low'] and
            df_copy.iloc[i]['low'] < df_copy.iloc[i+1]['low'] and
            df_copy.iloc[i]['low'] < df_copy.iloc[i+2]['low']):
            df_copy.loc[df_copy.index[i], 'swing_low'] = True

    swing_highs_df = df_copy[df_copy['swing_high']]
    swing_lows_df = df_copy[df_copy['swing_low']]

    # Store as list of tuples (time, price)
    swing_highs = list(zip(swing_highs_df.index, swing_highs_df['high']))
    swing_lows = list(zip(swing_lows_df.index, swing_lows_df['low']))

    structure = {
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'current_structure': 'unknown'
    }

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_high_val = swing_highs[-1][1]
        prev_high_val = swing_highs[-2][1]
        last_low_val = swing_lows[-1][1]
        prev_low_val = swing_lows[-2][1]

        logger.debug(f"Recent Swings - Highs: {prev_high_val:.5f} -> {last_high_val:.5f}, Lows: {prev_low_val:.5f} -> {last_low_val:.5f}")

        if last_high_val > prev_high_val and last_low_val > prev_low_val:
            structure['current_structure'] = 'bullish' # HH, HL
        elif last_high_val < prev_high_val and last_low_val < prev_low_val:
            structure['current_structure'] = 'bearish' # LH, LL
        elif last_high_val < prev_high_val and last_low_val > prev_low_val:
             structure['current_structure'] = 'ranging_LH_HL' # LH, HL
        elif last_high_val > prev_high_val and last_low_val < prev_low_val:
             structure['current_structure'] = 'ranging_HH_LL' # HH, LL
        else:
             structure['current_structure'] = 'consolidating'

    elif len(swing_highs) > 0 and len(swing_lows) > 0:
         last_close = df_copy['close'].iloc[-1]
         last_high = swing_highs[-1][1]
         last_low = swing_lows[-1][1]
         if last_close > last_high: structure['current_structure'] = 'breaking_high'
         elif last_close < last_low: structure['current_structure'] = 'breaking_low'
         else: structure['current_structure'] = 'ranging_inside'

    logger.info(f"Determined market structure: {structure['current_structure']}")
    return structure


def find_order_blocks(df, structure):
    """Identify potential order blocks (last down/up candle before strong move)."""
    logger.debug("Identifying order blocks...")
    order_blocks = {'bullish': [], 'bearish': []}
    if df is None or len(df) < 4: return order_blocks

    df_reset = df.reset_index()

    # Find bullish order blocks (last down candle before strong up move)
    for i in range(1, len(df_reset)-1): # Need i-1 and i
        candle_prev = df_reset.iloc[i-1] # Potential OB candle
        candle_curr = df_reset.iloc[i]   # Move away candle

        # Condition: Previous candle is bearish, Current candle is bullish and engulfs or moves strongly away
        is_bullish_ob_candidate = (candle_prev['close'] < candle_prev['open'] and
                                   candle_curr['close'] > candle_curr['open'] and
                                   candle_curr['close'] > candle_prev['high']) # Strong move breaking OB high

        if is_bullish_ob_candidate:
            # Optional: Add volume check or compare move size if volume data available
            order_blocks['bullish'].append({
                'time': candle_prev['time'],
                'high': candle_prev['high'],
                'low': candle_prev['low'],
                'open': candle_prev['open'],
                'close': candle_prev['close']
            })
            # logger.debug(f"Potential Bullish OB found at {candle_prev['time']}")

    # Find bearish order blocks (last up candle before strong down move)
    for i in range(1, len(df_reset)-1):
        candle_prev = df_reset.iloc[i-1] # Potential OB candle
        candle_curr = df_reset.iloc[i]   # Move away candle

        # Condition: Previous candle is bullish, Current candle is bearish and engulfs or moves strongly away
        is_bearish_ob_candidate = (candle_prev['close'] > candle_prev['open'] and
                                   candle_curr['close'] < candle_curr['open'] and
                                   candle_curr['close'] < candle_prev['low']) # Strong move breaking OB low

        if is_bearish_ob_candidate:
            order_blocks['bearish'].append({
                'time': candle_prev['time'],
                'high': candle_prev['high'],
                'low': candle_prev['low'],
                'open': candle_prev['open'],
                'close': candle_prev['close']
            })
            # logger.debug(f"Potential Bearish OB found at {candle_prev['time']}")

    logger.info(f"Found {len(order_blocks['bullish'])} potential bullish OBs, {len(order_blocks['bearish'])} potential bearish OBs.")
    # Optional: Log details of the last few found
    # if order_blocks['bullish']: logger.debug(f"Last Bullish OB: {order_blocks['bullish'][-1]}")
    # if order_blocks['bearish']: logger.debug(f"Last Bearish OB: {order_blocks['bearish'][-1]}")
    return order_blocks


def find_fair_value_gaps(df):
    """Identify Fair Value Gaps (FVGs) - 3 candle pattern imbalance."""
    logger.debug("Identifying Fair Value Gaps (FVGs)...")
    fvgs = {'bullish': [], 'bearish': []}
    if df is None or len(df) < 3: return fvgs

    df_reset = df.reset_index()

    # Find bullish FVGs (Gap between candle i-2 low and candle i high)
    for i in range(2, len(df_reset)):
        candle_prev2 = df_reset.iloc[i-2]
        candle_prev1 = df_reset.iloc[i-1] # Middle candle (ignored for gap boundary)
        candle_curr = df_reset.iloc[i]

        if candle_prev2['low'] > candle_curr['high']:
            # Bullish FVG exists between prev2.low and curr.high
            fvgs['bullish'].append({
                'start_time': candle_prev2['time'],
                'end_time': candle_curr['time'],
                'top': candle_prev2['low'],      # Top of the gap
                'bottom': candle_curr['high'],   # Bottom of the gap
                'middle_candle_time': candle_prev1['time']
            })
            # logger.debug(f"Bullish FVG found between {candle_prev2['time']} and {candle_curr['time']}")

    # Find bearish FVGs (Gap between candle i-2 high and candle i low)
    for i in range(2, len(df_reset)):
        candle_prev2 = df_reset.iloc[i-2]
        candle_prev1 = df_reset.iloc[i-1]
        candle_curr = df_reset.iloc[i]

        if candle_prev2['high'] < candle_curr['low']:
            # Bearish FVG exists between prev2.high and curr.low
            fvgs['bearish'].append({
                'start_time': candle_prev2['time'],
                'end_time': candle_curr['time'],
                'top': candle_curr['low'],       # Top of the gap
                'bottom': candle_prev2['high'],  # Bottom of the gap
                'middle_candle_time': candle_prev1['time']
            })
            # logger.debug(f"Bearish FVG found between {candle_prev2['time']} and {candle_curr['time']}")

    logger.info(f"Found {len(fvgs['bullish'])} potential bullish FVGs, {len(fvgs['bearish'])} potential bearish FVGs.")
    # Optional: Log details of the last few found
    # if fvgs['bullish']: logger.debug(f"Last Bullish FVG: {fvgs['bullish'][-1]}")
    # if fvgs['bearish']: logger.debug(f"Last Bearish FVG: {fvgs['bearish'][-1]}")
    return fvgs


# --- CRT Pattern Detection ---
def find_crt_patterns(df_high_tf, df_low_tf, key_levels=None):
    """Identify potential CRT patterns (Range, Manipulation, Distribution)."""
    logger.debug("Finding CRT patterns...")
    patterns = []
    if df_high_tf is None or len(df_high_tf) < 3: # Need 3 candles for the pattern
        logger.warning("Not enough high timeframe data for CRT pattern detection.")
        return patterns

    df = df_high_tf.reset_index()
    digits = mt5.symbol_info(SYMBOL).digits if mt5.symbol_info(SYMBOL) else 5
    point_value = mt5.symbol_info(SYMBOL).point if mt5.symbol_info(SYMBOL) else (10**-digits)
    key_level_proximity_threshold = 15 * point_value # e.g., 1.5 pips proximity to key level

    # Iterate through candles (i = distribution, i-1 = manipulation, i-2 = range)
    for i in range(2, len(df)):
        try:
            range_candle = df.iloc[i-2]
            manipulation_candle = df.iloc[i-1]
            distribution_candle = df.iloc[i] # Current candle completing the pattern sequence

            # --- Check for Bullish CRT Pattern ---
            # 1. Manipulation low breaks below range low
            # 2. Manipulation closes back inside range low (prompt logic)
            # 3. Distribution shows bullish intent (moves up)

            is_bullish_manipulation = manipulation_candle['low'] < range_candle['low']
            is_bullish_close_condition = manipulation_candle['close'] >= range_candle['low'] # Prompt logic
            is_bullish_distribution = distribution_candle['close'] > manipulation_candle['close'] # Basic check

            if is_bullish_manipulation and is_bullish_close_condition: # Core check from prompt
                logger.debug(f"Potential Bullish CRT sequence ending at {distribution_candle['time']}")

                near_key_level = False
                if key_levels:
                    for level in key_levels:
                        if abs(manipulation_candle['low'] - level) <= key_level_proximity_threshold:
                            near_key_level = True
                            logger.debug(f"Bullish CRT manip low {manipulation_candle['low']:.{digits}f} near key level: {level:.{digits}f}")
                            break

                range_height = range_candle['high'] - range_candle['low']
                if range_height <= 0: continue # Avoid division by zero or negative range

                # SL/TP based on prompt logic (manipulation low - 10% range height)
                # Ensure SL is strictly below manipulation low
                sl_offset = max(range_height * 0.1, point_value * 5) # Min 0.5 pip offset
                stop_loss = manipulation_candle['low'] - sl_offset
                take_profit = range_candle['high'] # Target range high
                entry_price = manipulation_candle['low'] # Ideal entry at liquidity grab

                if stop_loss >= entry_price: continue # Invalid SL

                patterns.append({
                    'type': 'bullish',
                    'time': manipulation_candle['time'], # Time of manipulation candle
                    'range_candle': range_candle.to_dict(),
                    'manipulation_candle': manipulation_candle.to_dict(),
                    'distribution_candle': distribution_candle.to_dict(),
                    'entry_price': entry_price,
                    'stop_loss': round(stop_loss, digits),
                    'take_profit': round(take_profit, digits),
                    'high_probability': is_bullish_close_condition, # Based on prompt
                    'near_key_level': near_key_level,
                    'trigger_index': i # Index of the distribution candle
                })

            # --- Check for Bearish CRT Pattern ---
            # 1. Manipulation high breaks above range high
            # 2. Manipulation closes back inside range high (prompt logic)
            # 3. Distribution shows bearish intent (moves down)

            is_bearish_manipulation = manipulation_candle['high'] > range_candle['high']
            is_bearish_close_condition = manipulation_candle['close'] <= range_candle['high'] # Prompt logic
            is_bearish_distribution = distribution_candle['close'] < manipulation_candle['close'] # Basic check

            if is_bearish_manipulation and is_bearish_close_condition: # Core check from prompt
                logger.debug(f"Potential Bearish CRT sequence ending at {distribution_candle['time']}")

                near_key_level = False
                if key_levels:
                    for level in key_levels:
                         if abs(manipulation_candle['high'] - level) <= key_level_proximity_threshold:
                            near_key_level = True
                            logger.debug(f"Bearish CRT manip high {manipulation_candle['high']:.{digits}f} near key level: {level:.{digits}f}")
                            break

                range_height = range_candle['high'] - range_candle['low']
                if range_height <= 0: continue

                # SL/TP based on prompt logic (manipulation high + 10% range height)
                sl_offset = max(range_height * 0.1, point_value * 5) # Min 0.5 pip offset
                stop_loss = manipulation_candle['high'] + sl_offset
                take_profit = range_candle['low'] # Target range low
                entry_price = manipulation_candle['high'] # Ideal entry

                if stop_loss <= entry_price: continue # Invalid SL

                patterns.append({
                    'type': 'bearish',
                    'time': manipulation_candle['time'], # Time of manipulation candle
                    'range_candle': range_candle.to_dict(),
                    'manipulation_candle': manipulation_candle.to_dict(),
                    'distribution_candle': distribution_candle.to_dict(),
                    'entry_price': entry_price,
                    'stop_loss': round(stop_loss, digits),
                    'take_profit': round(take_profit, digits),
                    'high_probability': is_bearish_close_condition, # Based on prompt
                    'near_key_level': near_key_level,
                    'trigger_index': i # Index of the distribution candle
                })
        except Exception as e:
             logger.error(f"Error processing candle index {i} for CRT pattern: {e}", exc_info=True)
             continue

    if patterns:
        logger.info(f"Found {len(patterns)} potential CRT patterns.")
    else:
        logger.debug("No CRT patterns found in the current dataset.")

    return patterns

# --- Risk Management ---
def calculate_lot_size(symbol, risk_amount, sl_distance_pips):
    """Calculate appropriate lot size based on risk parameters."""
    logger.debug(f"Calculating lot size for {symbol}. Risk Amount: {risk_amount:.2f}, SL Pips: {sl_distance_pips}")

    min_lot = 0.01 # Default minimum

    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found, attempting to select.")
            if mt5.symbol_select(symbol, True):
                 symbol_info = mt5.symbol_info(symbol)
                 if symbol_info is None:
                      logger.error(f"Still cannot get info for {symbol} after selecting.")
                      return min_lot
            else:
                 logger.error(f"Failed to select symbol {symbol} to get info.")
                 return min_lot

        min_lot = symbol_info.volume_min
        lot_step = symbol_info.volume_step
        lot_max = symbol_info.volume_max
        digits = symbol_info.digits
        contract_size = symbol_info.trade_contract_size
        tick_value = symbol_info.trade_tick_value # Value of a tick per lot in account currency
        tick_size = symbol_info.trade_tick_size # Size of a tick (e.g., 0.00001)

        if sl_distance_pips <= 0:
             logger.warning(f"Stop loss distance in pips is zero or negative ({sl_distance_pips}). Cannot calculate lot size.")
             return min_lot
        if tick_size <= 0 or tick_value <= 0:
             logger.error(f"Invalid tick_size ({tick_size}) or tick_value ({tick_value}) for {symbol}. Cannot calculate lot size.")
             return min_lot
        if lot_step <= 0:
             logger.warning(f"Invalid lot step ({lot_step}) for {symbol}. Using 0.01.")
             lot_step = 0.01

        # Calculate value per pip per lot
        points_per_pip = 10 if digits in [3, 5] else 1
        value_per_point_per_lot = tick_value / tick_size
        value_per_pip_per_lot = value_per_point_per_lot * points_per_pip * tick_size # Simplified: tick_value * points_per_pip
        # Let's use the most direct interpretation: tick_value is for one tick, so pip value is tick_value * points_in_pip
        pip_value_per_lot = tick_value * points_per_pip
        logger.debug(f"{symbol} Info: TickValue={tick_value}, TickSize={tick_size}, PtsPerPip={points_per_pip}, PipValuePerLot={pip_value_per_lot:.5f}")


        if pip_value_per_lot <= 0:
            logger.error(f"Calculated pip value per lot is zero or negative ({pip_value_per_lot:.5f}). Cannot calculate lot size.")
            return min_lot

        # Calculate lot size
        lot_size = risk_amount / (sl_distance_pips * pip_value_per_lot)
        logger.debug(f"Raw calculated lot size: {lot_size}")

        # Adjust to lot step (round down to avoid exceeding risk)
        lot_size = math.floor(lot_size / lot_step) * lot_step
        logger.debug(f"Lot size rounded down to step {lot_step}: {lot_size}")

        # Ensure lot size is within allowed range
        if lot_size < min_lot:
            logger.warning(f"Calculated lot size {lot_size} is below minimum {min_lot}. Adjusting to minimum.")
            lot_size = min_lot
        elif lot_size > lot_max:
            logger.warning(f"Calculated lot size {lot_size} exceeds maximum {lot_max}. Adjusting to maximum.")
            lot_size = lot_max

        # Final check
        if lot_size < min_lot: # Could happen if min_lot itself is adjusted down due to rounding
             logger.error(f"Final lot size {lot_size} is still below minimum {min_lot}. Returning minimum.")
             return min_lot

        # Determine precision for formatting based on lot_step
        precision = 0
        if lot_step > 0:
           precision = int(abs(math.log10(lot_step))) if lot_step < 1 else 0

        logger.info(f"Calculated Lot Size for {symbol}: {lot_size:.{precision}f}")
        return lot_size

    except Exception as e:
        logger.error(f"Error calculating lot size for {symbol}: {e}", exc_info=True)
        try: # Try to return symbol minimum on error
             si = mt5.symbol_info(symbol)
             if si: return si.volume_min
        except: pass
        return 0.01 # Fallback default minimum


# --- Order Execution ---
def place_order(symbol, order_type, lot_size, entry_price, sl_price, tp_price, deviation=20, comment="CRT Trade"):
    """Place a market order with the specified parameters."""
    logger.info(f"Attempting to place {order_type} order for {lot_size} lots of {symbol} near {entry_price:.5f} [SL={sl_price:.5f}, TP={tp_price:.5f}]")

    order_types_map = {
        'buy': mt5.ORDER_TYPE_BUY,
        'sell': mt5.ORDER_TYPE_SELL,
    }

    if order_type not in order_types_map:
        logger.error(f"Invalid order type specified: '{order_type}'")
        return None

    mt5_order_type = order_types_map[order_type]

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbol {symbol} not found, cannot place order.")
        if mt5.symbol_select(symbol, True):
             symbol_info = mt5.symbol_info(symbol)
             if symbol_info is None:
                  logger.error(f"Still cannot get info for {symbol} after selecting.")
                  return None
        else:
             logger.error(f"Failed to select symbol {symbol} to get info.")
             return None

    if not symbol_info.visible:
        logger.info(f"Symbol {symbol} not visible, attempting to select.")
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol} in MarketWatch. Cannot place order.")
            return None
        symbol_info = mt5.symbol_info(symbol) # Re-fetch info
        if symbol_info is None:
             logger.error(f"Could not get symbol info for {symbol} even after selecting.")
             return None

    digits = symbol_info.digits
    point = symbol_info.point
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"Could not get tick data for {symbol}. Cannot determine market price.")
        return None

    price = tick.ask if mt5_order_type == mt5.ORDER_TYPE_BUY else tick.bid
    sl_price_norm = round(sl_price, digits)
    tp_price_norm = round(tp_price, digits)

    logger.debug(f"Normalized prices: SL={sl_price_norm}, TP={tp_price_norm}. Current Market Price={price:.{digits}f}")

    # Basic validation for SL/TP relative to current market price
    min_stop_level_points = symbol_info.trade_stops_level # Minimum distance in points
    min_stop_level_price = min_stop_level_points * point

    if mt5_order_type == mt5.ORDER_TYPE_BUY:
        if sl_price_norm >= price - min_stop_level_price:
             logger.warning(f"SL price {sl_price_norm} too close to market bid {price}. Min distance: {min_stop_level_price:.{digits}f}")
             # Adjust SL slightly if too close? Or let broker reject? For now, warn only.
             # sl_price_norm = round(price - min_stop_level_price * 1.1, digits) # Example adjustment
        if tp_price_norm <= price + min_stop_level_price:
             logger.warning(f"TP price {tp_price_norm} too close to market ask {price}. Min distance: {min_stop_level_price:.{digits}f}")

    if mt5_order_type == mt5.ORDER_TYPE_SELL:
        if sl_price_norm <= price + min_stop_level_price:
             logger.warning(f"SL price {sl_price_norm} too close to market ask {price}. Min distance: {min_stop_level_price:.{digits}f}")
        if tp_price_norm >= price - min_stop_level_price:
             logger.warning(f"TP price {tp_price_norm} too close to market bid {price}. Min distance: {min_stop_level_price:.{digits}f}")

    # Prepare the trade request dictionary
    request = {
        "action": mt5.TRADE_ACTION_DEAL, # Market execution
        "symbol": symbol,
        "volume": float(lot_size),
        "type": mt5_order_type,
        "price": price, # Market price for TRADE_ACTION_DEAL
        "sl": float(sl_price_norm),
        "tp": float(tp_price_norm),
        "deviation": deviation,
        "magic": MAGIC_NUMBER, # Use magic number from config
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, # Or FOK depending on preference
    }

    logger.debug(f"Prepared order request: {request}")

    # Send the order to the server
    try:
        result = mt5.order_send(request)
    except Exception as e:
         logger.error(f"Exception during order_send for {symbol}: {e}", exc_info=True)
         return None

    if result is None:
        logger.error(f"Order send failed for {symbol}. MT5 returned None. Last error: {mt5.last_error()}")
        return None

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Order placed successfully for {symbol}: OrderID={result.order}, Price={result.price:.{digits}f}, Volume={result.volume}")
        return result
    else:
        logger.error(f"Order placement failed for {symbol}. Retcode: {result.retcode} - {mt5.last_error()} - Comment: {result.comment}")
        logger.error(f"Failed request details: {result.request}")
        return None


# --- Killzone Check ---
def is_in_killzone(tz_name='America/New_York'):
    """Check if current UTC time falls within defined Killzones converted to specified timezone."""
    try:
        now_utc = datetime.now(pytz.utc)
        target_tz = pytz.timezone(tz_name)
        now_local = now_utc.astimezone(target_tz)
        current_time_local = now_local.time()

        # --- Define Killzones in Local Time (HH:MM) ---
        # IMPORTANT: Adjust these times based on your strategy and broker/market DST rules!
        # Example for New York (EST/EDT sensitive):
        # London Open KZ: 02:00 - 05:00 Local NY time
        # New York Open KZ: 08:00 - 11:00 Local NY time
        # London Close KZ: 10:00 - 12:00 Local NY time (overlaps NY morning)
        # Asian KZ: 20:00 - 00:00 Local NY time (previous evening to midnight)

        killzones = {
            "Asian KZ": (time(20, 0), time(23, 59, 59)), # Evening before
            "London Open KZ": (time(2, 0), time(5, 0)),
            "New York Open KZ": (time(8, 0), time(11, 0)),
            "London Close KZ": (time(10, 0), time(12, 0))
        }

        # Check Asian session (spans midnight)
        if killzones["Asian KZ"][0] <= current_time_local <= killzones["Asian KZ"][1]:
             return True, "Asian KZ"
        # Check other sessions
        for name, (start, end) in killzones.items():
             if name == "Asian KZ": continue # Already checked
             if start <= current_time_local < end:
                  # Handle London Close / NY Open overlap: prioritize NY Open if active
                  if name == "London Close KZ" and killzones["New York Open KZ"][0] <= current_time_local < killzones["New York Open KZ"][1]:
                       continue # Skip London Close if during NY Open KZ
                  return True, name

        return False, None # Not in any defined killzone

    except Exception as e:
        logger.warning(f"Error checking killzone: {e}", exc_info=True)
        return False, None # Default to false if error


# --- Main Trading Loop ---
def run_crt_trading_agent():
    """Main function to run the CRT trading agent."""
    logger.info(f"Starting CRT Trading Agent for {SYMBOL}...")
    logger.info(f"Parameters: HTF={HIGHER_TIMEFRAME}, LTF={LOWER_TIMEFRAME}, Risk={RISK_PERCENT}%, MaxTrades={MAX_TRADES}, Interval={CHECK_INTERVAL}s, Backtest={BACKTEST_MODE}, UseKillzones={USE_KILLZONES}, Magic={MAGIC_NUMBER}")

    while True:
        try:
            # --- Connection Check ---
            if not mt5.terminal_info():
                logger.error("MT5 terminal disconnected. Attempting to reconnect...")
                if not initialize_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH):
                    logger.error("Reconnection failed. Sleeping...")
                    time_module.sleep(CHECK_INTERVAL * 2)
                    continue
                else:
                    logger.info("Reconnected to MT5 successfully.")

            # --- Killzone Check ---
            zone_name = None # Initialize zone_name
            if USE_KILLZONES:
                in_killzone, zone_name = is_in_killzone()
                if not in_killzone:
                    logger.debug(f"Not currently in a killzone. Waiting...")
                    time_module.sleep(CHECK_INTERVAL)
                    continue
                else:
                    logger.info(f"Currently in {zone_name}.")

            # --- Account and Position Check ---
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning(f"Failed to get account info: {mt5.last_error()}. Retrying...")
                time_module.sleep(CHECK_INTERVAL / 2)
                continue
            balance = account_info.balance
            currency = account_info.currency
            logger.info(f"Account Balance: {balance:.2f} {currency}")

            risk_amount = balance * (RISK_PERCENT / 100.0)
            logger.debug(f"Risk Amount per Trade: {risk_amount:.2f} {currency}")

            positions = mt5.positions_get(symbol=SYMBOL)
            if positions is None:
                logger.warning(f"Could not get positions for {SYMBOL}: {mt5.last_error()}. Assuming 0 positions.")
                positions = []
            else:
                # Filter positions by magic number if needed
                my_positions = [p for p in positions if p.magic == MAGIC_NUMBER]
                num_positions = len(my_positions)
                logger.info(f"Current open positions for {SYMBOL} (Magic: {MAGIC_NUMBER}): {num_positions}")

            if num_positions >= MAX_TRADES:
                logger.info(f"Maximum number of trades ({MAX_TRADES}) reached for {SYMBOL}. Monitoring...")
                # Optional: Add logic here to manage existing trades (e.g., trailing stops)
                time_module.sleep(CHECK_INTERVAL)
                continue

            # --- Data Fetching ---
            logger.info("Fetching latest market data...")
            df_high = get_ohlc_data(SYMBOL, HIGHER_TIMEFRAME, 250) # Fetch more for analysis context
            df_low = get_ohlc_data(SYMBOL, LOWER_TIMEFRAME, 150) # Fetch LTF data

            if df_high is None or df_high.empty:
                logger.warning("Failed to get sufficient high timeframe data. Retrying...")
                time_module.sleep(CHECK_INTERVAL)
                continue
            if df_low is None or df_low.empty:
                 logger.warning("Failed to get low timeframe data (required for future LTF entry logic).")
                 # Continue for now as CRT logic doesn't use it yet for entry


            # --- Analysis ---
            logger.info("Analyzing market data...")
            structure = analyze_market_structure(df_high)
            order_blocks = find_order_blocks(df_high, structure)
            fvgs = find_fair_value_gaps(df_high)
            key_levels = find_key_levels(df_high, n_levels=12) # Get slightly more levels
            patterns = find_crt_patterns(df_high, df_low, key_levels)

            if not patterns:
                logger.info("No potential CRT patterns found in the latest data.")
            else:
                logger.info(f"Found {len(patterns)} potential CRT patterns. Filtering...")

                # --- Filtering and Trade Selection ---
                valid_patterns = []
                for p in patterns:
                    # Core Filters: High probability (close inside range) AND near a key level
                    is_valid = p['high_probability'] and p['near_key_level']
                    if not is_valid: continue

                    # --- Optional Confluence Filters (Uncomment to enable) ---

                    # 1. Market Structure Alignment Filter
                    # structure_aligned = False
                    # if p['type'] == 'bullish' and structure['current_structure'] in ['bullish', 'ranging_LH_HL', 'breaking_high']:
                    #      structure_aligned = True
                    # elif p['type'] == 'bearish' and structure['current_structure'] in ['bearish', 'ranging_HH_LL', 'breaking_low']:
                    #      structure_aligned = True
                    # if not structure_aligned:
                    #      logger.debug(f"Pattern skipped: Type {p['type']} misaligned with structure {structure['current_structure']}")
                    #      continue # Skip if structure doesn't align

                    # 2. Order Block / FVG Confluence Filter
                    # confluence_found = False
                    # entry = p['entry_price']
                    # digits = mt5.symbol_info(SYMBOL).digits if mt5.symbol_info(SYMBOL) else 5
                    # if p['type'] == 'bullish':
                    #      # Check if manipulation low tapped into recent Bullish OB or FVG
                    #      for ob in order_blocks['bullish'][-3:]: # Check last 3 bullish OBs
                    #           if ob['low'] <= entry <= ob['high']:
                    #                confluence_found = True; logger.debug(f"Bullish CRT confluence with Bullish OB: {ob['time']}") ; break
                    #      if not confluence_found:
                    #           for fvg in fvgs['bullish'][-3:]: # Check last 3 bullish FVGs
                    #                if fvg['bottom'] <= entry <= fvg['top']:
                    #                     confluence_found = True; logger.debug(f"Bullish CRT confluence with Bullish FVG"); break
                    # elif p['type'] == 'bearish':
                    #       # Check if manipulation high tapped into recent Bearish OB or FVG
                    #      for ob in order_blocks['bearish'][-3:]:
                    #           if ob['low'] <= entry <= ob['high']:
                    #                confluence_found = True; logger.debug(f"Bearish CRT confluence with Bearish OB: {ob['time']}"); break
                    #      if not confluence_found:
                    #           for fvg in fvgs['bearish'][-3:]:
                    #                if fvg['bottom'] <= entry <= fvg['top']:
                    #                     confluence_found = True; logger.debug(f"Bearish CRT confluence with Bearish FVG"); break
                    #
                    # if not confluence_found:
                    #      logger.debug(f"Pattern skipped: No OB/FVG confluence found near entry {entry:.{digits}f}")
                    #      continue # Skip if no confluence with recent OB/FVG


                    # If all filters passed
                    valid_patterns.append(p)

                # --- Trade Execution ---
                if not valid_patterns:
                    logger.info("No patterns passed the filtering criteria.")
                else:
                    logger.info(f"Found {len(valid_patterns)} valid CRT patterns after filtering.")

                    # Select the most recent valid pattern to trade
                    # Assumes patterns list is ordered chronologically (oldest first)
                    pattern_to_trade = valid_patterns[-1]
                    logger.info(f"Selected pattern to trade (Time: {pattern_to_trade['time']}): Type={pattern_to_trade['type']}, Entry={pattern_to_trade['entry_price']:.5f}")
                    logger.debug(f"Selected Pattern Details: {pattern_to_trade}")

                    order_type = 'buy' if pattern_to_trade['type'] == 'bullish' else 'sell'
                    entry_price = pattern_to_trade['entry_price'] # Ideal entry point
                    stop_loss = pattern_to_trade['stop_loss']
                    take_profit = pattern_to_trade['take_profit']

                    # Calculate SL distance in pips
                    sl_distance_points = abs(entry_price - stop_loss)
                    symbol_info = mt5.symbol_info(SYMBOL)
                    point_size = symbol_info.point if symbol_info else 0.00001
                    digits = symbol_info.digits if symbol_info else 5
                    points_per_pip = 10 if digits in [3, 5] else 1
                    sl_distance_pips = (sl_distance_points / point_size) / points_per_pip if point_size > 0 else 0

                    logger.info(f"Trade Params: SL Distance={sl_distance_points:.{digits}f} points ({sl_distance_pips:.2f} pips)")

                    if sl_distance_pips <= 0.1: # Add a minimum pip distance check (e.g., 0.1 pips)
                         logger.warning(f"Calculated SL distance in pips ({sl_distance_pips:.2f}) is too small. Skipping trade.")
                    else:
                        lot_size = calculate_lot_size(SYMBOL, risk_amount, sl_distance_pips)

                        if lot_size <= 0:
                             logger.warning(f"Calculated lot size is zero or invalid ({lot_size}). Skipping trade.")
                        else:
                            logger.info(f"Attempting to execute {order_type} trade for {SYMBOL}. Lot Size: {lot_size}")

                            if not BACKTEST_MODE:
                                trade_comment = f"CRT {pattern_to_trade['type']} {zone_name if zone_name else ''}".strip()
                                trade_result = place_order(
                                    symbol=SYMBOL,
                                    order_type=order_type,
                                    lot_size=lot_size,
                                    entry_price=entry_price, # Pass ideal entry for logging
                                    sl_price=stop_loss,
                                    tp_price=take_profit,
                                    comment=trade_comment
                                )

                                if trade_result:
                                    logger.info(f"Trade executed successfully: {trade_result}")
                                    # Prevent rapid-fire trades after success
                                    logger.info(f"Sleeping for {CHECK_INTERVAL * 2}s after successful trade placement...")
                                    time_module.sleep(CHECK_INTERVAL * 2)
                                else:
                                    logger.error("Trade execution failed.")
                                    # Optional: Add a shorter delay after failed attempt
                                    time_module.sleep(CHECK_INTERVAL / 2)
                            else:
                                logger.info("[Backtest Mode] Trade would be placed here:")
                                logger.info(f"  Type: {order_type}, Lots: {lot_size}, Entry: {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
                                # In backtest, might want to break or advance time simulatedly

            # --- End of Cycle ---
            logger.info(f"Waiting {CHECK_INTERVAL} seconds for the next cycle...")
            time_module.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Exiting trading loop...")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main trading loop: {e}", exc_info=True)
            logger.info(f"Sleeping for {CHECK_INTERVAL * 2} seconds after error...")
            time_module.sleep(CHECK_INTERVAL * 2) # Longer sleep after error


    # --- Shutdown ---
    logger.info("Shutting down MT5 connection.")
    mt5.shutdown()
    logger.info("MT5 connection closed.")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Script started.")
    if initialize_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH):
        run_crt_trading_agent()
    else:
        logger.error("Failed to initialize MT5. Exiting.")
    logger.info("Script finished.")