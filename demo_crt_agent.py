import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# --- Configuration ---
SYMBOL = "EURUSD"
RISK_PERCENT = 1.0
BACKTEST_MODE = True

# --- Account Settings ---
# Deriv-Demo Account
DERIV_LOGIN = 5715737
DERIV_PASSWORD = "189@Kab@rNet@189"
DERIV_SERVER = "Deriv-Demo"

# FBS-Demo Account
FBS_LOGIN = 101310805
FBS_PASSWORD = "[lt5z@UJ"
FBS_SERVER = "FBS-Demo"

# Set which account to use for this run
USE_ACCOUNT = "FBS"  # Options: "DERIV" or "FBS"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Generate Sample Data ---
def generate_sample_data(count=100, timeframe="H4"):
    """Generate sample OHLC data for demonstration."""
    logger.info(f"Generating sample {timeframe} data for {SYMBOL}...")

    # Start date
    end_date = datetime.now()
    if timeframe == "H4":
        start_date = end_date - timedelta(hours=4*count)
        freq = "4H"
    elif timeframe == "H1":
        start_date = end_date - timedelta(hours=count)
        freq = "H"
    elif timeframe == "M15":
        start_date = end_date - timedelta(minutes=15*count)
        freq = "15min"
    else:
        start_date = end_date - timedelta(hours=4*count)
        freq = "4H"

    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Generate random price data
    np.random.seed(42)  # For reproducibility

    # Start with a base price
    base_price = 1.10000

    # Generate random changes
    changes = np.random.normal(0, 0.0010, len(dates))

    # Calculate prices with a slight upward trend
    trend = np.linspace(0, 0.005, len(dates))
    prices = base_price + np.cumsum(changes) + trend

    # Generate OHLC data
    data = []
    for i, date in enumerate(dates):
        price = prices[i]
        high = price + abs(np.random.normal(0, 0.0005))
        low = price - abs(np.random.normal(0, 0.0005))
        open_price = price - np.random.normal(0, 0.0003)
        close_price = price + np.random.normal(0, 0.0003)

        # Ensure high is highest and low is lowest
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)

        data.append({
            'time': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'tick_volume': int(np.random.uniform(100, 1000)),
            'spread': int(np.random.uniform(1, 5)),
            'real_volume': int(np.random.uniform(1000, 10000))
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)

    # Add some specific patterns for CRT detection
    # This ensures we'll find some patterns in our demo
    if len(df) > 50:
        # Add a bullish CRT pattern
        i = 30
        df.iloc[i-2, df.columns.get_loc('high')] = base_price + 0.005
        df.iloc[i-2, df.columns.get_loc('low')] = base_price - 0.005
        df.iloc[i-2, df.columns.get_loc('close')] = base_price

        df.iloc[i-1, df.columns.get_loc('low')] = base_price - 0.008  # Manipulation low
        df.iloc[i-1, df.columns.get_loc('close')] = base_price - 0.003

        df.iloc[i, df.columns.get_loc('close')] = base_price + 0.002  # Distribution

        # Add a bearish CRT pattern
        j = 60
        df.iloc[j-2, df.columns.get_loc('high')] = base_price + 0.006
        df.iloc[j-2, df.columns.get_loc('low')] = base_price - 0.004
        df.iloc[j-2, df.columns.get_loc('close')] = base_price

        df.iloc[j-1, df.columns.get_loc('high')] = base_price + 0.009  # Manipulation high
        df.iloc[j-1, df.columns.get_loc('close')] = base_price + 0.003

        df.iloc[j, df.columns.get_loc('close')] = base_price - 0.002  # Distribution

    logger.info(f"Generated {len(df)} {timeframe} candles for {SYMBOL}.")
    return df

# --- Analysis Functions ---
def find_key_levels(df, n_levels=10):
    """Identify key price levels from the data."""
    logger.debug(f"Finding up to {n_levels} key levels...")
    if df is None or len(df) < 5:
        logger.warning("Not enough data to find key levels.")
        return []

    levels = []
    df_reset = df.reset_index()

    # Find swing highs and lows
    logger.debug("Detecting swing points...")
    for i in range(2, len(df_reset)-2):
        # Swing high
        is_swing_high = (df_reset.iloc[i]['high'] > df_reset.iloc[i-1]['high'] and
                         df_reset.iloc[i]['high'] > df_reset.iloc[i-2]['high'] and
                         df_reset.iloc[i]['high'] > df_reset.iloc[i+1]['high'] and
                         df_reset.iloc[i]['high'] > df_reset.iloc[i+2]['high'])
        if is_swing_high:
            levels.append(df_reset.iloc[i]['high'])

        # Swing low
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

    # Round levels
    levels = [round(level, 5) for level in levels]

    # Remove duplicates and sort
    levels = sorted(list(set(levels)))
    logger.debug(f"Raw levels identified: {levels}")

    # Simplify by grouping nearby levels
    if len(levels) > 1:
        simplified_levels = [levels[0]]
        for level in levels[1:]:
            # If this level is close to the last one kept, skip it
            threshold = 0.0005
            if abs(level - simplified_levels[-1]) / max(abs(level), abs(simplified_levels[-1]), 1e-9) < threshold:
                continue
            simplified_levels.append(level)
        levels = simplified_levels
        logger.debug(f"Simplified levels: {levels}")

    # Return the most relevant n_levels
    final_levels = levels[-n_levels:]

    logger.info(f"Identified key levels: {final_levels}")
    return final_levels

def analyze_market_structure(df):
    """Analyze market structure to identify trends, swings, and potential reversals."""
    logger.debug("Analyzing market structure...")
    if df is None or len(df) < 5:
        logger.warning("Not enough data for market structure analysis.")
        return {'current_structure': 'unknown', 'swing_highs': [], 'swing_lows': []}

    df_copy = df.copy()

    # Find swing highs and lows
    df_copy['swing_high'] = False
    df_copy['swing_low'] = False

    logger.debug("Detecting swing points for structure analysis...")
    for i in range(2, len(df_copy)-2):
        # Swing high
        if (df_copy.iloc[i]['high'] > df_copy.iloc[i-1]['high'] and
            df_copy.iloc[i]['high'] > df_copy.iloc[i-2]['high'] and
            df_copy.iloc[i]['high'] > df_copy.iloc[i+1]['high'] and
            df_copy.iloc[i]['high'] > df_copy.iloc[i+2]['high']):
            df_copy.loc[df_copy.index[i], 'swing_high'] = True

        # Swing low
        if (df_copy.iloc[i]['low'] < df_copy.iloc[i-1]['low'] and
            df_copy.iloc[i]['low'] < df_copy.iloc[i-2]['low'] and
            df_copy.iloc[i]['low'] < df_copy.iloc[i+1]['low'] and
            df_copy.iloc[i]['low'] < df_copy.iloc[i+2]['low']):
            df_copy.loc[df_copy.index[i], 'swing_low'] = True

    swing_highs_df = df_copy[df_copy['swing_high']]
    swing_lows_df = df_copy[df_copy['swing_low']]

    swing_highs = list(zip(swing_highs_df.index, swing_highs_df['high']))
    swing_lows = list(zip(swing_lows_df.index, swing_lows_df['low']))

    structure = {
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'current_structure': 'unknown'
    }

    # Determine structure based on recent swings
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_high_val = swing_highs[-1][1]
        prev_high_val = swing_highs[-2][1]
        last_low_val = swing_lows[-1][1]
        prev_low_val = swing_lows[-2][1]

        logger.debug(f"Recent Swings - Highs: {prev_high_val:.5f} -> {last_high_val:.5f}, Lows: {prev_low_val:.5f} -> {last_low_val:.5f}")

        if last_high_val > prev_high_val and last_low_val > prev_low_val:
            structure['current_structure'] = 'bullish'
        elif last_high_val < prev_high_val and last_low_val < prev_low_val:
            structure['current_structure'] = 'bearish'
        elif last_high_val < prev_high_val and last_low_val > prev_low_val:
            structure['current_structure'] = 'ranging_LH_HL'
        elif last_high_val > prev_high_val and last_low_val < prev_low_val:
            structure['current_structure'] = 'ranging_HH_LL'
        else:
            structure['current_structure'] = 'consolidating'

    logger.info(f"Determined market structure: {structure['current_structure']}")
    return structure

# --- CRT Pattern Detection ---
def find_crt_patterns(df_high_tf, df_low_tf, key_levels=None):
    """Identify potential CRT patterns in the data."""
    logger.debug("Finding CRT patterns...")
    patterns = []

    if df_high_tf is None or len(df_high_tf) < 4:
        logger.warning("Not enough high timeframe data for CRT pattern detection.")
        return patterns

    # Reset index for easier iloc access
    df = df_high_tf.reset_index()

    # Minimum number of candles to look back
    min_lookback = 3

    # Iterate through candles
    for i in range(min_lookback, len(df)-1):
        try:
            # Candle indices: Range = i-2, Manipulation = i-1, Distribution = i
            range_candle = df.iloc[i-2]
            manipulation_candle = df.iloc[i-1]
            distribution_candle = df.iloc[i]

            # --- Check for Bullish CRT Pattern ---
            if manipulation_candle['low'] < range_candle['low'] and manipulation_candle['close'] >= range_candle['low']:
                logger.debug(f"Potential Bullish CRT detected at {manipulation_candle['time']}")

                # Check if near key level if provided
                near_key_level = False
                if key_levels is not None:
                    for level in key_levels:
                        if abs(manipulation_candle['low'] - level) / max(abs(level), 1e-9) < 0.0015:
                            near_key_level = True
                            logger.debug(f"Bullish CRT near key level: {level:.5f}")
                            break

                # Calculate SL/TP
                range_height = range_candle['high'] - range_candle['low']
                stop_loss = manipulation_candle['low'] - (range_height * 0.1)
                take_profit = range_candle['high']
                entry_price = manipulation_candle['low']

                # Record the pattern
                patterns.append({
                    'type': 'bullish',
                    'time': manipulation_candle['time'],
                    'range_candle': range_candle.to_dict(),
                    'manipulation_candle': manipulation_candle.to_dict(),
                    'distribution_candle': distribution_candle.to_dict(),
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'high_probability': True,
                    'near_key_level': near_key_level
                })

            # --- Check for Bearish CRT Pattern ---
            if manipulation_candle['high'] > range_candle['high'] and manipulation_candle['close'] <= range_candle['high']:
                logger.debug(f"Potential Bearish CRT detected at {manipulation_candle['time']}")

                # Check if near key level if provided
                near_key_level = False
                if key_levels is not None:
                    for level in key_levels:
                        if abs(manipulation_candle['high'] - level) / max(abs(level), 1e-9) < 0.0015:
                            near_key_level = True
                            logger.debug(f"Bearish CRT near key level: {level:.5f}")
                            break

                # Calculate SL/TP
                range_height = range_candle['high'] - range_candle['low']
                stop_loss = manipulation_candle['high'] + (range_height * 0.1)
                take_profit = range_candle['low']
                entry_price = manipulation_candle['high']

                # Record the pattern
                patterns.append({
                    'type': 'bearish',
                    'time': manipulation_candle['time'],
                    'range_candle': range_candle.to_dict(),
                    'manipulation_candle': manipulation_candle.to_dict(),
                    'distribution_candle': distribution_candle.to_dict(),
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'high_probability': True,
                    'near_key_level': near_key_level
                })
        except Exception as e:
            logger.error(f"Error processing candle index {i}: {e}")
            continue

    if patterns:
        logger.info(f"Found {len(patterns)} potential CRT patterns.")
    else:
        logger.debug("No CRT patterns found in the current dataset.")

    return patterns

# --- Order Block Detection ---
def find_order_blocks(df, structure=None):
    """Identify potential order blocks in the data."""
    logger.debug("Finding order blocks...")
    order_blocks = {'bullish': [], 'bearish': []}

    if df is None or len(df) < 4:
        logger.warning("Not enough data for order block detection.")
        return order_blocks

    df_reset = df.reset_index()

    # Iterate through candles
    for i in range(3, len(df_reset)):
        try:
            # Check for bullish order block (bearish candle followed by strong bullish move)
            if (df_reset.iloc[i-2]['close'] < df_reset.iloc[i-2]['open'] and  # Bearish candle
                df_reset.iloc[i-1]['close'] > df_reset.iloc[i-1]['open'] and  # Bullish candle
                df_reset.iloc[i]['close'] > df_reset.iloc[i]['open'] and      # Bullish candle
                df_reset.iloc[i]['close'] > df_reset.iloc[i-1]['high']):      # Strong move up

                # The bearish candle is the bullish order block
                order_blocks['bullish'].append({
                    'time': df_reset.iloc[i-2]['time'],
                    'high': df_reset.iloc[i-2]['high'],
                    'low': df_reset.iloc[i-2]['low'],
                    'open': df_reset.iloc[i-2]['open'],
                    'close': df_reset.iloc[i-2]['close']
                })
                logger.debug(f"Bullish order block found at {df_reset.iloc[i-2]['time']}")

            # Check for bearish order block (bullish candle followed by strong bearish move)
            if (df_reset.iloc[i-2]['close'] > df_reset.iloc[i-2]['open'] and  # Bullish candle
                df_reset.iloc[i-1]['close'] < df_reset.iloc[i-1]['open'] and  # Bearish candle
                df_reset.iloc[i]['close'] < df_reset.iloc[i]['open'] and      # Bearish candle
                df_reset.iloc[i]['close'] < df_reset.iloc[i-1]['low']):       # Strong move down

                # The bullish candle is the bearish order block
                order_blocks['bearish'].append({
                    'time': df_reset.iloc[i-2]['time'],
                    'high': df_reset.iloc[i-2]['high'],
                    'low': df_reset.iloc[i-2]['low'],
                    'open': df_reset.iloc[i-2]['open'],
                    'close': df_reset.iloc[i-2]['close']
                })
                logger.debug(f"Bearish order block found at {df_reset.iloc[i-2]['time']}")

        except Exception as e:
            logger.error(f"Error processing candle index {i} for order blocks: {e}")
            continue

    logger.info(f"Found {len(order_blocks['bullish'])} bullish order blocks and {len(order_blocks['bearish'])} bearish order blocks.")
    return order_blocks

# --- Fair Value Gap Detection ---
def find_fair_value_gaps(df):
    """Identify fair value gaps (FVGs) in the data."""
    logger.debug("Finding fair value gaps...")
    fvgs = {'bullish': [], 'bearish': []}

    if df is None or len(df) < 3:
        logger.warning("Not enough data for FVG detection.")
        return fvgs

    df_reset = df.reset_index()

    # Iterate through candles
    for i in range(2, len(df_reset)):
        try:
            candle_prev2 = df_reset.iloc[i-2]
            candle_prev1 = df_reset.iloc[i-1]
            candle_curr = df_reset.iloc[i]

            # Bullish FVG: Current candle's low is above previous candle's high
            if candle_curr['low'] > candle_prev1['high']:
                fvgs['bullish'].append({
                    'time': candle_curr['time'],
                    'bottom': candle_prev1['high'],
                    'top': candle_curr['low'],
                    'size': candle_curr['low'] - candle_prev1['high']
                })
                logger.debug(f"Bullish FVG found between {candle_prev1['time']} and {candle_curr['time']}")

            # Bearish FVG: Current candle's high is below previous candle's low
            if candle_curr['high'] < candle_prev1['low']:
                fvgs['bearish'].append({
                    'time': candle_curr['time'],
                    'top': candle_prev1['low'],
                    'bottom': candle_curr['high'],
                    'size': candle_prev1['low'] - candle_curr['high']
                })
                logger.debug(f"Bearish FVG found between {candle_prev1['time']} and {candle_curr['time']}")

        except Exception as e:
            logger.error(f"Error processing candle index {i} for FVGs: {e}")
            continue

    logger.info(f"Found {len(fvgs['bullish'])} bullish FVGs and {len(fvgs['bearish'])} bearish FVGs.")
    return fvgs

# --- Risk Management ---
def calculate_position_size(account_balance, risk_percent, sl_pips):
    """Calculate position size based on risk parameters."""
    if sl_pips <= 0:
        logger.warning("Stop loss distance must be positive.")
        return 0.01

    # For demo purposes, assume 1 pip = $10 per standard lot
    pip_value = 10

    # Calculate risk amount
    risk_amount = account_balance * (risk_percent / 100)

    # Calculate lot size
    lot_size = risk_amount / (sl_pips * pip_value)

    # Round to 2 decimal places (0.01 lot precision)
    lot_size = round(lot_size, 2)

    # Ensure minimum lot size
    lot_size = max(lot_size, 0.01)

    logger.info(f"Calculated position size: {lot_size} lots (Risk: ${risk_amount:.2f}, SL: {sl_pips:.1f} pips)")
    return lot_size

# --- Main Function ---
def main():
    # Set account details based on selection
    if USE_ACCOUNT == "DERIV":
        account_login = DERIV_LOGIN
        account_password = DERIV_PASSWORD
        account_server = DERIV_SERVER
        account_balance = 10000.0  # Example balance for Deriv account
    else:  # FBS
        account_login = FBS_LOGIN
        account_password = FBS_PASSWORD
        account_server = FBS_SERVER
        account_balance = 5000.0  # Example balance for FBS account

    logger.info(f"Starting Demo CRT Agent for {SYMBOL}...")
    logger.info(f"Using account: {USE_ACCOUNT} (Login: {account_login}, Server: {account_server})")

    try:
        # Generate sample data
        df_high = generate_sample_data(count=100, timeframe="H4")
        df_low = generate_sample_data(count=100, timeframe="M15")

        # Analyze data
        logger.info("Analyzing market data...")
        key_levels = find_key_levels(df_high, n_levels=10)
        structure = analyze_market_structure(df_high)
        order_blocks = find_order_blocks(df_high, structure)
        fvgs = find_fair_value_gaps(df_high)
        patterns = find_crt_patterns(df_high, df_low, key_levels)

        if not patterns:
            logger.info("No potential CRT patterns found in the sample data.")
        else:
            # Filter patterns
            valid_patterns = []
            for p in patterns:
                # Core Filters: High probability (close inside range) AND near a key level
                is_valid = p['high_probability'] and p['near_key_level']
                if is_valid:
                    valid_patterns.append(p)

            if not valid_patterns:
                logger.info("No patterns passed the filtering criteria.")
            else:
                logger.info(f"Found {len(valid_patterns)} valid CRT patterns after filtering.")

                # Display patterns
                for i, pattern in enumerate(valid_patterns, 1):
                    logger.info(f"Pattern {i}:")
                    logger.info(f"  Type: {pattern['type'].upper()}")
                    logger.info(f"  Time: {pattern['time']}")
                    logger.info(f"  Entry: {pattern['entry_price']:.5f}")
                    logger.info(f"  Stop Loss: {pattern['stop_loss']:.5f}")
                    logger.info(f"  Take Profit: {pattern['take_profit']:.5f}")

                    # Calculate risk-reward ratio
                    rr_ratio = abs(pattern['take_profit'] - pattern['entry_price']) / abs(pattern['stop_loss'] - pattern['entry_price'])
                    logger.info(f"  Risk-Reward Ratio: {rr_ratio:.2f}")

                    # Calculate position size
                    sl_pips = abs(pattern['entry_price'] - pattern['stop_loss']) * 10000  # Convert to pips
                    lot_size = calculate_position_size(account_balance, RISK_PERCENT, sl_pips)

                    logger.info(f"  Position Size: {lot_size} lots")
                    logger.info(f"  Potential Profit: ${abs(pattern['take_profit'] - pattern['entry_price']) * 10000 * lot_size * 10:.2f}")
                    logger.info(f"  Potential Loss: ${abs(pattern['stop_loss'] - pattern['entry_price']) * 10000 * lot_size * 10:.2f}")
                    logger.info("")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    logger.info("Demo completed.")

if __name__ == "__main__":
    main()
