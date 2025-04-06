"""
Run the CRT agent with MT5 connection
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import sys
import os

# Import configuration
from config import *

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

def initialize_mt5():
    """Initialize connection to MT5 terminal."""
    logger.info("Initializing MT5 connection...")

    # First try to initialize without parameters
    logger.info("Attempting basic initialization...")
    if not mt5.initialize():
        logger.warning(f"Basic initialization failed, error code = {mt5.last_error()}")
        logger.info("Trying with parameters...")

        init_params = {
            "login": MT5_LOGIN,
            "password": MT5_PASSWORD,
            "server": MT5_SERVER,
        }

        # Attempt to initialize connection
        if not mt5.initialize(**init_params):
            logger.error(f"initialize() failed, error code = {mt5.last_error()}")
            return False
    else:
        logger.info("Basic initialization successful.")

        # Now try to login
        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            logger.info(f"Attempting login to {MT5_SERVER} with account {MT5_LOGIN}...")
            if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
                logger.error(f"Login failed, error code = {mt5.last_error()}")
                mt5.shutdown()
                return False
            logger.info("Login successful.")

    # Get account info
    account_info = mt5.account_info()
    if account_info:
        logger.info(f"Connected to account: {account_info.login} on {account_info.server}")
        logger.info(f"Balance: {account_info.balance} {account_info.currency}")
    else:
        logger.error(f"Failed to get account info: {mt5.last_error()}")
        mt5.shutdown()
        return False

    return True

def get_ohlc_data(symbol, timeframe, count=100):
    """Retrieve OHLC data for the specified symbol and timeframe."""
    logger.debug(f"Fetching {count} candles for {symbol} on {timeframe}")
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to get rates for {symbol} on {timeframe}: {mt5.last_error()}")
            return None

        # Convert to pandas DataFrame for easier analysis
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        logger.debug(f"Successfully fetched {len(df)} candles.")
        return df
    except Exception as e:
        logger.error(f"Error getting OHLC data: {e}")
        return None

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

def find_crt_patterns(df_high_tf, df_low_tf=None, key_levels=None):
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

    # Get the number of decimal places for the instrument
    # For forex, typically 5 decimal places for most pairs, 3 for JPY pairs
    sample_price = df['close'].iloc[-1]
    digits = 5 if sample_price > 1 else 3

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
                            logger.debug(f"Bullish CRT manip low {manipulation_candle['low']:.{digits}f} near key level: {level:.{digits}f}")
                            break

                # Calculate SL/TP
                range_height = range_candle['high'] - range_candle['low']
                stop_loss = manipulation_candle['low'] - (range_height * 0.1)
                take_profit = range_candle['high']
                entry_price = manipulation_candle['low']

                # Calculate risk-reward ratio
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
                risk_reward_ratio = reward / risk if risk > 0 else 0

                # Record the pattern
                patterns.append({
                    'type': 'bullish',
                    'time': manipulation_candle['time'],
                    'range_candle': {
                        'time': range_candle['time'],
                        'open': range_candle['open'],
                        'high': range_candle['high'],
                        'low': range_candle['low'],
                        'close': range_candle['close']
                    },
                    'manipulation_candle': {
                        'time': manipulation_candle['time'],
                        'open': manipulation_candle['open'],
                        'high': manipulation_candle['high'],
                        'low': manipulation_candle['low'],
                        'close': manipulation_candle['close']
                    },
                    'distribution_candle': {
                        'time': distribution_candle['time'],
                        'open': distribution_candle['open'],
                        'high': distribution_candle['high'],
                        'low': distribution_candle['low'],
                        'close': distribution_candle['close']
                    },
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward_ratio': risk_reward_ratio,
                    'near_key_level': near_key_level,
                    'confidence': 85
                })

                logger.info(f"Bullish CRT detected at {manipulation_candle['time']} - Entry: {entry_price:.{digits}f}, SL: {stop_loss:.{digits}f}, TP: {take_profit:.{digits}f}, RR: {risk_reward_ratio:.2f}")

            # --- Check for Bearish CRT Pattern ---
            if manipulation_candle['high'] > range_candle['high'] and manipulation_candle['close'] <= range_candle['high']:
                logger.debug(f"Potential Bearish CRT detected at {manipulation_candle['time']}")

                # Check if near key level if provided
                near_key_level = False
                if key_levels is not None:
                    for level in key_levels:
                        if abs(manipulation_candle['high'] - level) / max(abs(level), 1e-9) < 0.0015:
                            near_key_level = True
                            logger.debug(f"Bearish CRT manip high {manipulation_candle['high']:.{digits}f} near key level: {level:.{digits}f}")
                            break

                # Calculate SL/TP
                range_height = range_candle['high'] - range_candle['low']
                stop_loss = manipulation_candle['high'] + (range_height * 0.1)
                take_profit = range_candle['low']
                entry_price = manipulation_candle['high']

                # Calculate risk-reward ratio
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                risk_reward_ratio = reward / risk if risk > 0 else 0

                # Record the pattern
                patterns.append({
                    'type': 'bearish',
                    'time': manipulation_candle['time'],
                    'range_candle': {
                        'time': range_candle['time'],
                        'open': range_candle['open'],
                        'high': range_candle['high'],
                        'low': range_candle['low'],
                        'close': range_candle['close']
                    },
                    'manipulation_candle': {
                        'time': manipulation_candle['time'],
                        'open': manipulation_candle['open'],
                        'high': manipulation_candle['high'],
                        'low': manipulation_candle['low'],
                        'close': manipulation_candle['close']
                    },
                    'distribution_candle': {
                        'time': distribution_candle['time'],
                        'open': distribution_candle['open'],
                        'high': distribution_candle['high'],
                        'low': distribution_candle['low'],
                        'close': distribution_candle['close']
                    },
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward_ratio': risk_reward_ratio,
                    'near_key_level': near_key_level,
                    'confidence': 85
                })

                logger.info(f"Bearish CRT detected at {manipulation_candle['time']} - Entry: {entry_price:.{digits}f}, SL: {stop_loss:.{digits}f}, TP: {take_profit:.{digits}f}, RR: {risk_reward_ratio:.2f}")

        except Exception as e:
            logger.error(f"Error processing candle index {i}: {e}")
            continue

    logger.info(f"Found {len(patterns)} CRT patterns")
    return patterns

def generate_signals(patterns):
    """Generate trading signals based on CRT patterns"""
    logger.info(f"Generating signals from {len(patterns)} patterns...")

    signals = []
    for pattern in patterns:
        signal = {
            'type': pattern['type'],
            'time': pattern['time'],
            'entry': pattern['entry_price'],
            'stop_loss': pattern['stop_loss'],
            'take_profit': pattern['take_profit'],
            'risk_reward': pattern['risk_reward_ratio'],
            'confidence': pattern['confidence'],
            'action': 'BUY' if pattern['type'] == 'bullish' else 'SELL'
        }
        signals.append(signal)

    logger.info(f"Generated {len(signals)} signals")
    return signals

def calculate_lot_size(symbol, risk_amount, sl_distance_pips):
    """Calculate appropriate lot size based on risk parameters."""
    logger.debug(f"Calculating lot size for {symbol}. Risk Amount: {risk_amount:.2f}, SL Pips: {sl_distance_pips}")

    min_lot = 0.01

    try:
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found, cannot calculate lot size.")
            return min_lot

        # Update min_lot based on actual symbol info
        min_lot = symbol_info.volume_min

        # Check for zero SL distance
        if sl_distance_pips <= 0:
            logger.warning(f"Stop loss distance is zero or negative ({sl_distance_pips}). Cannot calculate lot size.")
            return min_lot

        # Get contract size, tick value, tick size, digits, volume step
        contract_size = symbol_info.trade_contract_size
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        digits = symbol_info.digits
        lot_step = symbol_info.volume_step

        # Calculate value per pip
        point_value = tick_size * (10**(digits))
        points_per_pip = 10 if digits == 5 or digits == 3 else 1

        value_per_point = tick_value / tick_size if tick_size > 0 else 0
        pip_value_per_lot = value_per_point * points_per_pip

        if pip_value_per_lot <= 0:
            logger.error(f"Calculated pip value per lot is zero or negative. Cannot calculate lot size.")
            return min_lot

        # Calculate lot size
        lot_size = risk_amount / (sl_distance_pips * pip_value_per_lot)
        logger.debug(f"Raw calculated lot size: {lot_size}")

        # Round to broker's lot step
        lot_size = round(lot_size / lot_step) * lot_step

        # Ensure lot size is within allowed range
        if lot_size < symbol_info.volume_min:
            logger.warning(f"Calculated lot size {lot_size} is below minimum {symbol_info.volume_min}. Adjusting to minimum.")
            lot_size = symbol_info.volume_min
        elif lot_size > symbol_info.volume_max:
            logger.warning(f"Calculated lot size {lot_size} exceeds maximum {symbol_info.volume_max}. Adjusting to maximum.")
            lot_size = symbol_info.volume_max

        logger.info(f"Calculated Lot Size for {symbol}: {lot_size:.2f}")
        return lot_size

    except Exception as e:
        logger.error(f"Error calculating lot size: {e}")
        return min_lot

def execute_trade(signal, symbol):
    """Execute a trade based on the signal"""
    if BACKTEST_MODE:
        logger.info(f"[BACKTEST] Would execute {signal['action']} for {symbol} at {signal['entry']}")
        return True

    try:
        # Check if market is open
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
            return False

        if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            logger.warning(f"Market for {symbol} is not open for trading. Current trade mode: {symbol_info.trade_mode}")
            return False

        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return False

        # Calculate risk amount
        risk_amount = account_info.balance * (RISK_PERCENT / 100)

        # Calculate SL distance in pips
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
            return False

        digits = symbol_info.digits
        point = symbol_info.point

        # Get minimum stop level in points
        min_stop_level = symbol_info.trade_stops_level
        if min_stop_level == 0:
            min_stop_level = 20  # Default to 20 points if not specified by broker

        # Convert to price value
        min_stop_distance = min_stop_level * point
        logger.info(f"Minimum stop distance for {symbol}: {min_stop_distance}")

        sl_distance = abs(signal['entry'] - signal['stop_loss'])
        sl_distance_pips = sl_distance / (point * 10)

        # Calculate lot size
        lot_size = calculate_lot_size(symbol, risk_amount, sl_distance_pips)

        # Get current market price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get current price for {symbol}: {mt5.last_error()}")
            return False

        # For Deriv broker, we need to omit the filling mode completely
        # as they don't support the standard MT5 filling modes
        logger.info(f"Preparing order for {symbol} without specifying filling mode (Deriv broker requirement)")

        # Get current tick data
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick data for {symbol}: {mt5.last_error()}")
            return False

        # Prepare trade request
        if signal['action'] == 'BUY':
            # For BUY orders: SL must be below current price, TP must be above current price
            # Ensure SL is at least min_stop_distance below current price
            current_price = tick.ask

            # For BUY orders, SL must be below current price
            # Ensure SL is at least min_stop_distance * 5 below current price for safety
            adjusted_sl = min(signal['stop_loss'], current_price - (min_stop_distance * 5))

            # Ensure TP is at least min_stop_distance * 5 above current price for safety
            adjusted_tp = max(signal['take_profit'], current_price + (min_stop_distance * 5))

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": current_price,
                "sl": adjusted_sl,
                "tp": adjusted_tp,
                "deviation": 10,
                "magic": MAGIC_NUMBER,
                "comment": "CRT Agent BUY",
                "type_time": mt5.ORDER_TIME_GTC,
                # No filling type for Deriv broker
            }
            logger.info(f"BUY {symbol} at {current_price:.5f} - Adjusted SL from {signal['stop_loss']:.5f} to {adjusted_sl:.5f} and TP from {signal['take_profit']:.5f} to {adjusted_tp:.5f}")
        else:  # SELL
            # For SELL orders: SL must be above current price, TP must be below current price
            # Ensure SL is at least min_stop_distance above current price
            current_price = tick.bid

            # For SELL orders, SL must be above current price
            # Ensure SL is at least min_stop_distance * 5 above current price for safety
            adjusted_sl = max(signal['stop_loss'], current_price + (min_stop_distance * 5))

            # Ensure TP is at least min_stop_distance * 5 below current price for safety
            adjusted_tp = min(signal['take_profit'], current_price - (min_stop_distance * 5))

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "sl": adjusted_sl,
                "tp": adjusted_tp,
                "deviation": 10,
                "magic": MAGIC_NUMBER,
                "comment": "CRT Agent SELL",
                "type_time": mt5.ORDER_TIME_GTC,
                # No filling type for Deriv broker
            }
            logger.info(f"SELL {symbol} at {current_price:.5f} - Adjusted SL from {signal['stop_loss']:.5f} to {adjusted_sl:.5f} and TP from {signal['take_profit']:.5f} to {adjusted_tp:.5f}")

        # Send the trade request
        result = mt5.order_send(request)

        # Check the result
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed, retcode={result.retcode}")
            logger.error(f"   {result.comment}")

            # If error is about AutoTrading being disabled, provide instructions
            if result.retcode == 10027:  # AutoTrading disabled by client
                logger.error("Please enable AutoTrading in MetaTrader 5:")
                logger.error("1. Open MetaTrader 5")
                logger.error("2. Click the 'AutoTrading' button in the toolbar (or press F12)")
                logger.error("3. Make sure the button is highlighted, indicating AutoTrading is enabled")

            return False

        logger.info(f"Order executed successfully: {signal['action']} {lot_size} {symbol} at {request['price']}")
        logger.info(f"   SL: {signal['stop_loss']}, TP: {signal['take_profit']}")
        return True

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def get_available_symbols():
    """Get a list of available symbols from the broker"""
    logger.info("Getting available symbols from broker...")
    symbols_info = mt5.symbols_get()
    if symbols_info is None:
        logger.error(f"Failed to get symbols: {mt5.last_error()}")
        return []

    # Extract symbol names
    available_symbols = [symbol.name for symbol in symbols_info]
    logger.info(f"Found {len(available_symbols)} available symbols")
    return available_symbols

def filter_active_symbols(available_symbols):
    """Filter the list of symbols to trade based on what's available"""
    global ACTIVE_SYMBOLS

    if MULTI_SYMBOL_MODE:
        # Filter symbols that are available in the broker
        ACTIVE_SYMBOLS = [symbol for symbol in SYMBOLS if symbol in available_symbols]
        logger.info(f"Trading {len(ACTIVE_SYMBOLS)} symbols in multi-symbol mode")
    else:
        # Single symbol mode
        if SYMBOL in available_symbols:
            ACTIVE_SYMBOLS = [SYMBOL]
            logger.info(f"Trading single symbol: {SYMBOL}")
        else:
            logger.error(f"Symbol {SYMBOL} not available in this broker")
            ACTIVE_SYMBOLS = []

    return ACTIVE_SYMBOLS

def run_crt_agent(specific_symbol=None):
    """Run the CRT agent

    Args:
        specific_symbol (str, optional): Symbol to process. If None, uses config settings.
    """
    logger.info("Starting CRT Agent...")

    try:
        # Initialize MT5
        if not initialize_mt5():
            logger.error("Failed to initialize MT5. Exiting.")
            return False

        # Process specific symbol if provided
        if specific_symbol:
            logger.info(f"Processing specific symbol: {specific_symbol}")
            symbols_to_process = [specific_symbol]
        else:
            # Get available symbols
            available_symbols = get_available_symbols()

            # Filter active symbols
            symbols_to_process = filter_active_symbols(available_symbols)

            if not symbols_to_process:
                logger.error("No active symbols to trade. Exiting.")
                mt5.shutdown()
                return False

        # Track total signals and patterns
        total_patterns = 0
        total_signals = 0

        # Process each symbol
        for symbol in symbols_to_process:
            logger.info(f"Processing {symbol}...")

            # Get data
            logger.info(f"Getting data for {symbol}...")
            df_high = get_ohlc_data(symbol, HIGHER_TIMEFRAME, 100)
            df_low = get_ohlc_data(symbol, LOWER_TIMEFRAME, 100)

            if df_high is None or df_low is None:
                logger.warning(f"Failed to get data for {symbol}. Skipping.")
                continue

            # Find key levels
            key_levels = find_key_levels(df_high)

            # Analyze market structure
            structure = analyze_market_structure(df_high)

            # Find CRT patterns
            patterns = find_crt_patterns(df_high, df_low, key_levels)
            total_patterns += len(patterns)

            # Generate signals
            signals = generate_signals(patterns)
            total_signals += len(signals)

            # Execute trades (limited by MAX_TRADES)
            open_positions = len(mt5.positions_get(symbol=symbol)) or 0
            available_slots = MAX_TRADES - open_positions

            if available_slots <= 0:
                logger.info(f"Maximum number of trades ({MAX_TRADES}) already open for {symbol}. Skipping execution.")
                continue

            # Sort signals by risk-reward ratio (highest first)
            signals.sort(key=lambda x: x['risk_reward'], reverse=True)

            # Execute only the best signals up to available slots
            for signal in signals[:available_slots]:
                execute_trade(signal, symbol)

        logger.info(f"Processed {len(symbols_to_process)} symbols, found {total_patterns} patterns and generated {total_signals} signals")

        # Shutdown MT5
        mt5.shutdown()
        logger.info("MT5 connection closed.")

    except Exception as e:
        logger.error(f"Error running CRT agent: {e}")
        mt5.shutdown()

if __name__ == "__main__":
    logger.info("Starting CRT Agent...")
    run_crt_agent()
