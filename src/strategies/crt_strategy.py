"""
🕯️ Candle Range Theory (CRT) Strategy

This strategy identifies CRT patterns based on the following criteria:
1. Range Candle: Forms the initial price range
2. Manipulation Candle: Breaks either above or below the range candle to "grab liquidity"
3. Distribution Candle: Moves in the opposite direction of the manipulation

The strategy detects two main pattern types:
- Bullish CRT: When price breaks below a range, then reverses upward
- Bearish CRT: When price breaks above a range, then reverses downward

Additional filters include:
- Key level proximity
- Market structure alignment
- Volume confirmation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CRTStrategy:
    def __init__(self, config=None):
        """Initialize the CRT strategy with optional configuration"""
        self.config = config or {}

        # Default configuration values
        self.default_config = {
            'min_range_height_pips': 5,   # Minimum height of range candle in pips
            'max_range_height_pips': 200, # Maximum height of range candle in pips
            'min_manipulation_pips': 3,   # Minimum manipulation distance in pips
            'key_level_proximity_pips': 30, # Maximum distance to consider "near" a key level
            'volume_threshold': 1.0,       # Volume multiplier threshold for confirmation
            'use_key_levels': False,       # Whether to filter by key levels
            'use_market_structure': False, # Whether to filter by market structure
            'use_volume_confirmation': False # Whether to use volume for confirmation
        }

        # Apply default config for any missing values
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        logger.info(f"CRT Strategy initialized with config: {self.config}")

    def find_key_levels(self, df, n_levels=10):
        """Identify key price levels from the data"""
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

    def analyze_market_structure(self, df):
        """Analyze market structure to identify trends, swings, and potential reversals"""
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

    def detect_crt_patterns(self, df, key_levels=None):
        """Detect CRT patterns in the provided data"""
        logger.info("Detecting CRT patterns...")

        if df is None or len(df) < 4:
            logger.warning("Not enough data for CRT pattern detection.")
            return []

        # Get market structure if needed
        market_structure = None
        if self.config['use_market_structure']:
            market_structure = self.analyze_market_structure(df)

        # Get key levels if needed and not provided
        if self.config['use_key_levels'] and key_levels is None:
            key_levels = self.find_key_levels(df)

        # Reset index for easier iloc access
        df = df.reset_index()

        # Store detected patterns
        patterns = []

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

                # Calculate range height in pips
                range_height = range_candle['high'] - range_candle['low']
                range_height_pips = range_height * (10**digits)

                # Skip if range is too small or too large
                if range_height_pips < self.config['min_range_height_pips'] or range_height_pips > self.config['max_range_height_pips']:
                    continue

                # --- Check for Bullish CRT Pattern ---
                # Manipulation candle breaks below range candle's low
                # Distribution candle closes back inside the range
                if (manipulation_candle['low'] < range_candle['low'] and
                    manipulation_candle['close'] >= range_candle['low']):

                    # Calculate manipulation distance in pips
                    manip_distance = (range_candle['low'] - manipulation_candle['low']) * (10**digits)

                    # Skip if manipulation is too small
                    if manip_distance < self.config['min_manipulation_pips']:
                        continue

                    # Check if near key level if enabled
                    near_key_level = False
                    if self.config['use_key_levels'] and key_levels:
                        for level in key_levels:
                            level_distance = abs(manipulation_candle['low'] - level) * (10**digits)
                            if level_distance <= self.config['key_level_proximity_pips']:
                                near_key_level = True
                                logger.debug(f"Bullish CRT near key level: {level:.{digits}f}")
                                break

                    # Skip if key level filter is enabled but not near a key level
                    if self.config['use_key_levels'] and not near_key_level:
                        continue

                    # Check volume confirmation if enabled
                    volume_confirmed = True
                    if self.config['use_volume_confirmation']:
                        # Check if distribution candle has higher volume than manipulation candle
                        if 'volume' in df.columns:
                            volume_confirmed = distribution_candle['volume'] >= manipulation_candle['volume'] * self.config['volume_threshold']

                    # Skip if volume confirmation is enabled but not confirmed
                    if self.config['use_volume_confirmation'] and not volume_confirmed:
                        continue

                    # Check market structure alignment if enabled
                    structure_aligned = True
                    if self.config['use_market_structure'] and market_structure:
                        # For bullish CRT, prefer bearish or ranging market structure
                        structure_aligned = market_structure['current_structure'] in ['bearish', 'ranging_LH_HL', 'ranging_HH_LL', 'consolidating']

                    # Skip if market structure filter is enabled but not aligned
                    if self.config['use_market_structure'] and not structure_aligned:
                        continue

                    # Calculate SL/TP
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
                        'range_candle': range_candle.to_dict(),
                        'manipulation_candle': manipulation_candle.to_dict(),
                        'distribution_candle': distribution_candle.to_dict(),
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward_ratio': risk_reward_ratio,
                        'near_key_level': near_key_level,
                        'volume_confirmed': volume_confirmed,
                        'structure_aligned': structure_aligned,
                        'confidence': self._calculate_confidence(near_key_level, volume_confirmed, structure_aligned, risk_reward_ratio)
                    })

                    logger.info(f"Bullish CRT detected at {manipulation_candle['time']} - Entry: {entry_price:.{digits}f}, SL: {stop_loss:.{digits}f}, TP: {take_profit:.{digits}f}, RR: {risk_reward_ratio:.2f}")

                # --- Check for Bearish CRT Pattern ---
                # Manipulation candle breaks above range candle's high
                # Distribution candle closes back inside the range
                if (manipulation_candle['high'] > range_candle['high'] and
                    manipulation_candle['close'] <= range_candle['high']):

                    # Calculate manipulation distance in pips
                    manip_distance = (manipulation_candle['high'] - range_candle['high']) * (10**digits)

                    # Skip if manipulation is too small
                    if manip_distance < self.config['min_manipulation_pips']:
                        continue

                    # Check if near key level if enabled
                    near_key_level = False
                    if self.config['use_key_levels'] and key_levels:
                        for level in key_levels:
                            level_distance = abs(manipulation_candle['high'] - level) * (10**digits)
                            if level_distance <= self.config['key_level_proximity_pips']:
                                near_key_level = True
                                logger.debug(f"Bearish CRT near key level: {level:.{digits}f}")
                                break

                    # Skip if key level filter is enabled but not near a key level
                    if self.config['use_key_levels'] and not near_key_level:
                        continue

                    # Check volume confirmation if enabled
                    volume_confirmed = True
                    if self.config['use_volume_confirmation']:
                        # Check if distribution candle has higher volume than manipulation candle
                        if 'volume' in df.columns:
                            volume_confirmed = distribution_candle['volume'] >= manipulation_candle['volume'] * self.config['volume_threshold']

                    # Skip if volume confirmation is enabled but not confirmed
                    if self.config['use_volume_confirmation'] and not volume_confirmed:
                        continue

                    # Check market structure alignment if enabled
                    structure_aligned = True
                    if self.config['use_market_structure'] and market_structure:
                        # For bearish CRT, prefer bullish or ranging market structure
                        structure_aligned = market_structure['current_structure'] in ['bullish', 'ranging_LH_HL', 'ranging_HH_LL', 'consolidating']

                    # Skip if market structure filter is enabled but not aligned
                    if self.config['use_market_structure'] and not structure_aligned:
                        continue

                    # Calculate SL/TP
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
                        'range_candle': range_candle.to_dict(),
                        'manipulation_candle': manipulation_candle.to_dict(),
                        'distribution_candle': distribution_candle.to_dict(),
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward_ratio': risk_reward_ratio,
                        'near_key_level': near_key_level,
                        'volume_confirmed': volume_confirmed,
                        'structure_aligned': structure_aligned,
                        'confidence': self._calculate_confidence(near_key_level, volume_confirmed, structure_aligned, risk_reward_ratio)
                    })

                    logger.info(f"Bearish CRT detected at {manipulation_candle['time']} - Entry: {entry_price:.{digits}f}, SL: {stop_loss:.{digits}f}, TP: {take_profit:.{digits}f}, RR: {risk_reward_ratio:.2f}")

            except Exception as e:
                logger.error(f"Error processing candle index {i}: {e}")
                continue

        logger.info(f"Found {len(patterns)} CRT patterns")
        return patterns

    def _calculate_confidence(self, near_key_level, volume_confirmed, structure_aligned, risk_reward_ratio):
        """Calculate confidence score for a pattern based on various factors"""
        base_confidence = 50  # Start with 50% confidence

        # Add confidence based on key level proximity
        if near_key_level:
            base_confidence += 15

        # Add confidence based on volume confirmation
        if volume_confirmed:
            base_confidence += 10

        # Add confidence based on market structure alignment
        if structure_aligned:
            base_confidence += 10

        # Add confidence based on risk-reward ratio
        if risk_reward_ratio >= 3:
            base_confidence += 15
        elif risk_reward_ratio >= 2:
            base_confidence += 10
        elif risk_reward_ratio >= 1:
            base_confidence += 5

        # Cap confidence at 95%
        return min(base_confidence, 95)

    def generate_signals(self, df, timeframe):
        """Generate trading signals based on CRT patterns"""
        logger.info(f"Generating CRT signals for {timeframe} timeframe...")

        # Detect patterns
        patterns = self.detect_crt_patterns(df)

        # Convert patterns to signals
        signals = []
        for pattern in patterns:
            signal = {
                'strategy': 'CRT',
                'timeframe': timeframe,
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

        logger.info(f"Generated {len(signals)} CRT signals")
        return signals

    def run(self, df_high, df_low=None):
        """Run the CRT strategy on the provided data"""
        logger.info("Running CRT strategy...")

        # Generate signals for higher timeframe
        high_tf_signals = self.generate_signals(df_high, 'high')

        # Generate signals for lower timeframe if provided
        low_tf_signals = []
        if df_low is not None:
            low_tf_signals = self.generate_signals(df_low, 'low')

        # Combine signals
        all_signals = high_tf_signals + low_tf_signals

        # Sort by confidence
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)

        return all_signals

# Example usage
if __name__ == "__main__":
    # This is just for testing
    import pandas as pd
    import numpy as np

    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    np.random.seed(42)

    # Create a dataframe with OHLCV data
    df = pd.DataFrame({
        'time': dates,
        'open': np.random.normal(1.1, 0.01, 100),
        'high': np.random.normal(1.11, 0.01, 100),
        'low': np.random.normal(1.09, 0.01, 100),
        'close': np.random.normal(1.1, 0.01, 100),
        'volume': np.random.normal(1000, 200, 100)
    })

    # Make sure high is highest and low is lowest
    for i in range(len(df)):
        high = max(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'high'])
        low = min(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'low'])
        df.loc[i, 'high'] = high
        df.loc[i, 'low'] = low

    # Create a CRT strategy instance
    strategy = CRTStrategy()

    # Run the strategy
    signals = strategy.run(df)

    # Print the signals
    for signal in signals:
        print(f"Signal: {signal['action']} at {signal['entry']:.5f}, SL: {signal['stop_loss']:.5f}, TP: {signal['take_profit']:.5f}, Confidence: {signal['confidence']}%")
