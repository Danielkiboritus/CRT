"""
Test script for the CRT strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import the CRT strategy
from src.strategies.crt_strategy import CRTStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_sample_data(count=100):
    """Generate sample OHLC data for testing"""
    logger.info(f"Generating sample data with {count} candles...")

    # Start date
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=count)

    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='H')

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
            'volume': int(np.random.uniform(100, 1000)),
            'spread': int(np.random.uniform(1, 5)),
            'real_volume': int(np.random.uniform(1000, 10000))
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add some specific patterns for CRT detection
    # This ensures we'll find some patterns in our test
    if len(df) > 50:
        # Add a bullish CRT pattern
        i = 30
        df.iloc[i-2, df.columns.get_loc('high')] = base_price + 0.010
        df.iloc[i-2, df.columns.get_loc('low')] = base_price - 0.010
        df.iloc[i-2, df.columns.get_loc('close')] = base_price
        df.iloc[i-2, df.columns.get_loc('open')] = base_price

        df.iloc[i-1, df.columns.get_loc('low')] = base_price - 0.015  # Manipulation low
        df.iloc[i-1, df.columns.get_loc('high')] = base_price + 0.005
        df.iloc[i-1, df.columns.get_loc('close')] = base_price
        df.iloc[i-1, df.columns.get_loc('open')] = base_price - 0.005

        df.iloc[i, df.columns.get_loc('close')] = base_price + 0.005  # Distribution
        df.iloc[i, df.columns.get_loc('open')] = base_price
        df.iloc[i, df.columns.get_loc('high')] = base_price + 0.008
        df.iloc[i, df.columns.get_loc('low')] = base_price - 0.002

        # Add a bearish CRT pattern
        j = 60
        df.iloc[j-2, df.columns.get_loc('high')] = base_price + 0.010
        df.iloc[j-2, df.columns.get_loc('low')] = base_price - 0.010
        df.iloc[j-2, df.columns.get_loc('close')] = base_price
        df.iloc[j-2, df.columns.get_loc('open')] = base_price

        df.iloc[j-1, df.columns.get_loc('high')] = base_price + 0.015  # Manipulation high
        df.iloc[j-1, df.columns.get_loc('low')] = base_price - 0.005
        df.iloc[j-1, df.columns.get_loc('close')] = base_price
        df.iloc[j-1, df.columns.get_loc('open')] = base_price + 0.005

        df.iloc[j, df.columns.get_loc('close')] = base_price - 0.005  # Distribution
        df.iloc[j, df.columns.get_loc('open')] = base_price
        df.iloc[j, df.columns.get_loc('low')] = base_price - 0.008
        df.iloc[j, df.columns.get_loc('high')] = base_price + 0.002

    logger.info(f"Generated {len(df)} candles.")
    return df

def main():
    """Main function to test the CRT strategy"""
    logger.info("Testing CRT Strategy...")

    # Generate sample data
    df = generate_sample_data(100)

    # Create a CRT strategy instance
    strategy = CRTStrategy()

    # Detect CRT patterns
    patterns = strategy.detect_crt_patterns(df)

    # Print the patterns
    logger.info(f"Found {len(patterns)} CRT patterns:")
    for i, pattern in enumerate(patterns, 1):
        logger.info(f"Pattern {i}:")
        logger.info(f"  Type: {pattern['type'].upper()}")
        logger.info(f"  Time: {pattern['time']}")
        logger.info(f"  Entry: {pattern['entry_price']:.5f}")
        logger.info(f"  Stop Loss: {pattern['stop_loss']:.5f}")
        logger.info(f"  Take Profit: {pattern['take_profit']:.5f}")
        logger.info(f"  Risk-Reward Ratio: {pattern['risk_reward_ratio']:.2f}")
        logger.info(f"  Confidence: {pattern['confidence']}%")
        logger.info("")

if __name__ == "__main__":
    main()
