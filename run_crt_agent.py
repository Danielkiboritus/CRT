"""
Simple script to run the CRT agent
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the CRT strategy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.strategies.crt_strategy import CRTStrategy

def generate_sample_data(count=100):
    """Generate sample OHLC data for testing"""
    logger.info(f"Generating sample data with {count} candles...")
    
    # Start date
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=count)
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    
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

def run_crt_strategy():
    """Run the CRT strategy on sample data"""
    logger.info("Running CRT Strategy on sample data...")
    
    # Generate sample data
    df_high = generate_sample_data(100)
    df_low = generate_sample_data(400)
    
    # Create a CRT strategy instance
    strategy = CRTStrategy()
    
    # Run the strategy
    signals = strategy.run(df_high, df_low)
    
    # Print the signals
    logger.info(f"Found {len(signals)} CRT signals:")
    for i, signal in enumerate(signals, 1):
        logger.info(f"Signal {i}:")
        logger.info(f"  Action: {signal['action']}")
        logger.info(f"  Type: {signal['type']}")
        logger.info(f"  Time: {signal['time']}")
        logger.info(f"  Entry: {signal['entry']:.5f}")
        logger.info(f"  Stop Loss: {signal['stop_loss']:.5f}")
        logger.info(f"  Take Profit: {signal['take_profit']:.5f}")
        logger.info(f"  Risk-Reward: {signal['risk_reward']:.2f}")
        logger.info(f"  Confidence: {signal['confidence']}%")
        logger.info("")

if __name__ == "__main__":
    logger.info("Starting CRT Agent Test...")
    run_crt_strategy()
