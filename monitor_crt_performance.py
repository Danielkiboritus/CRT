"""
Monitor the performance of the CRT agent
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import argparse
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crt_monitor.log")
    ]
)
logger = logging.getLogger(__name__)

def initialize_mt5(login, password, server):
    """Initialize connection to MT5 terminal."""
    logger.info("Initializing MT5 connection...")

    # First try to initialize without parameters
    logger.info("Attempting basic initialization...")
    if not mt5.initialize():
        logger.warning(f"Basic initialization failed, error code = {mt5.last_error()}")
        logger.info("Trying with parameters...")

        init_params = {
            "login": login,
            "password": password,
            "server": server,
        }

        # Attempt to initialize connection
        if not mt5.initialize(**init_params):
            logger.error(f"initialize() failed, error code = {mt5.last_error()}")
            return False
    else:
        logger.info("Basic initialization successful.")

        # Now try to login
        if login and password and server:
            logger.info(f"Attempting login to {server} with account {login}...")
            if not mt5.login(login, password, server):
                logger.error(f"Login failed, error code = {mt5.last_error()}")
                mt5.shutdown()
                return False
            logger.info("Login successful.")

    # Get account info
    account_info = mt5.account_info()
    if account_info:
        logger.info(f"Connected to account: {account_info.login} on {account_info.server}")
        logger.info(f"Balance: {account_info.balance} {account_info.currency}")
        logger.info(f"Equity: {account_info.equity} {account_info.currency}")
        logger.info(f"Profit: {account_info.profit} {account_info.currency}")
    else:
        logger.error(f"Failed to get account info: {mt5.last_error()}")
        mt5.shutdown()
        return False

    return True

def get_open_positions(magic_number=None):
    """Get all open positions, optionally filtered by magic number"""
    positions = mt5.positions_get()

    if positions is None or len(positions) == 0:
        logger.warning(f"No positions found: {mt5.last_error() if positions is None else 'No open positions'}")
        return pd.DataFrame()  # Return empty DataFrame

    # Convert to DataFrame
    positions_df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())

    # Filter by magic number if provided
    if magic_number is not None:
        positions_df = positions_df[positions_df['magic'] == magic_number]

    return positions_df

def get_position_history(days=7, magic_number=None):
    """Get position history for the specified number of days"""
    # Calculate the start date
    now = datetime.now()
    start_date = now - timedelta(days=days)

    # Convert to timestamp
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(now.timestamp())

    # Get history
    history = mt5.history_deals_get(start_timestamp, end_timestamp)

    if history is None:
        logger.warning(f"No history found: {mt5.last_error()}")
        return pd.DataFrame()

    # Convert to DataFrame
    history_df = pd.DataFrame(list(history), columns=history[0]._asdict().keys())

    # Filter by magic number if provided
    if magic_number is not None:
        history_df = history_df[history_df['magic'] == magic_number]

    return history_df

def calculate_performance_metrics(history_df):
    """Calculate performance metrics from history"""
    if history_df.empty:
        logger.warning("No history data to calculate metrics")
        return {}

    # Filter to only include closed positions
    closed_positions = history_df[history_df['entry'] == 1]

    if closed_positions.empty:
        logger.warning("No closed positions found in history")
        return {}

    # Calculate metrics
    total_trades = len(closed_positions)
    profitable_trades = len(closed_positions[closed_positions['profit'] > 0])
    losing_trades = len(closed_positions[closed_positions['profit'] <= 0])

    win_rate = profitable_trades / total_trades if total_trades > 0 else 0

    total_profit = closed_positions['profit'].sum()
    avg_profit = closed_positions[closed_positions['profit'] > 0]['profit'].mean() if profitable_trades > 0 else 0
    avg_loss = closed_positions[closed_positions['profit'] <= 0]['profit'].mean() if losing_trades > 0 else 0

    profit_factor = abs(closed_positions[closed_positions['profit'] > 0]['profit'].sum() /
                        closed_positions[closed_positions['profit'] <= 0]['profit'].sum()) if losing_trades > 0 else float('inf')

    metrics = {
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }

    return metrics

def print_account_summary():
    """Print a summary of the account"""
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to get account info: {mt5.last_error()}")
        return

    print("\n" + "=" * 50)
    print(f"ACCOUNT SUMMARY: {account_info.login} ({account_info.server})")
    print("=" * 50)
    print(f"Balance: {account_info.balance:.2f} {account_info.currency}")
    print(f"Equity: {account_info.equity:.2f} {account_info.currency}")
    print(f"Profit: {account_info.profit:.2f} {account_info.currency}")
    print(f"Margin: {account_info.margin:.2f} {account_info.currency}")
    print(f"Free Margin: {account_info.margin_free:.2f} {account_info.currency}")
    print(f"Margin Level: {account_info.margin_level:.2f}%")
    print("=" * 50)

def print_open_positions(positions_df):
    """Print open positions"""
    if positions_df.empty:
        print("\nNo open positions.")
        return

    print("\n" + "=" * 100)
    print("OPEN POSITIONS")
    print("=" * 100)

    for _, pos in positions_df.iterrows():
        direction = "BUY" if pos['type'] == 0 else "SELL"
        profit_color = "\033[92m" if pos['profit'] >= 0 else "\033[91m"  # Green for profit, red for loss

        print(f"Ticket: {pos['ticket']}, Symbol: {pos['symbol']}, Type: {direction}")
        print(f"  Volume: {pos['volume']:.2f}, Open Price: {pos['price_open']:.5f}")
        print(f"  SL: {pos['sl']:.5f}, TP: {pos['tp']:.5f}")
        print(f"  Profit: {profit_color}{pos['profit']:.2f}\033[0m, Swap: {pos['swap']:.2f}")
        print(f"  Time: {datetime.fromtimestamp(pos['time'])}")
        print("-" * 100)

def print_performance_metrics(metrics):
    """Print performance metrics"""
    if not metrics:
        print("\nNo performance metrics available.")
        return

    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Profitable Trades: {metrics['profitable_trades']} ({metrics['win_rate']*100:.2f}%)")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Total Profit: {metrics['total_profit']:.2f}")
    print(f"Average Profit: {metrics['avg_profit']:.2f}")
    print(f"Average Loss: {metrics['avg_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print("=" * 50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Monitor the performance of the CRT agent")
    parser.add_argument("--account", type=int, choices=[1, 2], default=1, help="Demo account to monitor (1 or 2)")
    parser.add_argument("--days", type=int, default=7, help="Number of days of history to analyze")
    parser.add_argument("--magic", type=int, default=678901, help="Magic number to filter trades")
    parser.add_argument("--continuous", action="store_true", help="Run continuously with specified interval")
    parser.add_argument("--interval", type=int, default=300, help="Interval between updates in seconds (default: 300)")

    args = parser.parse_args()

    # Set account credentials based on selection
    if args.account == 1:
        login = 5715737
        password = "189@Kab@rNet@189"
        server = "Deriv-Demo"
    else:
        login = 31819922
        password = "189@Kab@rNet@189"
        server = "Deriv-Demo"

    # Print banner
    print("=" * 50)
    print("CRT Agent Performance Monitor")
    print("=" * 50)
    print(f"Account: {args.account} ({login})")
    print(f"Days of History: {args.days}")
    print(f"Magic Number: {args.magic}")
    print(f"Continuous: {'Yes' if args.continuous else 'No'}")
    if args.continuous:
        print(f"Interval: {args.interval} seconds")
    print("=" * 50)

    # Initialize MT5
    if not initialize_mt5(login, password, server):
        print("Failed to initialize MT5. Exiting.")
        return

    try:
        # Run once or continuously
        if args.continuous:
            print(f"Running continuously with {args.interval} second intervals. Press Ctrl+C to stop.")
            try:
                while True:
                    # Clear screen
                    os.system('cls' if os.name == 'nt' else 'clear')

                    # Print current time
                    print(f"\n[{datetime.now()}] Updating performance data...")

                    # Get data
                    positions_df = get_open_positions(args.magic)
                    history_df = get_position_history(args.days, args.magic)
                    metrics = calculate_performance_metrics(history_df)

                    # Print reports
                    print_account_summary()
                    print_open_positions(positions_df)
                    print_performance_metrics(metrics)

                    print(f"\n[{datetime.now()}] Next update in {args.interval} seconds...")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped by user.")
        else:
            # Get data
            positions_df = get_open_positions(args.magic)
            history_df = get_position_history(args.days, args.magic)
            metrics = calculate_performance_metrics(history_df)

            # Print reports
            print_account_summary()
            print_open_positions(positions_df)
            print_performance_metrics(metrics)

    finally:
        # Shutdown MT5
        mt5.shutdown()
        logger.info("MT5 connection closed.")

if __name__ == "__main__":
    main()
