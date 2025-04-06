"""
Run the CRT agent in live mode with selected demo account
"""

import argparse
import sys
import os
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crt_live.log")
    ]
)
logger = logging.getLogger(__name__)

def update_config(account_num, live_mode, multi_symbol=False):
    """Update the config.py file with the selected account and mode"""
    logger.info(f"Updating config.py with account {account_num}, live_mode={live_mode}, multi_symbol={multi_symbol}")

    # Read the current config file
    with open("config.py", "r") as f:
        config_lines = f.readlines()

    # Update account settings
    account1_active = False
    account2_active = False

    if account_num == 1:
        account1_active = True
    elif account_num == 2:
        account2_active = True
    else:
        logger.error(f"Invalid account number: {account_num}")
        return False

    # Find and update the account settings
    for i, line in enumerate(config_lines):
        # Account 2 (5715737)
        if "MT5_LOGIN = 5715737" in line or "# MT5_LOGIN = 5715737" in line:
            if account1_active:
                config_lines[i] = "MT5_LOGIN = 5715737             # Deriv-Demo account number 2\n"
            else:
                config_lines[i] = "# MT5_LOGIN = 5715737             # Deriv-Demo account number 2\n"

        # Account 1 (31819922)
        if "MT5_LOGIN = 31819922" in line or "# MT5_LOGIN = 31819922" in line:
            if account2_active:
                config_lines[i] = "MT5_LOGIN = 31819922            # Deriv-Demo account number 1\n"
            else:
                config_lines[i] = "# MT5_LOGIN = 31819922            # Deriv-Demo account number 1\n"

        # Update backtest mode
        if "BACKTEST_MODE =" in line:
            if live_mode:
                config_lines[i] = "BACKTEST_MODE = False  # Set to True to disable actual order placement (for testing logic)\n"
            else:
                config_lines[i] = "BACKTEST_MODE = True   # Set to True to disable actual order placement (for testing logic)\n"

        # Update multi-symbol mode
        if "MULTI_SYMBOL_MODE =" in line:
            if multi_symbol:
                config_lines[i] = "MULTI_SYMBOL_MODE = True  # Set to True to trade multiple symbols\n"
            else:
                config_lines[i] = "MULTI_SYMBOL_MODE = False  # Set to True to trade multiple symbols\n"

    # Write the updated config back to the file
    with open("config.py", "w") as f:
        f.writelines(config_lines)

    logger.info("Config file updated successfully")
    return True

def run_crt_agent():
    """Run the CRT agent"""
    logger.info("Starting CRT agent...")

    try:
        # Import the run_mt5_crt_agent module
        import run_mt5_crt_agent

        # Run the agent
        run_mt5_crt_agent.run_crt_agent()

        logger.info("CRT agent run completed")
        return True

    except Exception as e:
        logger.error(f"Error running CRT agent: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run the CRT agent in live mode with selected demo account")
    parser.add_argument("--account", type=int, choices=[1, 2], default=1, help="Demo account to use (1 or 2)")
    parser.add_argument("--live", action="store_true", help="Run in live mode (place actual trades)")
    parser.add_argument("--multi", action="store_true", help="Run in multi-symbol mode (trade multiple instruments)")
    parser.add_argument("--continuous", action="store_true", help="Run continuously with specified interval")
    parser.add_argument("--interval", type=int, default=3600, help="Interval between runs in seconds (default: 3600)")

    args = parser.parse_args()

    # Print banner
    print("=" * 50)
    print("CRT Agent Live Trading")
    print("=" * 50)
    print(f"Account: {args.account} ({'5715737' if args.account == 1 else '31819922'})")
    print(f"Mode: {'LIVE' if args.live else 'BACKTEST'}")
    print(f"Symbol Mode: {'MULTI-SYMBOL' if args.multi else 'SINGLE-SYMBOL'}")
    print(f"Continuous: {'Yes' if args.continuous else 'No'}")
    if args.continuous:
        print(f"Interval: {args.interval} seconds")
    print("=" * 50)

    # Confirm with user
    if args.live:
        confirm = input("You are about to run in LIVE mode which will place actual trades. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    # Update config
    if not update_config(args.account, args.live, args.multi):
        print("Failed to update config. Aborting.")
        return

    # Run once or continuously
    if args.continuous:
        print(f"Running continuously with {args.interval} second intervals. Press Ctrl+C to stop.")
        try:
            while True:
                print(f"\n[{datetime.now()}] Starting CRT agent run...")
                run_crt_agent()
                print(f"[{datetime.now()}] Waiting {args.interval} seconds until next run...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        run_crt_agent()

if __name__ == "__main__":
    main()
