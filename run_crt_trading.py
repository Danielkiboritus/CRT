"""
Run the CRT Trading System

This script provides a unified interface to run the CRT trading system
in various modes:
- Live trading with continuous execution
- Dashboard for monitoring and control
- Performance monitoring

Features:
- Support for multiple trading accounts
- Customizable timeframes for analysis
- Real-time monitoring of running agents
- Symbol selection for targeted trading
- Recent trade history tracking

Usage:
    python run_crt_trading.py --mode [live|dashboard|monitor] [options]
"""

import argparse
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
        logging.FileHandler("crt_trading.log")
    ]
)
logger = logging.getLogger(__name__)

def update_config(account_num, live_mode, multi_symbol=False, timeframe=None):
    """Update the config.py file with the selected account and mode"""
    logger.info(f"Updating config.py with account {account_num}, live_mode={live_mode}, multi_symbol={multi_symbol}, timeframe={timeframe}")

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
        # Account 1 (5715737)
        if "MT5_LOGIN = 5715737" in line or "# MT5_LOGIN = 5715737" in line:
            if account1_active:  # Account 1 selected
                config_lines[i] = "MT5_LOGIN = 5715737             # Deriv-Demo account number 1\n"
            else:
                config_lines[i] = "# MT5_LOGIN = 5715737             # Deriv-Demo account number 1\n"

        # Account 2 (31819922)
        if "MT5_LOGIN = 31819922" in line or "# MT5_LOGIN = 31819922" in line:
            if account2_active:  # Account 2 selected
                config_lines[i] = "MT5_LOGIN = 31819922            # Deriv-Demo account number 2\n"
            else:
                config_lines[i] = "# MT5_LOGIN = 31819922            # Deriv-Demo account number 2\n"

        # Update backtest mode
        if "BACKTEST_MODE =" in line:
            if live_mode:
                config_lines[i] = "BACKTEST_MODE = False  # Set to False to enable actual order placement (for live trading)\n"
            else:
                config_lines[i] = "BACKTEST_MODE = True   # Set to True to disable actual order placement (for testing logic)\n"

        # Update multi-symbol mode
        if "MULTI_SYMBOL_MODE =" in line:
            if multi_symbol:
                config_lines[i] = "MULTI_SYMBOL_MODE = True  # Set to True to trade multiple symbols\n"
            else:
                config_lines[i] = "MULTI_SYMBOL_MODE = False  # Set to True to trade multiple symbols\n"

        # Update timeframe if specified
        if timeframe and "TIMEFRAME =" in line:
            config_lines[i] = f"TIMEFRAME = {timeframe}  # Timeframe for analysis (e.g., mt5.TIMEFRAME_H1)\n"

    # Write the updated config back to the file
    with open("config.py", "w") as f:
        f.writelines(config_lines)

    logger.info("Config file updated successfully")
    return True

def run_live_trading(account, live, multi, continuous, interval, timeframe=None, symbols=None):
    """Run the CRT agent in live trading mode"""
    logger.info("Starting CRT agent in live trading mode...")

    # Update config
    if not update_config(account, live, multi, timeframe):
        logger.error("Failed to update config. Aborting.")
        return False

    # Store selected symbols if provided
    if symbols:
        with open("selected_symbols.txt", "w") as f:
            f.write("\n".join(symbols))
        logger.info(f"Selected symbols saved: {symbols}")

    # Import the run_mt5_crt_agent module
    import run_mt5_crt_agent

    # Run once or continuously
    if continuous:
        logger.info(f"Running continuously with {interval} second intervals. Press Ctrl+C to stop.")
        try:
            while True:
                logger.info(f"[{datetime.now()}] Starting CRT agent run...")
                run_mt5_crt_agent.run_crt_agent()
                logger.info(f"[{datetime.now()}] Waiting {interval} seconds until next run...")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
    else:
        run_mt5_crt_agent.run_crt_agent()

    return True

def run_dashboard(host, port, debug, account=None):
    """Run the CRT trading dashboard"""
    logger.info("Starting CRT trading dashboard...")

    try:
        # Update config for the selected account if specified
        if account:
            update_config(account, False, True)
            logger.info(f"Dashboard configured for account {account}")

        # Import the improved dashboard
        from improved_dashboard import app

        # Run the Flask app
        app.run(host=host, port=port, debug=debug, use_reloader=False)

        return True
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return False

def run_performance_monitor(account, days, magic, continuous, interval):
    """Run the CRT performance monitor"""
    logger.info("Starting CRT performance monitor...")

    # Update config for the selected account
    if not update_config(account, False, False):
        logger.error("Failed to update config. Aborting.")
        return False

    try:
        # Import the monitor module
        import monitor_crt_performance

        # Run once or continuously
        if continuous:
            logger.info(f"Running continuously with {interval} second intervals. Press Ctrl+C to stop.")
            try:
                while True:
                    logger.info(f"[{datetime.now()}] Updating performance metrics...")
                    monitor_crt_performance.monitor_performance(days, magic)
                    logger.info(f"[{datetime.now()}] Waiting {interval} seconds until next update...")
                    time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
        else:
            monitor_crt_performance.monitor_performance(days, magic)

        return True
    except Exception as e:
        logger.error(f"Error running performance monitor: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run the CRT Trading System")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["live", "dashboard", "monitor"], required=True,
                        help="Mode to run: live (trading), dashboard (web UI), or monitor (performance)")

    # Common options
    parser.add_argument("--account", type=int, choices=[1, 2], default=1,
                        help="Demo account to use: 1 (5715737) or 2 (31819922)")
    parser.add_argument("--accounts", type=str, default=None,
                        help="Comma-separated list of account numbers to use (e.g., '1,2')")

    # Live trading options
    parser.add_argument("--live", action="store_true",
                        help="Run in live mode (place actual trades)")
    parser.add_argument("--multi", action="store_true",
                        help="Run in multi-symbol mode (trade multiple instruments)")
    parser.add_argument("--continuous", action="store_true",
                        help="Run continuously with specified interval")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Interval between runs in seconds (default: 3600)")
    parser.add_argument("--timeframe", type=str, choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"], default=None,
                        help="Timeframe for analysis (e.g., H1 for 1-hour, D1 for daily)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated list of symbols to trade (e.g., 'EURUSD,GBPUSD')")

    # Dashboard options
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to run the dashboard on")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode")

    # Monitor options
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days of history to analyze")
    parser.add_argument("--magic", type=int, default=678901,
                        help="Magic number to filter trades")
    parser.add_argument("--autonomous", action="store_true",
                        help="Run in autonomous mode with automatic decision making")

    args = parser.parse_args()

    # Process account selection
    accounts = []
    if args.accounts:
        accounts = [int(acc.strip()) for acc in args.accounts.split(',')]
    elif args.account:
        accounts = [args.account]

    # Process symbol selection
    symbols = None
    if args.symbols:
        symbols = [sym.strip() for sym in args.symbols.split(',')]

    # Process timeframe
    timeframe = None
    if args.timeframe:
        timeframe_map = {
            "M1": "mt5.TIMEFRAME_M1",
            "M5": "mt5.TIMEFRAME_M5",
            "M15": "mt5.TIMEFRAME_M15",
            "M30": "mt5.TIMEFRAME_M30",
            "H1": "mt5.TIMEFRAME_H1",
            "H4": "mt5.TIMEFRAME_H4",
            "D1": "mt5.TIMEFRAME_D1",
            "W1": "mt5.TIMEFRAME_W1",
            "MN1": "mt5.TIMEFRAME_MN1"
        }
        timeframe = timeframe_map.get(args.timeframe)

    # Print banner
    print("=" * 50)
    print("CRT Trading System")
    print("=" * 50)
    print(f"Mode: {args.mode.upper()}")

    if args.mode == "live":
        account_info = [f"Account {acc} ({'5715737' if acc == 1 else '31819922'})" for acc in accounts]
        print(f"Accounts: {', '.join(account_info)}")
        print(f"Trading Mode: {'LIVE' if args.live else 'BACKTEST'}")
        print(f"Symbol Mode: {'MULTI-SYMBOL' if args.multi else 'SINGLE-SYMBOL'}")
        if symbols:
            print(f"Selected Symbols: {', '.join(symbols)}")
        if args.timeframe:
            print(f"Timeframe: {args.timeframe}")
        print(f"Continuous: {'Yes' if args.continuous else 'No'}")
        if args.continuous:
            print(f"Interval: {args.interval} seconds")
        print(f"Autonomous: {'Yes' if args.autonomous else 'No'}")

        # Confirm with user if live mode and not autonomous
        if args.live and not args.autonomous:
            confirm = input("You are about to run in LIVE mode which will place actual trades. Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return

        # Run live trading for each account
        for account in accounts:
            print(f"\nStarting trading for Account {account}...")
            run_live_trading(account, args.live, args.multi, args.continuous, args.interval, timeframe, symbols)

    elif args.mode == "dashboard":
        account_info = [f"Account {acc} ({'31819922' if acc == 2 else '5715737'})" for acc in accounts]
        print(f"Accounts: {', '.join(account_info)}")
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"Debug: {'Enabled' if args.debug else 'Disabled'}")
        print("=" * 50)
        print("Open your browser and navigate to:")
        print(f"http://{args.host}:{args.port}")

        # Run dashboard with the first account in the list
        account = accounts[0] if accounts else None
        run_dashboard(args.host, args.port, args.debug, account)

    elif args.mode == "monitor":
        account_info = [f"Account {acc} ({'31819922' if acc == 2 else '5715737'})" for acc in accounts]
        print(f"Accounts: {', '.join(account_info)}")
        print(f"Days: {args.days}")
        print(f"Magic Number: {args.magic}")
        print(f"Continuous: {'Yes' if args.continuous else 'No'}")
        if args.continuous:
            print(f"Interval: {args.interval} seconds")

        # Run performance monitor for each account
        for account in accounts:
            print(f"\nMonitoring Account {account}...")
            run_performance_monitor(account, args.days, args.magic, args.continuous, args.interval)

    print("=" * 50)

if __name__ == "__main__":
    main()
