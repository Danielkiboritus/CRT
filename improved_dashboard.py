"""
Improved CRT Trading Dashboard

A web-based dashboard for monitoring and controlling the CRT trading agent.
Features:
- Real-time performance monitoring
- Symbol selection for trading
- Manual trade management
- Account information display
- Improved UI with better aesthetics
"""

import os
import json
import time
import threading
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import MetaTrader5 as mt5
import logging

# Import configuration
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dashboard.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Global variables
running_agents = {}
selected_symbols = []
account_info = {
    'login': 0,
    'server': '',
    'balance': 0.0,
    'equity': 0.0,
    'profit': 0.0,
    'margin': 0.0,
    'margin_free': 0.0,
    'margin_level': 0.0,
    'currency': 'USD'
}
open_positions = []
trade_history = []
recent_trades = []
performance_metrics = {
    'total_trades': 0,
    'profitable_trades': 0,
    'losing_trades': 0,
    'win_rate': 0.0,
    'total_profit': 0.0,
    'avg_profit': 0.0,
    'avg_loss': 0.0,
    'profit_factor': 0.0,
    'symbol_metrics': {}
}
available_symbols = {
    'forex': [],
    'indices': [],
    'commodities': [],
    'crypto': [],
    'synthetic': [],
    'other': []
}
available_timeframes = [
    {'name': 'M1', 'value': 'mt5.TIMEFRAME_M1', 'description': '1 minute'},
    {'name': 'M5', 'value': 'mt5.TIMEFRAME_M5', 'description': '5 minutes'},
    {'name': 'M15', 'value': 'mt5.TIMEFRAME_M15', 'description': '15 minutes'},
    {'name': 'M30', 'value': 'mt5.TIMEFRAME_M30', 'description': '30 minutes'},
    {'name': 'H1', 'value': 'mt5.TIMEFRAME_H1', 'description': '1 hour'},
    {'name': 'H4', 'value': 'mt5.TIMEFRAME_H4', 'description': '4 hours'},
    {'name': 'D1', 'value': 'mt5.TIMEFRAME_D1', 'description': '1 day'},
    {'name': 'W1', 'value': 'mt5.TIMEFRAME_W1', 'description': '1 week'},
    {'name': 'MN1', 'value': 'mt5.TIMEFRAME_MN1', 'description': '1 month'}
]
available_accounts = [
    {'number': 1, 'login': 5715737, 'server': 'Deriv-Demo', 'description': 'Deriv-Demo Account 1'},
    {'number': 2, 'login': 31819922, 'server': 'Deriv-Demo', 'description': 'Deriv-Demo Account 2'}
]
current_timeframe = 'H1'
current_account = 1
last_update_time = datetime.now()

# Initialize MT5 connection
def initialize_mt5():
    """Initialize connection to MT5 terminal."""
    try:
        logger.info("Initializing MT5 connection...")

        # First try to initialize without parameters
        if not mt5.initialize():
            logger.warning(f"Basic initialize() failed, error code = {mt5.last_error()}")

            # Try with explicit parameters
            init_params = {
                "login": MT5_LOGIN,
                "password": MT5_PASSWORD,
                "server": MT5_SERVER,
            }

            if MT5_TERMINAL_PATH:
                init_params["path"] = MT5_TERMINAL_PATH

            logger.info(f"Trying to initialize with parameters: login={MT5_LOGIN}, server={MT5_SERVER}")

            if not mt5.initialize(**init_params):
                logger.error(f"Initialize with parameters failed, error code = {mt5.last_error()}")
                return False

        # Login to MT5
        if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
            logger.error(f"Login failed, error code = {mt5.last_error()}")
            mt5.shutdown()
            return False

        # Display connection status
        logger.info(f"MetaTrader5 package version: {mt5.__version__}")

        # Display account info
        acc_info = mt5.account_info()
        if acc_info:
            logger.info(f"Connected to account: {acc_info.login} on {acc_info.server} [{acc_info.name}]")
        else:
            logger.error(f"Failed to get account info after initialize: {mt5.last_error()}")
            mt5.shutdown()
            return False

        return True
    except Exception as e:
        logger.error(f"Error initializing MT5: {e}", exc_info=True)
        return False

# Get account information
def get_account_info():
    """Get account information from MT5."""
    global account_info

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                logger.error("Failed to initialize MT5")
                return account_info

        acc_info = mt5.account_info()
        if acc_info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return account_info

        account_info = {
            'login': acc_info.login,
            'server': acc_info.server,
            'balance': acc_info.balance,
            'equity': acc_info.equity,
            'profit': acc_info.profit,
            'margin': acc_info.margin,
            'margin_free': acc_info.margin_free,
            'margin_level': acc_info.margin_level,
            'currency': acc_info.currency
        }

        logger.info(f"Account info updated: Balance={account_info['balance']}, Equity={account_info['equity']}")
        return account_info
    except Exception as e:
        logger.error(f"Error getting account info: {e}", exc_info=True)
        return account_info

# Get open positions
def get_open_positions():
    """Get open positions from MT5."""
    global open_positions

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                logger.error("Failed to initialize MT5")
                return open_positions

        # Get positions
        positions = mt5.positions_get()
        if positions is None:
            logger.error(f"Failed to get positions: {mt5.last_error()}")
            return open_positions

        # Convert to list of dictionaries
        open_positions = []
        for position in positions:
            # Get symbol info for additional details
            symbol_info = mt5.symbol_info(position.symbol)

            # Determine position type string
            position_type = "Buy" if position.type == 0 else "Sell"

            # Calculate profit in pips
            point_value = symbol_info.point if symbol_info else 0.0001
            pips = (position.price_current - position.price_open) / point_value if position.type == 0 else (position.price_open - position.price_current) / point_value

            # Add position to list
            open_positions.append({
                'ticket': position.ticket,
                'symbol': position.symbol,
                'type': position.type,
                'type_str': position_type,
                'volume': position.volume,
                'price_open': position.price_open,
                'price_current': position.price_current,
                'sl': position.sl,
                'tp': position.tp,
                'profit': position.profit,
                'pips': pips,
                'magic': position.magic,
                'comment': position.comment,
                'time': datetime.fromtimestamp(position.time).strftime('%Y-%m-%d %H:%M:%S')
            })

        logger.info(f"Open positions updated: {len(open_positions)} positions found")
        return open_positions
    except Exception as e:
        logger.error(f"Error getting open positions: {e}", exc_info=True)
        return open_positions

# Get trade history
def get_trade_history(days=7):
    """Get trade history from MT5."""
    global trade_history, recent_trades

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                logger.error("Failed to initialize MT5")
                return trade_history

        # Calculate start time (days ago)
        from_date = datetime.now() - pd.Timedelta(days=days)
        from_timestamp = int(from_date.timestamp())

        # Get history
        history = mt5.history_deals_get(from_timestamp)
        if history is None:
            logger.error(f"Failed to get history: {mt5.last_error()}")
            return trade_history

        # Convert to list of dictionaries
        trade_history = []
        for deal in history:
            # Only include deals with profit/loss (entry and exit)
            if deal.profit != 0:
                # Determine deal type string
                deal_type = "Buy" if deal.type == 0 else "Sell" if deal.type == 1 else "Other"

                # Add deal to list
                trade_history.append({
                    'ticket': deal.ticket,
                    'symbol': deal.symbol,
                    'type': deal.type,
                    'type_str': deal_type,
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'magic': deal.magic,
                    'comment': deal.comment,
                    'time': datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S')
                })

        # Sort by time (newest first) and get recent trades
        trade_history.sort(key=lambda x: x['time'], reverse=True)
        recent_trades = trade_history[:10]  # Get 10 most recent trades

        logger.info(f"Trade history updated: {len(trade_history)} deals found")
        return trade_history
    except Exception as e:
        logger.error(f"Error getting trade history: {e}", exc_info=True)
        return trade_history

# Calculate performance metrics
def calculate_performance_metrics():
    """Calculate performance metrics from trade history."""
    global performance_metrics

    try:
        if not trade_history:
            return performance_metrics

        # Basic metrics
        total_trades = len(trade_history)
        profitable_trades = sum(1 for deal in trade_history if deal['profit'] > 0)
        losing_trades = sum(1 for deal in trade_history if deal['profit'] < 0)

        # Win rate
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        # Profit metrics
        total_profit = sum(deal['profit'] for deal in trade_history)
        gross_profit = sum(deal['profit'] for deal in trade_history if deal['profit'] > 0)
        gross_loss = sum(deal['profit'] for deal in trade_history if deal['profit'] < 0)

        # Average profit/loss
        avg_profit = gross_profit / profitable_trades if profitable_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0

        # Profit factor
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

        # Symbol-specific metrics
        symbol_metrics = {}
        symbols = set(deal['symbol'] for deal in trade_history)

        for symbol in symbols:
            symbol_deals = [deal for deal in trade_history if deal['symbol'] == symbol]
            symbol_total = len(symbol_deals)
            symbol_profitable = sum(1 for deal in symbol_deals if deal['profit'] > 0)
            symbol_losing = sum(1 for deal in symbol_deals if deal['profit'] < 0)
            symbol_win_rate = symbol_profitable / symbol_total if symbol_total > 0 else 0
            symbol_total_profit = sum(deal['profit'] for deal in symbol_deals)

            symbol_metrics[symbol] = {
                'total_trades': symbol_total,
                'profitable_trades': symbol_profitable,
                'losing_trades': symbol_losing,
                'win_rate': symbol_win_rate,
                'total_profit': symbol_total_profit
            }

        # Update performance metrics
        performance_metrics = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'symbol_metrics': symbol_metrics
        }

        logger.info(f"Performance metrics updated: Win rate={win_rate*100:.2f}%, Total profit={total_profit}")
        return performance_metrics
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
        return performance_metrics

# Get available symbols
def get_available_symbols():
    """Get available symbols from MT5 and categorize them."""
    global available_symbols

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                logger.error("Failed to initialize MT5")
                return available_symbols

        # Get all symbols
        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error(f"Failed to get symbols: {mt5.last_error()}")
            return available_symbols

        # Reset categories
        available_symbols = {
            'forex': [],
            'indices': [],
            'commodities': [],
            'crypto': [],
            'synthetic': [],
            'other': []
        }

        # Categorize symbols
        for symbol in symbols:
            symbol_name = symbol.name

            # Skip symbols that are not visible in Market Watch
            if not symbol.visible:
                continue

            # Categorize based on name patterns
            if "Volatility" in symbol_name or "Crash" in symbol_name or "Boom" in symbol_name or "Step" in symbol_name or "Jump" in symbol_name or "Range Break" in symbol_name:
                available_symbols['synthetic'].append(symbol_name)
            elif any(crypto in symbol_name for crypto in ["BTC", "ETH", "LTC", "XRP", "BCH", "EOS", "XLM", "TRX", "ADA", "BNB"]):
                available_symbols['crypto'].append(symbol_name)
            elif any(commodity in symbol_name for commodity in ["GOLD", "SILVER", "XAU", "XAG", "OIL", "BRENT", "WTI", "NATURAL_GAS", "COPPER"]):
                available_symbols['commodities'].append(symbol_name)
            elif any(index in symbol_name for index in ["US30", "US500", "USTEC", "NAS100", "SPX500", "UK100", "DE30", "JP225", "AUS200", "FRA40", "EU50"]):
                available_symbols['indices'].append(symbol_name)
            elif len(symbol_name) == 6 and symbol_name.isupper() and any(major in symbol_name for major in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]):
                available_symbols['forex'].append(symbol_name)
            else:
                available_symbols['other'].append(symbol_name)

        # Sort each category
        for category in available_symbols:
            available_symbols[category].sort()

        logger.info(f"Available symbols updated: {sum(len(symbols) for symbols in available_symbols.values())} symbols found")
        return available_symbols
    except Exception as e:
        logger.error(f"Error getting available symbols: {e}", exc_info=True)
        return available_symbols

# Get running agents
def get_running_agents():
    """Get information about running CRT agents."""
    global running_agents

    try:
        # Check if there are any selected symbols with agents
        for symbol in selected_symbols:
            if symbol not in running_agents:
                running_agents[symbol] = {
                    'running': False,
                    'started': None,
                    'last_run': None,
                    'error': None
                }

        # Read selected_symbols.txt if it exists
        if os.path.exists('selected_symbols.txt'):
            with open('selected_symbols.txt', 'r') as f:
                saved_symbols = f.read().splitlines()
                for symbol in saved_symbols:
                    if symbol and symbol not in selected_symbols:
                        selected_symbols.append(symbol)
                        if symbol not in running_agents:
                            running_agents[symbol] = {
                                'running': False,
                                'started': None,
                                'last_run': None,
                                'error': None
                            }

        logger.info(f"Running agents updated: {sum(1 for agent in running_agents.values() if agent['running'])} agents running")
        return running_agents
    except Exception as e:
        logger.error(f"Error getting running agents: {e}", exc_info=True)
        return running_agents

# Update all data
def update_all_data():
    """Update all data from MT5."""
    global last_update_time

    try:
        # Initialize MT5 if needed
        if not mt5.terminal_info():
            if not initialize_mt5():
                logger.error("Failed to initialize MT5")
                return False

        # Update data
        get_account_info()
        get_open_positions()
        get_trade_history()
        calculate_performance_metrics()
        get_available_symbols()
        get_running_agents()

        # Update last update time
        last_update_time = datetime.now()

        logger.info(f"All data updated at {last_update_time}")
        return True
    except Exception as e:
        logger.error(f"Error updating all data: {e}", exc_info=True)
        return False

# Close position
def close_position(ticket):
    """Close a position by ticket number."""
    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                logger.error("Failed to initialize MT5")
                return False, "Failed to initialize MT5"

        # Get position
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Failed to get position {ticket}: {mt5.last_error()}")
            return False, f"Failed to get position {ticket}"

        # Get position details
        position = position[0]
        symbol = position.symbol
        lot = position.volume
        type_op = 1 if position.type == 0 else 0  # Reverse for close (SELL for BUY, BUY for SELL)

        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": type_op,
            "position": ticket,
            "magic": MAGIC_NUMBER,
            "comment": "Close position from dashboard",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send close request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {ticket}: {result.retcode}, {result.comment}")
            return False, f"Failed to close position: {result.comment}"

        logger.info(f"Position {ticket} closed successfully")
        return True, f"Position {ticket} closed successfully"
    except Exception as e:
        logger.error(f"Error closing position {ticket}: {e}", exc_info=True)
        return False, f"Error closing position: {str(e)}"

# Start agent for symbol
def start_agent_for_symbol(symbol):
    """Start a CRT agent for a specific symbol."""
    global running_agents

    try:
        # Check if agent is already running
        if symbol in running_agents and running_agents[symbol]['running']:
            logger.warning(f"Agent for {symbol} is already running")
            return False, f"Agent for {symbol} is already running"

        # Import the run_mt5_crt_agent module
        import run_mt5_crt_agent

        # Create a thread to run the agent
        def agent_thread():
            try:
                # Update config for this symbol
                run_mt5_crt_agent.SYMBOL = symbol

                # Run the agent
                run_mt5_crt_agent.run_crt_agent()

                # Update agent status
                running_agents[symbol]['running'] = False
                running_agents[symbol]['last_run'] = datetime.now()
                logger.info(f"Agent for {symbol} completed")
            except Exception as e:
                logger.error(f"Error running agent for {symbol}: {e}", exc_info=True)
                running_agents[symbol]['running'] = False
                running_agents[symbol]['error'] = str(e)

        # Start the thread
        thread = threading.Thread(target=agent_thread)
        thread.daemon = True
        thread.start()

        # Update agent status
        running_agents[symbol] = {
            'running': True,
            'started': datetime.now(),
            'thread': thread,
            'error': None
        }

        logger.info(f"Agent for {symbol} started")
        return True, f"Agent for {symbol} started"
    except Exception as e:
        logger.error(f"Error starting agent for {symbol}: {e}", exc_info=True)
        return False, f"Error starting agent: {str(e)}"

# Stop agent for symbol
def stop_agent_for_symbol(symbol):
    """Stop a CRT agent for a specific symbol."""
    global running_agents

    try:
        # Check if agent is running
        if symbol not in running_agents or not running_agents[symbol]['running']:
            logger.warning(f"Agent for {symbol} is not running")
            return False, f"Agent for {symbol} is not running"

        # Update agent status
        running_agents[symbol]['running'] = False

        logger.info(f"Agent for {symbol} stopped")
        return True, f"Agent for {symbol} stopped"
    except Exception as e:
        logger.error(f"Error stopping agent for {symbol}: {e}", exc_info=True)
        return False, f"Error stopping agent: {str(e)}"

# Routes
@app.route('/')
def index():
    """Dashboard home page."""
    # Update data if needed
    if (datetime.now() - last_update_time).total_seconds() > 60:
        update_all_data()

    return render_template('improved_index.html',
                          account_info=account_info,
                          open_positions=open_positions,
                          performance_metrics=performance_metrics,
                          selected_symbols=selected_symbols,
                          running_agents=running_agents,
                          recent_trades=recent_trades,
                          available_timeframes=available_timeframes,
                          available_accounts=available_accounts,
                          current_timeframe=current_timeframe,
                          current_account=current_account,
                          last_update=last_update_time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/symbols')
def symbols():
    """Symbols page."""
    # Update data if needed
    if (datetime.now() - last_update_time).total_seconds() > 60:
        update_all_data()

    return render_template('symbols.html',
                          account_info=account_info,
                          available_symbols=available_symbols,
                          selected_symbols=selected_symbols,
                          running_agents=running_agents,
                          last_update=last_update_time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/positions')
def positions():
    """Positions page."""
    # Update data if needed
    if (datetime.now() - last_update_time).total_seconds() > 60:
        update_all_data()

    return render_template('positions.html',
                          account_info=account_info,
                          open_positions=open_positions,
                          last_update=last_update_time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/history')
def history():
    """History page."""
    # Update data if needed
    if (datetime.now() - last_update_time).total_seconds() > 60:
        update_all_data()

    return render_template('history.html',
                          account_info=account_info,
                          trade_history=trade_history,
                          performance_metrics=performance_metrics,
                          last_update=last_update_time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/settings')
def settings():
    """Settings page."""
    return render_template('settings.html',
                          account_info=account_info,
                          last_update=last_update_time.strftime('%Y-%m-%d %H:%M:%S'))

# API routes
@app.route('/api/update', methods=['GET'])
def api_update():
    """API endpoint to update all data."""
    success = update_all_data()
    return jsonify({
        'success': success,
        'last_update': last_update_time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/switch_account', methods=['POST'])
def api_switch_account():
    """API endpoint to switch between accounts."""
    global current_account

    account_num = int(request.form.get('account', 1))
    if account_num not in [1, 2]:
        return jsonify({'success': False, 'message': 'Invalid account number'})

    current_account = account_num

    # Update config.py with the selected account
    try:
        # Read the current config file
        with open("config.py", "r") as f:
            config_lines = f.readlines()

        # Find and update the account settings
        for i, line in enumerate(config_lines):
            # Account 2 (5715737)
            if "MT5_LOGIN = 5715737" in line or "# MT5_LOGIN = 5715737" in line:
                if account_num == 1:
                    config_lines[i] = "MT5_LOGIN = 5715737             # Deriv-Demo account number 2\n"
                else:
                    config_lines[i] = "# MT5_LOGIN = 5715737             # Deriv-Demo account number 2\n"

            # Account 1 (31819922)
            if "MT5_LOGIN = 31819922" in line or "# MT5_LOGIN = 31819922" in line:
                if account_num == 2:
                    config_lines[i] = "MT5_LOGIN = 31819922            # Deriv-Demo account number 1\n"
                else:
                    config_lines[i] = "# MT5_LOGIN = 31819922            # Deriv-Demo account number 1\n"

        # Write the updated config back to the file
        with open("config.py", "w") as f:
            f.writelines(config_lines)

        # Reinitialize MT5 with the new account
        mt5.shutdown()
        initialize_mt5()
        update_all_data()

        return jsonify({
            'success': True,
            'message': f'Switched to account {account_num}',
            'account_info': account_info
        })
    except Exception as e:
        logger.error(f"Error switching account: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Error switching account: {str(e)}'})

@app.route('/api/change_timeframe', methods=['POST'])
def api_change_timeframe():
    """API endpoint to change the analysis timeframe."""
    global current_timeframe

    timeframe = request.form.get('timeframe', 'H1')
    if timeframe not in [tf['name'] for tf in available_timeframes]:
        return jsonify({'success': False, 'message': 'Invalid timeframe'})

    current_timeframe = timeframe

    # Update config.py with the selected timeframe
    try:
        # Read the current config file
        with open("config.py", "r") as f:
            config_lines = f.readlines()

        # Find and update the timeframe setting
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

        for i, line in enumerate(config_lines):
            if "TIMEFRAME =" in line:
                config_lines[i] = f"TIMEFRAME = {timeframe_map[timeframe]}  # Timeframe for analysis\n"

        # Write the updated config back to the file
        with open("config.py", "w") as f:
            f.writelines(config_lines)

        return jsonify({
            'success': True,
            'message': f'Changed timeframe to {timeframe}'
        })
    except Exception as e:
        logger.error(f"Error changing timeframe: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Error changing timeframe: {str(e)}'})

@app.route('/api/close_position', methods=['POST'])
def api_close_position():
    """API endpoint to close a position."""
    ticket = int(request.form.get('ticket', 0))
    if ticket <= 0:
        return jsonify({'success': False, 'message': 'Invalid ticket number'})

    success, message = close_position(ticket)

    # Update positions if successful
    if success:
        get_open_positions()

    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/start_agent', methods=['POST'])
def api_start_agent():
    """API endpoint to start an agent for a symbol."""
    symbol = request.form.get('symbol', '')
    if not symbol:
        return jsonify({'success': False, 'message': 'Symbol is required'})

    success, message = start_agent_for_symbol(symbol)
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/stop_agent', methods=['POST'])
def api_stop_agent():
    """API endpoint to stop an agent for a symbol."""
    symbol = request.form.get('symbol', '')
    if not symbol:
        return jsonify({'success': False, 'message': 'Symbol is required'})

    success, message = stop_agent_for_symbol(symbol)
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/select_symbol', methods=['POST'])
def api_select_symbol():
    """API endpoint to select a symbol for trading."""
    symbol = request.form.get('symbol', '')
    if not symbol:
        return jsonify({'success': False, 'message': 'Symbol is required'})

    # Add to selected symbols if not already selected
    if symbol not in selected_symbols:
        selected_symbols.append(symbol)

    return jsonify({
        'success': True,
        'message': f'Symbol {symbol} selected for trading'
    })

@app.route('/api/deselect_symbol', methods=['POST'])
def api_deselect_symbol():
    """API endpoint to deselect a symbol for trading."""
    symbol = request.form.get('symbol', '')
    if not symbol:
        return jsonify({'success': False, 'message': 'Symbol is required'})

    # Remove from selected symbols if selected
    if symbol in selected_symbols:
        selected_symbols.remove(symbol)

    return jsonify({
        'success': True,
        'message': f'Symbol {symbol} deselected from trading'
    })

# Initialize data on startup
def initialize_data():
    """Initialize data on startup."""
    logger.info("Initializing data...")

    # Initialize MT5
    if not initialize_mt5():
        logger.error("Failed to initialize MT5")
        return False

    # Update all data
    update_all_data()

    logger.info("Data initialization complete")
    return True

# Initialize data on startup
initialize_data()

# Run the app if executed directly
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
