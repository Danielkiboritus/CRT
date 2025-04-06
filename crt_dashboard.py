"""
CRT Trading Dashboard

A web-based dashboard for monitoring and controlling the CRT trading agent.
Features:
- Real-time performance monitoring
- Symbol selection for trading
- Manual trade management
- Account information display
"""

import os
import json
import time
import threading
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
import MetaTrader5 as mt5

# Import configuration and CRT agent
from config import *
import run_mt5_crt_agent as crt_agent

# Initialize Flask app
app = Flask(__name__)

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
last_update_time = datetime.now()

# Initialize MT5 connection
def initialize_mt5():
    """Initialize connection to MT5 terminal."""
    try:
        if not mt5.initialize():
            print(f"initialize() failed, error code = {mt5.last_error()}")

            init_params = {
                "login": MT5_LOGIN,
                "password": MT5_PASSWORD,
                "server": MT5_SERVER,
            }

            if not mt5.initialize(**init_params):
                print(f"initialize() with parameters failed, error code = {mt5.last_error()}")
                return False

        # Login to MT5
        if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
            print(f"login() failed, error code = {mt5.last_error()}")
            mt5.shutdown()
            return False

        return True
    except Exception as e:
        print(f"Error initializing MT5: {e}")
        return False

# Get account information
def get_account_info():
    """Get account information from MT5."""
    global account_info

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                print("Failed to initialize MT5")
                return account_info

        acc_info = mt5.account_info()
        if acc_info is None:
            print(f"Failed to get account info: {mt5.last_error()}")
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

        return account_info
    except Exception as e:
        print(f"Error getting account info: {e}")
        return account_info

# Get open positions
def get_open_positions():
    """Get all open positions from MT5."""
    global open_positions

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                print("Failed to initialize MT5")
                return open_positions

        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            print(f"No positions found: {mt5.last_error() if positions is None else 'No open positions'}")
            open_positions = []
            return open_positions

        # Convert to list of dictionaries
        positions_list = []
        for pos in positions:
            try:
                pos_dict = pos._asdict()
                # Convert time to readable format
                pos_dict['time'] = datetime.fromtimestamp(pos_dict['time']).strftime('%Y-%m-%d %H:%M:%S')
                # Add symbol name
                symbol_info = mt5.symbol_info(pos_dict['symbol'])
                if symbol_info:
                    pos_dict['symbol_name'] = symbol_info.path
                else:
                    pos_dict['symbol_name'] = pos_dict['symbol']
                # Add position type as string
                pos_dict['type_str'] = 'BUY' if pos_dict['type'] == 0 else 'SELL'
                positions_list.append(pos_dict)
            except Exception as e:
                print(f"Error processing position: {e}")

        open_positions = positions_list
        return positions_list
    except Exception as e:
        print(f"Error getting open positions: {e}")
        return open_positions

# Get trade history
def get_trade_history(days=7):
    """Get trade history for the specified number of days."""
    global trade_history

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                print("Failed to initialize MT5")
                return trade_history

        # Calculate the start date
        now = datetime.now()
        start_date = now - pd.Timedelta(days=days)

        # Convert to timestamp
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(now.timestamp())

        # Get history
        history = mt5.history_deals_get(start_timestamp, end_timestamp)

        if history is None or len(history) == 0:
            print(f"No history found: {mt5.last_error() if history is None else 'No trade history'}")
            trade_history = []
            return trade_history

        # Convert to list of dictionaries
        history_list = []
        for deal in history:
            try:
                deal_dict = deal._asdict()
                # Convert time to readable format
                deal_dict['time'] = datetime.fromtimestamp(deal_dict['time']).strftime('%Y-%m-%d %H:%M:%S')
                # Add entry/exit type
                deal_dict['entry_str'] = 'Entry' if deal_dict['entry'] == 1 else 'Exit'
                # Add position type as string
                deal_dict['type_str'] = 'BUY' if deal_dict['type'] == 0 else 'SELL'
                history_list.append(deal_dict)
            except Exception as e:
                print(f"Error processing deal: {e}")

        trade_history = history_list
        return history_list
    except Exception as e:
        print(f"Error getting trade history: {e}")
        return trade_history

# Calculate performance metrics
def calculate_performance_metrics():
    """Calculate performance metrics from trade history."""
    global performance_metrics

    history = trade_history
    if not history:
        return {}

    # Filter to only include closed positions
    closed_positions = [deal for deal in history if deal['entry'] == 0]

    if not closed_positions:
        return {}

    # Calculate metrics
    total_trades = len(closed_positions)
    profitable_trades = len([deal for deal in closed_positions if deal['profit'] > 0])
    losing_trades = total_trades - profitable_trades

    win_rate = profitable_trades / total_trades if total_trades > 0 else 0

    total_profit = sum([deal['profit'] for deal in closed_positions])
    avg_profit = sum([deal['profit'] for deal in closed_positions if deal['profit'] > 0]) / profitable_trades if profitable_trades > 0 else 0
    avg_loss = sum([deal['profit'] for deal in closed_positions if deal['profit'] <= 0]) / losing_trades if losing_trades > 0 else 0

    profit_sum = sum([deal['profit'] for deal in closed_positions if deal['profit'] > 0])
    loss_sum = abs(sum([deal['profit'] for deal in closed_positions if deal['profit'] < 0]))
    profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

    # Calculate metrics by symbol
    symbols = set([deal['symbol'] for deal in closed_positions])
    symbol_metrics = {}

    for symbol in symbols:
        symbol_deals = [deal for deal in closed_positions if deal['symbol'] == symbol]
        symbol_total = len(symbol_deals)
        symbol_profitable = len([deal for deal in symbol_deals if deal['profit'] > 0])
        symbol_win_rate = symbol_profitable / symbol_total if symbol_total > 0 else 0
        symbol_profit = sum([deal['profit'] for deal in symbol_deals])

        symbol_metrics[symbol] = {
            'total_trades': symbol_total,
            'win_rate': symbol_win_rate,
            'total_profit': symbol_profit
        }

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

    return performance_metrics

# Get available symbols
def get_available_symbols():
    """Get a list of available symbols from MT5."""
    symbols_info = mt5.symbols_get()
    if symbols_info is None:
        print(f"Failed to get symbols: {mt5.last_error()}")
        return []

    # Extract symbol names and organize by category
    forex = []
    indices = []
    commodities = []
    crypto = []
    synthetic = []
    other = []

    for symbol in symbols_info:
        name = symbol.name
        path = symbol.path

        if "Forex" in path:
            forex.append(name)
        elif "Indices" in path or "Stocks" in path:
            indices.append(name)
        elif "Commodities" in path:
            commodities.append(name)
        elif "Crypto" in path:
            crypto.append(name)
        elif "Synthetic" in path or "Volatility" in path or "Crash" in path or "Boom" in path:
            synthetic.append(name)
        else:
            other.append(name)

    return {
        'forex': forex,
        'indices': indices,
        'commodities': commodities,
        'crypto': crypto,
        'synthetic': synthetic,
        'other': other
    }

# Close a position
def close_position(ticket):
    """Close a position by ticket number."""
    # Find the position
    position = mt5.positions_get(ticket=ticket)
    if position is None or len(position) == 0:
        return False, f"Position {ticket} not found"

    position = position[0]

    # Prepare close request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
        "deviation": 20,
        "magic": position.magic,
        "comment": "Close by CRT Dashboard",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Send the request
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"Order failed, retcode={result.retcode}: {result.comment}"

    return True, f"Position {ticket} closed successfully"

# Start CRT agent for a symbol
def start_crt_agent(symbol):
    """Start CRT agent for a specific symbol."""
    if symbol in running_agents:
        return False, f"Agent for {symbol} is already running"

    # Create a thread to run the agent
    def agent_thread():
        # Set the symbol in config
        global SYMBOL
        SYMBOL = symbol

        # Run the agent
        crt_agent.run_crt_agent()

    thread = threading.Thread(target=agent_thread)
    thread.daemon = True
    thread.start()

    running_agents[symbol] = {
        'thread': thread,
        'start_time': datetime.now(),
        'status': 'running'
    }

    return True, f"Agent for {symbol} started successfully"

# Stop CRT agent for a symbol
def stop_crt_agent(symbol):
    """Stop CRT agent for a specific symbol."""
    if symbol not in running_agents:
        return False, f"No agent running for {symbol}"

    # Mark the agent as stopped
    running_agents[symbol]['status'] = 'stopped'

    # Remove from running agents
    del running_agents[symbol]

    return True, f"Agent for {symbol} stopped successfully"

# Update data
def update_data():
    """Update all data from MT5."""
    global last_update_time

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                print("Failed to initialize MT5")

        # Update account info
        get_account_info()

        # Update open positions
        get_open_positions()

        # Update trade history
        get_trade_history()

        # Calculate performance metrics
        calculate_performance_metrics()

        last_update_time = datetime.now()
    except Exception as e:
        print(f"Error updating data: {e}")

# Background updater thread
def background_updater():
    """Background thread to update data periodically."""
    while True:
        try:
            update_data()
        except Exception as e:
            print(f"Error in background updater: {e}")

        # Sleep for a while before the next update
        try:
            time.sleep(10)  # Update every 10 seconds
        except KeyboardInterrupt:
            print("Background updater interrupted")
            break

# Routes
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html',
                          account_info=account_info,
                          open_positions=open_positions,
                          performance_metrics=performance_metrics,
                          running_agents=running_agents,
                          selected_symbols=selected_symbols,
                          last_update=last_update_time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/symbols')
def symbols():
    """Symbol selection page."""
    available_symbols = get_available_symbols()
    return render_template('symbols.html',
                          available_symbols=available_symbols,
                          selected_symbols=selected_symbols)

@app.route('/positions')
def positions():
    """Open positions page."""
    return render_template('positions.html',
                          open_positions=open_positions)

@app.route('/history')
def history():
    """Trade history page."""
    return render_template('history.html',
                          trade_history=trade_history,
                          performance_metrics=performance_metrics)

@app.route('/settings')
def settings():
    """Settings page."""
    return render_template('settings.html',
                          config={
                              'RISK_PERCENT': RISK_PERCENT,
                              'MAX_TRADES': MAX_TRADES,
                              'BACKTEST_MODE': BACKTEST_MODE,
                              'HIGHER_TIMEFRAME': HIGHER_TIMEFRAME,
                              'LOWER_TIMEFRAME': LOWER_TIMEFRAME
                          })

# API Routes
@app.route('/api/update', methods=['GET'])
def api_update():
    """API endpoint to update data."""
    update_data()
    return jsonify({
        'success': True,
        'last_update': last_update_time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/select_symbol', methods=['POST'])
def api_select_symbol():
    """API endpoint to select a symbol for trading."""
    symbol = request.form.get('symbol')
    if not symbol:
        return jsonify({'success': False, 'message': 'No symbol provided'})

    if symbol in selected_symbols:
        return jsonify({'success': False, 'message': f'Symbol {symbol} already selected'})

    selected_symbols.append(symbol)
    return jsonify({'success': True, 'message': f'Symbol {symbol} selected successfully'})

@app.route('/api/deselect_symbol', methods=['POST'])
def api_deselect_symbol():
    """API endpoint to deselect a symbol for trading."""
    symbol = request.form.get('symbol')
    if not symbol:
        return jsonify({'success': False, 'message': 'No symbol provided'})

    if symbol not in selected_symbols:
        return jsonify({'success': False, 'message': f'Symbol {symbol} not selected'})

    selected_symbols.remove(symbol)
    return jsonify({'success': True, 'message': f'Symbol {symbol} deselected successfully'})

@app.route('/api/start_agent', methods=['POST'])
def api_start_agent():
    """API endpoint to start a CRT agent for a symbol."""
    symbol = request.form.get('symbol')
    if not symbol:
        return jsonify({'success': False, 'message': 'No symbol provided'})

    success, message = start_crt_agent(symbol)
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop_agent', methods=['POST'])
def api_stop_agent():
    """API endpoint to stop a CRT agent for a symbol."""
    symbol = request.form.get('symbol')
    if not symbol:
        return jsonify({'success': False, 'message': 'No symbol provided'})

    success, message = stop_crt_agent(symbol)
    return jsonify({'success': success, 'message': message})

@app.route('/api/close_position', methods=['POST'])
def api_close_position():
    """API endpoint to close a position."""
    ticket = request.form.get('ticket')
    if not ticket:
        return jsonify({'success': False, 'message': 'No ticket provided'})

    success, message = close_position(int(ticket))
    return jsonify({'success': success, 'message': message})

@app.route('/api/update_settings', methods=['POST'])
def api_update_settings():
    """API endpoint to update settings."""
    global RISK_PERCENT, MAX_TRADES, BACKTEST_MODE

    risk_percent = request.form.get('risk_percent')
    max_trades = request.form.get('max_trades')
    backtest_mode = request.form.get('backtest_mode')

    if risk_percent:
        RISK_PERCENT = float(risk_percent)

    if max_trades:
        MAX_TRADES = int(max_trades)

    if backtest_mode:
        BACKTEST_MODE = backtest_mode.lower() == 'true'

    return jsonify({
        'success': True,
        'message': 'Settings updated successfully',
        'settings': {
            'RISK_PERCENT': RISK_PERCENT,
            'MAX_TRADES': MAX_TRADES,
            'BACKTEST_MODE': BACKTEST_MODE
        }
    })

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Main function
if __name__ == '__main__':
    try:
        # Initialize MT5 (but don't exit if it fails)
        initialize_mt5()

        # Update data initially
        update_data()

        # Start background updater thread
        updater_thread = threading.Thread(target=background_updater)
        updater_thread.daemon = True
        updater_thread.start()

        # Run Flask app
        app.run(debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error in main function: {e}")
