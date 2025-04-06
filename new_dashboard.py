"""
New CRT Trading Dashboard
"""

import os
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import MetaTrader5 as mt5

# Import configuration
from config import *

# Initialize Flask app
app = Flask(__name__)

# Global variables
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
running_agents = {}
selected_symbols = []
available_symbols = {
    'forex': [],
    'indices': [],
    'commodities': [],
    'crypto': [],
    'synthetic': [],
    'other': []
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

# Get available symbols
def get_available_symbols():
    """Get available symbols from MT5."""
    global available_symbols

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                print("Failed to initialize MT5")
                return available_symbols

        # Get all symbols
        symbols = mt5.symbols_get()
        if symbols is None:
            print(f"Failed to get symbols: {mt5.last_error()}")
            return available_symbols

        # Reset available symbols
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
            try:
                name = symbol.name
                path = symbol.path

                if "Forex" in path:
                    available_symbols['forex'].append(name)
                elif "Indices" in path or "Stocks" in path:
                    available_symbols['indices'].append(name)
                elif "Commodities" in path or "Metals" in path:
                    available_symbols['commodities'].append(name)
                elif "Crypto" in path:
                    available_symbols['crypto'].append(name)
                elif "Synthetic" in path or "Volatility" in path or "Crash" in path or "Boom" in path:
                    available_symbols['synthetic'].append(name)
                else:
                    available_symbols['other'].append(name)
            except Exception as e:
                print(f"Error processing symbol: {e}")

        # Sort symbols
        for category in available_symbols:
            available_symbols[category].sort()

        print(f"Found {sum(len(symbols) for symbols in available_symbols.values())} symbols")
        return available_symbols
    except Exception as e:
        print(f"Error getting available symbols: {e}")
        return available_symbols

# Close position
def close_position(ticket):
    """Close a position by ticket number."""
    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                print("Failed to initialize MT5")
                return False, "Failed to initialize MT5"

        # Find the position
        position = mt5.positions_get(ticket=int(ticket))
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
            "comment": "Close by Dashboard",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send the request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Order failed, retcode={result.retcode}: {result.comment}"

        return True, f"Position {ticket} closed successfully"
    except Exception as e:
        print(f"Error closing position: {e}")
        return False, f"Error: {str(e)}"



# Get available symbols
def get_available_symbols():
    """Get available symbols from MT5."""
    global available_symbols

    try:
        # Check if MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                print("Failed to initialize MT5")
                return available_symbols

        # Get all symbols
        symbols = mt5.symbols_get()
        if symbols is None:
            print(f"Failed to get symbols: {mt5.last_error()}")
            return available_symbols

        # Clear existing symbols
        for category in available_symbols:
            available_symbols[category] = []

        # Categorize symbols
        for symbol in symbols:
            name = symbol.name

            # Skip symbols that are not visible in MarketWatch
            if not symbol.visible:
                continue

            # Categorize based on name
            if 'USD' in name and len(name) <= 6 and not name.startswith('XAU') and not name.startswith('XAG'):
                available_symbols['forex'].append(name)
            elif name.startswith('XAU') or name.startswith('XAG') or name.startswith('XBR') or name.startswith('XTI'):
                available_symbols['commodities'].append(name)
            elif name in ['US30', 'US500', 'USTEC', 'UK100', 'DE30', 'JP225', 'AUS200']:
                available_symbols['indices'].append(name)
            elif name.startswith('BTC') or name.startswith('ETH') or name.startswith('LTC') or name.startswith('XRP'):
                available_symbols['crypto'].append(name)
            elif 'Volatility' in name or 'Crash' in name or 'Boom' in name or 'Step' in name or 'Jump' in name or 'Range' in name:
                available_symbols['synthetic'].append(name)
            else:
                available_symbols['other'].append(name)

        print(f"Found {sum(len(symbols) for symbols in available_symbols.values())} symbols")
        return available_symbols
    except Exception as e:
        print(f"Error getting available symbols: {e}")
        return available_symbols

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

        # Update available symbols (less frequently)
        if not available_symbols['forex'] and not available_symbols['synthetic']:
            get_available_symbols()

        last_update_time = datetime.now()
    except Exception as e:
        print(f"Error updating data: {e}")

# Start CRT agent for a symbol
def start_agent(symbol):
    """Start CRT agent for a specific symbol."""
    if symbol in running_agents:
        return False, f"Agent for {symbol} is already running"

    # In a real implementation, this would start the CRT agent
    # For now, we'll just add it to the running_agents dictionary
    running_agents[symbol] = {
        'status': 'running',
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return True, f"Agent for {symbol} started successfully"

# Stop CRT agent for a symbol
def stop_agent(symbol):
    """Stop CRT agent for a specific symbol."""
    if symbol not in running_agents:
        return False, f"No agent running for {symbol}"

    # In a real implementation, this would stop the CRT agent
    # For now, we'll just remove it from the running_agents dictionary
    del running_agents[symbol]

    return True, f"Agent for {symbol} stopped successfully"

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
    return render_template('new_index.html',
                          account_info=account_info,
                          open_positions=open_positions,
                          selected_symbols=selected_symbols,
                          running_agents=running_agents,
                          last_update=last_update_time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/symbols')
def symbols():
    """Symbol selection page."""
    return render_template('new_symbols.html',
                          available_symbols=available_symbols,
                          selected_symbols=selected_symbols,
                          last_update=last_update_time.strftime('%Y-%m-%d %H:%M:%S'))

# API Routes
@app.route('/api/update', methods=['GET'])
def api_update():
    """API endpoint to update data."""
    try:
        update_data()
        return jsonify({
            'success': True,
            'last_update': last_update_time.strftime('%Y-%m-%d %H:%M:%S'),
            'account_info': account_info,
            'open_positions': open_positions
        })
    except Exception as e:
        print(f"Error in API update: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'last_update': last_update_time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500

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

@app.route('/api/close_position', methods=['POST'])
def api_close_position():
    """API endpoint to close a position."""
    ticket = request.form.get('ticket')
    if not ticket:
        return jsonify({'success': False, 'message': 'No ticket provided'})

    success, message = close_position(ticket)
    return jsonify({'success': success, 'message': message})

@app.route('/api/start_agent', methods=['POST'])
def api_start_agent():
    """API endpoint to start a CRT agent for a symbol."""
    symbol = request.form.get('symbol')
    if not symbol:
        return jsonify({'success': False, 'message': 'No symbol provided'})

    success, message = start_agent(symbol)
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop_agent', methods=['POST'])
def api_stop_agent():
    """API endpoint to stop a CRT agent for a symbol."""
    symbol = request.form.get('symbol')
    if not symbol:
        return jsonify({'success': False, 'message': 'No symbol provided'})

    success, message = stop_agent(symbol)
    return jsonify({'success': success, 'message': message})

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Create new index template
with open('templates/new_index.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CRT Trading Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }
        .card {
            margin-bottom: 20px;
            border: none;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .card-header {
            background-color: #343a40;
            color: white;
            font-weight: bold;
        }
        .profit {
            color: #28a745;
            font-weight: bold;
        }
        .loss {
            color: #dc3545;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">CRT Trading Dashboard</h1>

        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        Account Summary
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <h5>Login</h5>
                                <p>{{ account_info.login }}</p>
                            </div>
                            <div class="col-md-3">
                                <h5>Server</h5>
                                <p>{{ account_info.server }}</p>
                            </div>
                            <div class="col-md-3">
                                <h5>Balance</h5>
                                <p>{{ account_info.balance|round(2) }} {{ account_info.currency }}</p>
                            </div>
                            <div class="col-md-3">
                                <h5>Equity</h5>
                                <p>{{ account_info.equity|round(2) }} {{ account_info.currency }}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        Open Positions
                    </div>
                    <div class="card-body">
                        {% if open_positions %}
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Type</th>
                                        <th>Volume</th>
                                        <th>Open Price</th>
                                        <th>Current Price</th>
                                        <th>Profit</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for position in open_positions %}
                                    <tr>
                                        <td>{{ position.symbol }}</td>
                                        <td>{{ position.type_str }}</td>
                                        <td>{{ position.volume }}</td>
                                        <td>{{ position.price_open }}</td>
                                        <td>{{ position.price_current }}</td>
                                        <td class="{% if position.profit > 0 %}profit{% elif position.profit < 0 %}loss{% endif %}">
                                            {{ position.profit|round(2) }}
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                        <div class="alert alert-info">No open positions</div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>

        <div class="text-muted mt-3">
            Last update: <span id="lastUpdateTime">{{ last_update }}</span>
            <button id="refreshBtn" class="btn btn-sm btn-primary ms-2">Refresh</button>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Refresh button
            document.getElementById('refreshBtn').addEventListener('click', function() {
                fetch('/api/update')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('lastUpdateTime').textContent = data.last_update;
                            location.reload();
                        }
                    });
            });

            // Auto refresh every 30 seconds
            setInterval(function() {
                fetch('/api/update')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('lastUpdateTime').textContent = data.last_update;
                        }
                    });
            }, 30000);
        });
    </script>
</body>
</html>""")

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
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error in main function: {e}")
