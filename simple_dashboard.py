"""
Simple CRT Dashboard
"""

from flask import Flask, render_template

app = Flask(__name__)

# Sample data
account_info = {
    'login': 5715737,
    'server': 'Deriv-Demo',
    'balance': 10000.0,
    'equity': 10000.0,
    'profit': 0.0,
    'margin': 0.0,
    'margin_free': 10000.0,
    'margin_level': 100.0,
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

@app.route('/')
def index():
    """Main dashboard page."""
    from datetime import datetime
    return render_template('index.html', 
                          account_info=account_info,
                          open_positions=open_positions,
                          performance_metrics=performance_metrics,
                          running_agents=running_agents,
                          selected_symbols=selected_symbols,
                          last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/symbols')
def symbols():
    """Symbol selection page."""
    available_symbols = {
        'forex': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'indices': ['US30', 'US500', 'USTEC'],
        'commodities': ['XAUUSD', 'XAGUSD'],
        'crypto': ['BTCUSD', 'ETHUSD'],
        'synthetic': ['Volatility 10 Index', 'Volatility 25 Index'],
        'other': []
    }
    return render_template('symbols.html', 
                          available_symbols=available_symbols,
                          selected_symbols=selected_symbols)

@app.route('/positions')
def positions():
    """Open positions page."""
    return render_template('positions.html', 
                          open_positions=open_positions,
                          account_info=account_info)

@app.route('/history')
def history():
    """Trade history page."""
    return render_template('history.html', 
                          trade_history=trade_history,
                          performance_metrics=performance_metrics,
                          account_info=account_info)

@app.route('/settings')
def settings():
    """Settings page."""
    config = {
        'RISK_PERCENT': 0.5,
        'MAX_TRADES': 2,
        'BACKTEST_MODE': True,
        'HIGHER_TIMEFRAME': 16408,
        'LOWER_TIMEFRAME': 16390
    }
    return render_template('settings.html', 
                          config=config,
                          account_info=account_info)

if __name__ == '__main__':
    app.run(debug=True)
