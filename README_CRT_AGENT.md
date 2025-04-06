# CRT Trading Agent

This is a trading agent that implements the Candle Range Theory (CRT) strategy for MetaTrader 5. The agent can run in backtest mode (no actual trades) or live mode (placing actual trades).

## Setup

1. Make sure MetaTrader 5 is installed and running
2. Ensure you have Python 3.7+ installed
3. Install required packages:
   ```
   pip install MetaTrader5 pandas numpy
   ```

## Configuration

The agent is configured through the `config.py` file. Key settings include:

- **MT5 Connection Settings**: Login credentials for your demo accounts
- **Trading Parameters**: Symbol and timeframes
- **Risk Management**: Risk percentage and maximum trades
- **Agent Behavior**: Check interval and backtest mode

## Available Demo Accounts

The agent is configured to work with the following demo accounts:

1. **Deriv-Demo Account 2**
   - Login: 5715737
   - Password: 189@Kab@rNet@189
   - Server: Deriv-Demo

2. **Deriv-Demo Account 1**
   - Login: 31819922
   - Password: 189@Kab@rNet@189
   - Server: Deriv-Demo

## Running the Agent

### Basic Run

To run the agent with default settings:

```
python run_mt5_crt_agent.py
```

This will run the agent once in backtest mode (no actual trades).

### Live Trading

To run the agent in live mode with a specific account:

```
python run_crt_live.py --account 1 --live
```

Options:
- `--account`: Select which demo account to use (1 or 2)
- `--live`: Enable live trading (place actual trades)
- `--multi`: Enable multi-symbol mode (trade multiple instruments)
- `--continuous`: Run continuously with specified interval
- `--interval`: Interval between runs in seconds (default: 3600)

Example for continuous running in multi-symbol mode:

```
python run_crt_live.py --account 1 --live --multi --continuous --interval 3600
```

This will run the agent every hour with account 1 in live mode, trading multiple symbols including forex pairs, commodities, indices, and cryptocurrencies.

### Web Dashboard

The CRT agent comes with a web-based dashboard for monitoring and controlling the trading system:

```
python run_dashboard.py
```

Options:
- `--host`: Host to run the dashboard on (default: 127.0.0.1)
- `--port`: Port to run the dashboard on (default: 5000)

Once running, open your browser and navigate to:
```
http://127.0.0.1:5000
```

The dashboard provides:
- Real-time account information and performance metrics
- Symbol selection for trading (including Deriv synthetic pairs)
- Manual trade management (close positions)
- Trade history and performance analysis
- Agent control (start/stop agents for specific symbols)
- Settings management

### Monitoring Performance (Command Line)

If you prefer command-line monitoring, you can use:

```
python monitor_crt_performance.py --account 1
```

Options:
- `--account`: Select which demo account to monitor (1 or 2)
- `--days`: Number of days of history to analyze (default: 7)
- `--magic`: Magic number to filter trades (default: 678901)
- `--continuous`: Run continuously with specified interval
- `--interval`: Interval between updates in seconds (default: 300)

Example for continuous monitoring:

```
python monitor_crt_performance.py --account 1 --continuous --interval 300
```

This will monitor account 1 and update every 5 minutes.

## CRT Strategy

The Candle Range Theory (CRT) strategy identifies specific candlestick patterns:

1. **Range Candle**: Forms the initial price range
2. **Manipulation Candle**: Breaks either above or below the range candle to "grab liquidity"
3. **Distribution Candle**: Moves in the opposite direction of the manipulation

The strategy detects two main pattern types:
- **Bullish CRT**: When price breaks below a range, then reverses upward
- **Bearish CRT**: When price breaks above a range, then reverses downward

Additional filters include:
- Key level proximity
- Market structure alignment
- Risk-reward ratio

## Risk Management

The agent implements conservative risk management:

- Default risk per trade: 0.5% of account balance
- Maximum concurrent trades: 2
- Stop loss: Automatically calculated based on the pattern
- Take profit: Set at the opposite end of the range candle

## Logs

The agent creates log files to track its activity:

- `crt_agent.log`: Main agent log
- `crt_live.log`: Log for the live trading script
- `crt_monitor.log`: Log for the monitoring script

## Troubleshooting

If you encounter issues:

1. Make sure MetaTrader 5 is running
2. Check that you're using the correct login credentials
3. Verify that the symbol (EURUSD) is available in your account
4. Check the log files for error messages

## Disclaimer

This trading agent is for educational purposes only. Trading involves risk, and even demo accounts should be treated with the same caution as real accounts to develop good trading habits.
