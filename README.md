# CRT Trading System

A trading system that implements the CRT (Confluence, Reversal, Trend) strategy for MetaTrader 5.

## Features

- Multi-symbol trading support
- Automated trade execution
- Risk management
- Support for Deriv and other MT5 brokers
- Continuous trading mode
- Dashboard for monitoring trades

## Requirements

- Python 3.8+
- MetaTrader 5
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Configure your MT5 account details in config.py

## Usage

Run the CRT trading system with:

```
python run_crt_trading.py --mode live --account 2 --live --multi --continuous --interval 300 --autonomous
```

### Command Line Arguments

- `--mode`: Trading mode (live or backtest)
- `--account`: Account number (1 or 2)
- `--live`: Use live trading mode
- `--multi`: Trade multiple symbols
- `--continuous`: Run continuously
- `--interval`: Interval between runs in seconds
- `--autonomous`: Run in autonomous mode

## License

MIT

## Disclaimer

Trading involves risk. This software is for educational purposes only. Use at your own risk.