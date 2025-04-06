# 🕯️ CRT Trading System Integration

This document explains how the Candle Range Theory (CRT) trading system is integrated with Moon Dev's AI Trading Agents framework.

## Overview

The CRT Trading System can operate in two modes:

1. **Standalone Mode**: The CRT agent runs independently and executes trades directly, while still using the shared risk management framework.

2. **Integrated Mode**: The CRT agent provides signals to the trading agent, which makes the final trading decisions based on multiple data sources.

## Components

### 1. CRT Strategy (`src/strategies/crt_strategy.py`)

The core implementation of the Candle Range Theory strategy:

- Detects CRT patterns (Range, Manipulation, Distribution candles)
- Filters patterns based on key levels, market structure, and volume
- Calculates entry, stop loss, and take profit levels
- Assigns confidence scores to patterns

### 2. CRT Agent (`src/agents/crt_agent.py`)

A standalone agent that:

- Uses the CRT strategy to detect patterns
- Enhances pattern validation using AI
- Can execute trades directly in standalone mode
- Formats signals for the trading agent in integrated mode

### 3. CRT Integration (`src/integrations/crt_integration.py`)

Handles the integration between the CRT agent and the trading agent:

- Filters CRT signals based on confidence and risk-reward
- Formats signals for the trading agent
- Manages the communication between agents

### 4. CRT Configuration (`src/config/crt_config.py`)

Contains all configuration settings for the CRT system:

- Strategy parameters (pattern detection thresholds)
- Agent parameters (standalone/integrated mode)
- Risk management settings
- Timeframe settings

### 5. Main Script (`src/run_crt_system.py`)

Runs the CRT trading system in either standalone or integrated mode:

- Initializes the necessary agents
- Handles risk management checks
- Manages the trading cycle
- Provides command-line options for mode selection

## Risk Management Integration

The CRT system shares the same risk management framework as the other agents:

- Uses the `RiskAgent` to enforce position size limits
- Respects daily PnL limits
- Maintains minimum balance protection
- Skips trading when risk limits are breached

## How It Works

### Standalone Mode

1. The risk agent checks if any risk limits are breached
2. If safe to proceed, the CRT agent runs:
   - Collects market data for monitored tokens
   - Detects CRT patterns using the strategy
   - Validates patterns using AI
   - Executes trades for valid patterns
3. The system sleeps until the next check interval

### Integrated Mode

1. The risk agent checks if any risk limits are breached
2. If safe to proceed, the CRT integration:
   - Gets signals from the CRT agent
   - Filters signals based on confidence and risk-reward
   - Formats signals for the trading agent
3. The trading agent runs with the CRT signals:
   - Combines CRT signals with other data sources
   - Makes final trading decisions
   - Executes trades
4. The system sleeps until the next check interval

## Usage

Run the CRT system with:

```bash
python src/run_crt_system.py --mode [standalone|integrated|auto]
```

Options:
- `standalone`: Run in standalone mode (CRT agent executes trades)
- `integrated`: Run in integrated mode (CRT agent provides signals to trading agent)
- `auto`: Use configuration settings to determine mode (default)

## Configuration

Edit `src/config/crt_config.py` to customize:

- CRT pattern detection parameters
- Mode selection (standalone/integrated)
- Risk management settings
- Timeframes and check intervals

## Adding New Features

To extend the CRT system:

1. **New Pattern Types**: Add new pattern detection methods to `CRTStrategy`
2. **Additional Filters**: Implement new filtering criteria in `detect_crt_patterns`
3. **Enhanced AI Validation**: Modify the `CRT_ANALYSIS_PROMPT` to include new factors
4. **Integration with Other Agents**: Create new integration modules similar to `crt_integration.py`
