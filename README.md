# Polymarket Trading Bot (Python)

A Python-based Polymarket bot and Polymarket trading bot for simulated copy-trading. Backtest and paper trade by copying a "leader" trader's positions on BTC price predictions with a powerful, professional workflow.

## Features

- **Market**: BTC price UP/DOWN predictions
- **Timeframe**: 15-minute resolution candles
- **Modes**: 
  - Backtest: Test strategy on historical simulated data
  - Paper Trading: Practice with simulated real-time data
- **Strategy**: Copy a leader trader's positions with confidence-based position sizing
- **Performance Metrics**:
  - Total PnL (Profit and Loss)
  - Win rate
  - Maximum drawdown
  - Sharpe ratio
  - Confusion matrix (UP/DOWN prediction accuracy)
  - Precision and recall metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/apemoonspin/polymarket-trading-bot-python.git
cd polymarket-trading-bot-python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run a backtest with default settings:
```bash
python main.py --mode backtest
```

Run paper trading simulation:
```bash
python main.py --mode paper
```

### Generate Sample Data

Generate sample market data and leader trades:
```bash
python main.py --generate-data --periods 1000
```

## Configuration

Edit `src/config.py` to customize settings:

```python
# Market settings
MARKET_NAME = "BTC_15MIN_UP_DOWN"
RESOLUTION_MINUTES = 15

# Trading settings
STARTING_BALANCE = 10000.0
MAX_POSITION_SIZE = 0.1  # 10% of balance maximum
TRADING_FEE = 0.02  # 2% per trade

# Leader trader settings
LEADER_SKILL_LEVEL = 0.65  # 65% win rate
```

## Project Structure

```
polymarket-trading-bot-python/
├── src/                     # Source code
├── notebooks/               # Analysis notebooks
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── main.py                 # Main entry point
```

## Author

- **GitHub**: https://github.com/apemoonspin
- **Telegram**: https://t.me/apemoonspin
- **Twitter/X**: https://x.com/apemoonspin
For a more effective, updated bot that aims to be more reliable, or if you need help running this bot, please contact me using the links above.

## SEO Keywords

Polymarket bot, Polymarket trading bot, powerful Polymarket bot, powerful Polymarket trading bot, top powerful Polymarket trading bot.

## GitHub SEO (Topics and Description)

**Suggested repository description**:
Polymarket bot and Polymarket trading bot in Python for backtesting and paper trading with a copy-trading strategy.

**Suggested topics**:
polymarket, polymarket-bot, polymarket-trading, polymarket-trading-bot, trading-bot, crypto, prediction-markets, backtesting, paper-trading, python

## Running Tests

Run the test suite:

```bash
python -m unittest tests/test_simulator.py
```

## Disclaimer

This is a simulation tool for educational purposes only. Not financial advice.
