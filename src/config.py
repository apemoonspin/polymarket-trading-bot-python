"""
Configuration settings for the Polymarket copy-trading bot.
"""

# Market settings
MARKET_NAME = "BTC_15MIN_UP_DOWN"
RESOLUTION_MINUTES = 15
BTC_INITIAL_PRICE = 40000.0
PRICE_VOLATILITY = 0.005  # 0.5% per 15min period

# Trading settings
STARTING_BALANCE = 10000.0
MAX_POSITION_SIZE = 0.1  # Maximum 10% of balance per trade
POSITION_SIZE = 100.0  # Default USD per trade
TRADING_FEE = 0.01  # 1% fee per trade (reduced for simulation)
MAX_POSITIONS = 10

# Leader trader settings
LEADER_SKILL_LEVEL = 0.65  # 65% win rate
LEADER_SEED = 42

# Simulation settings
BACKTEST_PERIODS = 1000  # Number of 15-min periods
PAPER_TRADE_PERIODS = 100
RANDOM_SEED = 42

# Data paths
DATA_DIR = "data"
BTC_PRICES_FILE = "data/btc_prices.csv"
LEADER_TRADES_FILE = "data/leader_trades.csv"

# Output settings
SHOW_PLOTS = True
SAVE_RESULTS = True
