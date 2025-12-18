"""
Main entry point for Polymarket Copy Trading Bot.
Supports both backtest and paper trading modes.
"""
import sys
import argparse
from datetime import datetime, timedelta

# Import configuration constants
from src.config import (
    BACKTEST_PERIODS,
    PAPER_TRADE_PERIODS,
    BTC_INITIAL_PRICE,
    PRICE_VOLATILITY,
    RANDOM_SEED,
    LEADER_SKILL_LEVEL,
    POSITION_SIZE,
    LEADER_SEED,
    DATA_DIR,
    RESOLUTION_MINUTES,
    BTC_PRICES_FILE,
    LEADER_TRADES_FILE,
    STARTING_BALANCE,
    MAX_POSITION_SIZE,
    TRADING_FEE,
    SAVE_RESULTS,
    MARKET_NAME,
)
from src.data_loader import DataLoader
from src.market import MarketSimulator
from src.leader import LeaderTrader
from src.simulator import TradingSimulator
from src.metrics import MetricsCalculator


def generate_sample_data(num_periods: int = BACKTEST_PERIODS):
    """
    Generate sample market data and leader trades.
    
    Args:
        num_periods: Number of 15-minute periods to generate
    """
    print(f"\nGenerating sample data for {num_periods} periods...")
    
    # Initialize components
    market_sim = MarketSimulator(
        initial_price=BTC_INITIAL_PRICE,
        volatility=PRICE_VOLATILITY,
        seed=RANDOM_SEED
    )
    
    leader = LeaderTrader(
        skill_level=LEADER_SKILL_LEVEL,
        position_size=POSITION_SIZE,
        seed=LEADER_SEED
    )
    
    # Generate market data
    start_time = datetime.now() - timedelta(minutes=15 * num_periods)
    market_data = market_sim.generate_market_data(num_periods, start_time)
    
    # Generate leader trades
    leader_trades = leader.generate_trades(market_data)
    
    # Save data
    data_loader = DataLoader(DATA_DIR, RESOLUTION_MINUTES)
    data_loader.save_btc_prices(market_data, BTC_PRICES_FILE)
    data_loader.save_leader_trades(leader_trades, LEADER_TRADES_FILE)
    
    print(f"✓ Generated {len(market_data)} market data points")
    print(f"✓ Generated {len(leader_trades)} leader trades")
    print(f"✓ Leader skill level: {LEADER_SKILL_LEVEL:.1%}")
    
    return market_data, leader_trades


def run_backtest():
    """Run backtest mode simulation."""
    print("\n" + "="*60)
    print(f"POLYMARKET COPY TRADING BOT - BACKTEST MODE")
    print(f"Market: {MARKET_NAME}")
    print(f"Resolution: {RESOLUTION_MINUTES} minutes")
    print("="*60)
    
    # Load or generate data
    data_loader = DataLoader(DATA_DIR, RESOLUTION_MINUTES)
    
    try:
        print("\nLoading data from files...")
        market_data = data_loader.load_btc_prices(BTC_PRICES_FILE)
        leader_trades = data_loader.load_leader_trades(LEADER_TRADES_FILE)
        print(f"✓ Loaded {len(market_data)} market data points")
        print(f"✓ Loaded {len(leader_trades)} leader trades")
    except FileNotFoundError:
        print("Data files not found. Generating sample data...")
        market_data, leader_trades = generate_sample_data(BACKTEST_PERIODS)
    
    # Align trades to price timestamps
    print("\nAligning trades to price timestamps...")
    leader_trades = data_loader.align_trades_to_prices(leader_trades, market_data)
    print(f"✓ Aligned {len(leader_trades)} trades")
    
    # Initialize simulator
    print(f"\nInitializing trading simulator...")
    print(f"  Starting Balance: ${STARTING_BALANCE:,.2f}")
    print(f"  Max Position Size: {MAX_POSITION_SIZE*100:.1f}% of balance")
    print(f"  Trading Fee: {TRADING_FEE*100:.1f}%")
    
    simulator = TradingSimulator(
        starting_balance=STARTING_BALANCE,
        max_position_size=MAX_POSITION_SIZE,
        trading_fee=TRADING_FEE,
        confidence_threshold=0.5
    )
    
    # Run simulation
    print("\nRunning backtest simulation...")
    trades_df = simulator.run_backtest(market_data, leader_trades)
    
    if trades_df.empty:
        print("⚠ No trades executed!")
        return
    
    print(f"✓ Executed {len(trades_df)} trades")
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(trades_df, STARTING_BALANCE)
    
    # Print results
    metrics_calc.print_metrics(metrics)
    
    # Save results
    if SAVE_RESULTS:
        results_file = f"results_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved to {results_file}")
    
    return metrics


def run_paper_trade():
    """Run paper trading mode simulation."""
    print("\n" + "="*60)
    print(f"POLYMARKET COPY TRADING BOT - PAPER TRADING MODE")
    print(f"Market: {MARKET_NAME}")
    print(f"Resolution: {RESOLUTION_MINUTES} minutes")
    print("="*60)
    
    # Generate fresh data for paper trading
    print("\nGenerating paper trading data...")
    market_data, leader_trades = generate_sample_data(PAPER_TRADE_PERIODS)
    
    # Initialize simulator
    print(f"\nInitializing paper trading simulator...")
    print(f"  Starting Balance: ${STARTING_BALANCE:,.2f}")
    print(f"  Max Position Size: {MAX_POSITION_SIZE*100:.1f}% of balance")
    print(f"  Trading Fee: {TRADING_FEE*100:.1f}%")
    
    simulator = TradingSimulator(
        starting_balance=STARTING_BALANCE,
        max_position_size=MAX_POSITION_SIZE,
        trading_fee=TRADING_FEE,
        confidence_threshold=0.5
    )
    
    # Run simulation
    print("\nRunning paper trading simulation...")
    trades_df = simulator.run_paper_trade(market_data, leader_trades)
    
    if trades_df.empty:
        print("⚠ No trades executed!")
        return
    
    print(f"✓ Executed {len(trades_df)} trades")
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(trades_df, STARTING_BALANCE)
    
    # Print results
    metrics_calc.print_metrics(metrics)
    
    # Save results
    if SAVE_RESULTS:
        results_file = f"results_paper_trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved to {results_file}")
    
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Polymarket Copy Trading Bot - Backtest and Paper Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode backtest          Run backtest simulation
  python main.py --mode paper             Run paper trading simulation
  python main.py --generate-data          Generate sample data only
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['backtest', 'paper'],
        default='backtest',
        help='Trading mode: backtest or paper trading (default: backtest)'
    )
    
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate sample data and exit'
    )
    
    parser.add_argument(
        '--periods',
        type=int,
        default=BACKTEST_PERIODS,
        help=f'Number of 15-minute periods to simulate (default: {BACKTEST_PERIODS})'
    )
    
    args = parser.parse_args()
    
    # Handle data generation only
    if args.generate_data:
        generate_sample_data(args.periods)
        return
    
    # Run simulation based on mode
    try:
        if args.mode == 'backtest':
            run_backtest()
        elif args.mode == 'paper':
            run_paper_trade()
    except KeyboardInterrupt:
        print("\n\n⚠ Simulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
