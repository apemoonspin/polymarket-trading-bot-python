"""
Tests for the Polymarket Copy Trading Bot simulator.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.market import MarketSimulator
from src.leader import LeaderTrader, MarketEvaluator
from src.simulator import TradingSimulator
from src.metrics import MetricsCalculator
from src.data_loader import DataLoader


class TestMarketSimulator(unittest.TestCase):
    """Test market data generation."""
    
    def setUp(self):
        self.simulator = MarketSimulator(seed=42)
    
    def test_generate_market_data(self):
        """Test that market data is generated correctly."""
        start_time = datetime(2024, 1, 1, 0, 0)
        data = self.simulator.generate_market_data(100, start_time)
        
        self.assertEqual(len(data), 100)
        self.assertIn('timestamp', data.columns)
        self.assertIn('btc_price', data.columns)
        self.assertIn('actual_outcome', data.columns)
        
        # Check outcomes are UP or DOWN
        self.assertTrue(all(data['actual_outcome'].isin(['UP', 'DOWN'])))
    
    def test_data_continuity(self):
        """Test that timestamps are continuous."""
        start_time = datetime(2024, 1, 1, 0, 0)
        data = self.simulator.generate_market_data(10, start_time)
        
        # Check 15-minute intervals
        for i in range(1, len(data)):
            time_diff = (data.iloc[i]['timestamp'] - data.iloc[i-1]['timestamp']).total_seconds()
            self.assertEqual(time_diff, 15 * 60)


class TestLeaderTrader(unittest.TestCase):
    """Test leader trader logic."""
    
    def setUp(self):
        self.leader = LeaderTrader(skill_level=0.7, seed=42)
        self.market_sim = MarketSimulator(seed=42)
        self.market_data = self.market_sim.generate_market_data(50, datetime.now())
    
    def test_generate_trades(self):
        """Test that leader generates trades."""
        trades = self.leader.generate_trades(self.market_data)
        
        self.assertEqual(len(trades), len(self.market_data))
        self.assertIn('timestamp', trades.columns)
        self.assertIn('side', trades.columns)
        self.assertIn('confidence', trades.columns)
        
        # Check sides are LONG or SHORT
        self.assertTrue(all(trades['side'].isin(['LONG', 'SHORT'])))
    
    def test_get_signal(self):
        """Test getting signal for specific timestamp."""
        trades = self.leader.generate_trades(self.market_data)
        
        timestamp = self.market_data.iloc[0]['timestamp']
        signal = self.leader.get_signal(timestamp)
        
        self.assertIsNotNone(signal)
        self.assertIn('direction', signal)
        self.assertIn('confidence', signal)


class TestMarketEvaluator(unittest.TestCase):
    """Test market outcome evaluation."""
    
    def test_determine_outcome(self):
        """Test outcome determination."""
        # Price goes up
        outcome = MarketEvaluator.determine_outcome(100, 105)
        self.assertEqual(outcome, "UP")
        
        # Price goes down
        outcome = MarketEvaluator.determine_outcome(100, 95)
        self.assertEqual(outcome, "DOWN")
    
    def test_is_prediction_correct(self):
        """Test prediction correctness."""
        self.assertTrue(MarketEvaluator.is_prediction_correct("UP", "UP"))
        self.assertTrue(MarketEvaluator.is_prediction_correct("DOWN", "DOWN"))
        self.assertFalse(MarketEvaluator.is_prediction_correct("UP", "DOWN"))
        self.assertFalse(MarketEvaluator.is_prediction_correct("DOWN", "UP"))


class TestTradingSimulator(unittest.TestCase):
    """Test trading simulator."""
    
    def setUp(self):
        self.market_sim = MarketSimulator(seed=42)
        self.leader = LeaderTrader(skill_level=0.65, seed=42)
        
        start_time = datetime(2024, 1, 1, 0, 0)
        self.market_data = self.market_sim.generate_market_data(100, start_time)
        self.leader_trades = self.leader.generate_trades(self.market_data)
        
        self.simulator = TradingSimulator(
            starting_balance=10000.0,
            max_position_size=0.1,
            trading_fee=0.02,
            confidence_threshold=0.5
        )
    
    def test_run_backtest(self):
        """Test backtest execution."""
        trades_df = self.simulator.run_backtest(self.market_data, self.leader_trades)
        
        self.assertGreater(len(trades_df), 0)
        self.assertIn('pnl', trades_df.columns)
        self.assertIn('correct', trades_df.columns)
        
        # Final balance should be different from starting
        final_balance = self.simulator.get_final_balance()
        self.assertNotEqual(final_balance, 10000.0)


class TestMetricsCalculator(unittest.TestCase):
    """Test metrics calculation."""
    
    def setUp(self):
        self.calc = MetricsCalculator()
        
        # Create sample trades data
        self.trades_df = pd.DataFrame({
            'pnl': [100, -50, 75, -25, 150],
            'predicted': ['UP', 'DOWN', 'UP', 'UP', 'DOWN'],
            'actual': ['UP', 'UP', 'UP', 'DOWN', 'DOWN'],
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='15min')
        })
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = self.calc.calculate_all_metrics(self.trades_df, 10000.0)
        
        # Check all required metrics exist
        self.assertIn('total_pnl', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Win rate should be reasonable
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertLessEqual(metrics['win_rate'], 100)
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        metrics = self.calc.calculate_all_metrics(self.trades_df, 10000.0)
        cm = metrics['confusion_matrix']
        
        self.assertIn('TP', cm)
        self.assertIn('TN', cm)
        self.assertIn('FP', cm)
        self.assertIn('FN', cm)
        
        # Total should equal number of trades
        total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
        self.assertEqual(total, len(self.trades_df))


class TestDataLoader(unittest.TestCase):
    """Test data loading functionality."""
    
    def setUp(self):
        self.loader = DataLoader(data_dir='test_data', resolution_minutes=15)
        os.makedirs('test_data', exist_ok=True)
    
    def tearDown(self):
        # Clean up test files
        import shutil
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
    
    def test_save_and_load_prices(self):
        """Test saving and loading BTC prices."""
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='15min'),
            'btc_price': np.random.uniform(40000, 45000, 10),
            'actual_outcome': np.random.choice(['UP', 'DOWN'], 10)
        })
        
        filepath = 'test_data/test_prices.csv'
        self.loader.save_btc_prices(df, filepath)
        
        loaded_df = self.loader.load_btc_prices(filepath)
        self.assertEqual(len(loaded_df), len(df))


if __name__ == '__main__':
    unittest.main()
