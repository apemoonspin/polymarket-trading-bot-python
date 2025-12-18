"""
Data loader for BTC prices and leader trades.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os


class DataLoader:
    """Load and manage market data and leader trades."""
    
    def __init__(self, data_dir: str = "data", resolution_minutes: int = 15):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files
            resolution_minutes: Expected time resolution in minutes (default: 15)
        """
        self.data_dir = data_dir
        self.resolution_minutes = resolution_minutes
        Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    def load_btc_prices(self, filepath: str) -> pd.DataFrame:
        """
        Load BTC price data from CSV and validate/resample to 15-minute intervals.
        
        Args:
            filepath: Path to BTC prices CSV file
            
        Returns:
            DataFrame with columns: timestamp, btc_price, actual_outcome
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"BTC prices file not found: {filepath}")
        
        # Load CSV
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        
        # Set timestamp as index for resampling
        df = df.set_index('timestamp')
        
        # Validate and resample to 15-minute intervals
        df = self._resample_to_interval(df)
        
        # Reset index to have timestamp as column
        df = df.reset_index()
        
        return df
    
    def _resample_to_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample or validate data to match the expected time interval.
        
        Args:
            df: DataFrame with timestamp index
            
        Returns:
            Resampled DataFrame
        """
        # Check if data already has correct interval
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            median_interval = time_diffs.median()
            expected_interval = pd.Timedelta(minutes=self.resolution_minutes)
            
            # If interval matches (within tolerance), return as-is
            if abs((median_interval - expected_interval).total_seconds()) < 60:
                return df
        
        # Resample to the expected interval
        freq = f'{self.resolution_minutes}min'
        
        # For numeric columns, use forward fill
        resampled = df.resample(freq).first()
        resampled = resampled.fillna(method='ffill')
        
        return resampled
    
    def load_leader_trades(self, filepath: str) -> pd.DataFrame:
        """
        Load leader trade data from CSV.
        
        Args:
            filepath: Path to leader trades CSV file
            
        Returns:
            DataFrame with columns: timestamp, side, amount, entry_price
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Leader trades file not found: {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        return df
    
    def align_trades_to_prices(self, trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align leader trades to price timestamps using nearest timestamp matching.
        
        Args:
            trades_df: DataFrame with leader trades
            prices_df: DataFrame with BTC prices
            
        Returns:
            Aligned trades DataFrame with matching timestamps from prices
        """
        # Get valid price timestamps
        valid_timestamps = set(prices_df['timestamp'])
        
        # Align each trade to nearest valid timestamp
        aligned_trades = []
        
        for idx, trade in trades_df.iterrows():
            trade_time = trade['timestamp']
            
            # Find nearest price timestamp
            nearest_timestamp = min(valid_timestamps, 
                                   key=lambda x: abs((x - trade_time).total_seconds()))
            
            # Create aligned trade
            aligned_trade = trade.copy()
            aligned_trade['timestamp'] = nearest_timestamp
            aligned_trade['original_timestamp'] = trade_time
            aligned_trades.append(aligned_trade)
        
        return pd.DataFrame(aligned_trades)
    
    def save_btc_prices(self, df: pd.DataFrame, filepath: str):
        """
        Save BTC price data to CSV.
        
        Args:
            df: DataFrame with BTC price data
            filepath: Path to save CSV file
        """
        df.to_csv(filepath, index=False)
        print(f"Saved BTC prices to {filepath}")
    
    def save_leader_trades(self, df: pd.DataFrame, filepath: str):
        """
        Save leader trade data to CSV.
        
        Args:
            df: DataFrame with leader trade data
            filepath: Path to save CSV file
        """
        df.to_csv(filepath, index=False)
        print(f"Saved leader trades to {filepath}")
    
    def generate_sample_data(self, num_periods: int, start_time: datetime, 
                           btc_filepath: str, trades_filepath: str, seed: int = 42):
        """
        Generate and save sample data for testing.
        
        Args:
            num_periods: Number of 15-minute periods
            start_time: Starting timestamp
            btc_filepath: Path to save BTC prices
            trades_filepath: Path to save leader trades
            seed: Random seed
        """
        from src.market import MarketSimulator
        from src.leader import LeaderTrader
        
        # Generate market data
        market_sim = MarketSimulator(seed=seed)
        market_data = market_sim.generate_market_data(num_periods, start_time)
        
        # Generate leader trades
        leader = LeaderTrader(skill_level=0.65, seed=seed)
        leader_trades = leader.generate_trades(market_data)
        
        # Save to CSV
        self.save_btc_prices(market_data, btc_filepath)
        self.save_leader_trades(leader_trades, trades_filepath)
        
        return market_data, leader_trades
