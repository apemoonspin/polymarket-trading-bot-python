"""
Market simulation for BTC price movements.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List


class MarketSimulator:
    """Simulates BTC market with 15-minute resolution."""
    
    def __init__(self, initial_price: float = 40000.0, volatility: float = 0.005, seed: int = 42):
        """
        Initialize the market simulator.
        
        Args:
            initial_price: Starting BTC price
            volatility: Price volatility (std dev of returns)
            seed: Random seed for reproducibility
        """
        self.initial_price = initial_price
        self.volatility = volatility
        np.random.seed(seed)
    
    def generate_market_data(self, num_periods: int, start_time: datetime) -> pd.DataFrame:
        """
        Generate simulated market data with 15-minute resolution.
        
        Args:
            num_periods: Number of 15-minute periods to generate
            start_time: Starting timestamp
            
        Returns:
            DataFrame with columns: timestamp, btc_price, actual_outcome
        """
        timestamps = []
        prices = []
        outcomes = []
        
        current_price = self.initial_price
        current_time = start_time
        
        for i in range(num_periods):
            timestamps.append(current_time)
            prices.append(current_price)
            
            # Generate next price with random walk
            price_change_pct = np.random.normal(0, self.volatility)
            next_price = current_price * (1 + price_change_pct)
            
            # Determine outcome (UP or DOWN)
            if next_price > current_price:
                outcome = "UP"
            else:
                outcome = "DOWN"
            
            outcomes.append(outcome)
            
            # Update for next iteration
            current_price = next_price
            current_time += timedelta(minutes=15)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'btc_price': prices,
            'actual_outcome': outcomes
        })
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price returns for market data.
        
        Args:
            df: DataFrame with btc_price column
            
        Returns:
            DataFrame with added returns column
        """
        df['returns'] = df['btc_price'].pct_change()
        return df
