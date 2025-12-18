"""
Leader trader simulation with configurable skill level.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


class LeaderTrader:
    """Simulates a leader trader that others copy."""
    
    def __init__(self, skill_level: float = 0.6, position_size: float = 100.0, seed: int = 42):
        """
        Initialize the leader trader.
        
        Args:
            skill_level: Probability of correct prediction (0.5-1.0)
            position_size: USD amount per trade
            seed: Random seed for reproducibility
        """
        self.skill_level = max(0.5, min(1.0, skill_level))
        self.position_size = position_size
        self.trades_df = None
        np.random.seed(seed)
    
    def generate_trades(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate leader's trading decisions based on market data.
        
        Args:
            market_data: DataFrame with columns: timestamp, btc_price, actual_outcome
            
        Returns:
            DataFrame with columns: timestamp, side, amount, entry_price, confidence
        """
        timestamps = []
        sides = []
        amounts = []
        entry_prices = []
        confidences = []
        
        for idx, row in market_data.iterrows():
            # Leader makes prediction with skill_level accuracy
            correct_prediction = np.random.random() < self.skill_level
            
            if correct_prediction:
                predicted_outcome = row['actual_outcome']
            else:
                # Predict opposite
                predicted_outcome = "DOWN" if row['actual_outcome'] == "UP" else "UP"
            
            # Convert prediction to position side
            # LONG = betting on UP, SHORT = betting on DOWN
            side = "LONG" if predicted_outcome == "UP" else "SHORT"
            
            # Entry price represents the odds (simplified as 0.5 +/- noise)
            entry_price = 0.5 + np.random.uniform(-0.05, 0.05)
            
            # Confidence based on skill level with some randomness
            # Higher skill = higher average confidence
            base_confidence = self.skill_level
            confidence = min(1.0, max(0.0, base_confidence + np.random.uniform(-0.1, 0.1)))
            
            timestamps.append(row['timestamp'])
            sides.append(side)
            amounts.append(self.position_size)
            entry_prices.append(entry_price)
            confidences.append(confidence)
        
        self.trades_df = pd.DataFrame({
            'timestamp': timestamps,
            'side': sides,
            'amount': amounts,
            'entry_price': entry_prices,
            'confidence': confidences
        })
        
        return self.trades_df
    
    def get_signal(self, timestamp: pd.Timestamp) -> Optional[Dict[str, any]]:
        """
        Get trading signal for a specific timestamp.
        
        Args:
            timestamp: The timestamp to get signal for
            
        Returns:
            Dictionary with 'direction' (LONG/SHORT) and 'confidence' (0-1),
            or None if no signal at this timestamp
        """
        if self.trades_df is None or self.trades_df.empty:
            return None
        
        # Find trade at this timestamp
        trade = self.trades_df[self.trades_df['timestamp'] == timestamp]
        
        if trade.empty:
            return None
        
        trade = trade.iloc[0]
        
        return {
            'direction': trade['side'],
            'confidence': trade['confidence'],
            'entry_price': trade['entry_price']
        }
    
    def get_prediction_accuracy(self, trades_df: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """
        Calculate the leader's prediction accuracy.
        
        Args:
            trades_df: Leader's trades
            market_data: Actual market outcomes
            
        Returns:
            Accuracy as a fraction (0-1)
        """
        correct = 0
        total = len(trades_df)
        
        for idx, trade in trades_df.iterrows():
            market_row = market_data[market_data['timestamp'] == trade['timestamp']]
            if market_row.empty:
                continue
            market_row = market_row.iloc[0]
            actual = market_row['actual_outcome']
            predicted = "UP" if trade['side'] == "LONG" else "DOWN"
            
            if actual == predicted:
                correct += 1
        
        return correct / total if total > 0 else 0.0


class MarketEvaluator:
    """Evaluates market outcomes and predictions."""
    
    @staticmethod
    def determine_outcome(price_open: float, price_close: float) -> str:
        """
        Determine market outcome based on two consecutive prices.
        
        Args:
            price_open: Opening price
            price_close: Closing price
            
        Returns:
            "UP" if price increased, "DOWN" if decreased
        """
        if price_close > price_open:
            return "UP"
        else:
            return "DOWN"
    
    @staticmethod
    def is_prediction_correct(predicted: str, actual: str) -> bool:
        """
        Check if leader prediction was correct.
        
        Args:
            predicted: Predicted outcome ("UP" or "DOWN")
            actual: Actual outcome ("UP" or "DOWN")
            
        Returns:
            True if prediction was correct, False otherwise
        """
        return predicted == actual
    
    @staticmethod
    def evaluate_trade(leader_side: str, price_open: float, price_close: float) -> Tuple[str, bool]:
        """
        Evaluate a trade outcome.
        
        Args:
            leader_side: Leader's position side (LONG or SHORT)
            price_open: Opening price
            price_close: Closing price
            
        Returns:
            Tuple of (actual_outcome, is_correct)
        """
        actual = MarketEvaluator.determine_outcome(price_open, price_close)
        predicted = "UP" if leader_side == "LONG" else "DOWN"
        is_correct = MarketEvaluator.is_prediction_correct(predicted, actual)
        
        return actual, is_correct

