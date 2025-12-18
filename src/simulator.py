"""
Trading simulator for backtesting and paper trading.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class TradingSimulator:
    """Simulates trading by iterating through 15-minute candles."""
    
    def __init__(self, starting_balance: float, max_position_size: float, 
                 trading_fee: float, confidence_threshold: float = 0.5):
        """
        Initialize the trading simulator.
        
        Args:
            starting_balance: Initial capital in USD
            max_position_size: Maximum position size as fraction of balance
            trading_fee: Fee per trade (as fraction, e.g., 0.02 = 2%)
            confidence_threshold: Minimum confidence to take a trade
        """
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.max_position_size = max_position_size
        self.trading_fee = trading_fee
        self.confidence_threshold = confidence_threshold
        self.trades_executed = []
    
    def run_backtest(self, market_data: pd.DataFrame, leader_trades: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest simulation through historical data.
        
        Args:
            market_data: DataFrame with columns: timestamp, btc_price, actual_outcome
            leader_trades: DataFrame with columns: timestamp, side, amount, entry_price, confidence
            
        Returns:
            DataFrame with executed trades and results
        """
        self.balance = self.starting_balance
        self.trades_executed = []
        
        # Iterate through each candle
        for idx in range(len(market_data) - 1):
            current_candle = market_data.iloc[idx]
            next_candle = market_data.iloc[idx + 1]
            
            timestamp = current_candle['timestamp']
            price_open = current_candle['btc_price']
            price_close = next_candle['btc_price']
            actual_outcome = current_candle['actual_outcome']
            
            # Get leader's signal for this timestamp
            leader_trade = leader_trades[leader_trades['timestamp'] == timestamp]
            
            if leader_trade.empty:
                continue
            
            leader_trade = leader_trade.iloc[0]
            
            # Execute trade using copy trading strategy
            trade_result = self._execute_trade(
                timestamp=timestamp,
                leader_side=leader_trade['side'],
                leader_confidence=leader_trade['confidence'],
                entry_price=leader_trade['entry_price'],
                price_open=price_open,
                price_close=price_close,
                actual_outcome=actual_outcome
            )
            
            if trade_result is not None:
                self.trades_executed.append(trade_result)
        
        return pd.DataFrame(self.trades_executed)
    
    def _execute_trade(self, timestamp, leader_side: str, leader_confidence: float,
                      entry_price: float, price_open: float, price_close: float,
                      actual_outcome: str) -> Dict:
        """
        Execute a single trade with copy trading strategy.
        
        Args:
            timestamp: Trade timestamp
            leader_side: Leader's position (LONG or SHORT)
            leader_confidence: Leader's confidence (0-1)
            entry_price: Entry price/odds
            price_open: Candle open price
            price_close: Candle close price
            actual_outcome: Actual market outcome
            
        Returns:
            Dictionary with trade results, or None if trade skipped
        """
        # Skip if confidence below threshold
        if leader_confidence < self.confidence_threshold:
            return None
        
        # Calculate position size: balance * confidence * max_position_size
        # max_position_size is already a fraction, confidence scales it further
        position_size = self.balance * leader_confidence * self.max_position_size
        
        # Ensure we don't exceed available balance
        position_size = min(position_size, self.balance * 0.95)  # Keep 5% reserve
        
        if position_size < 1.0:  # Minimum trade size
            return None
        
        # Apply entry fee
        entry_fee = position_size * self.trading_fee
        effective_position = position_size - entry_fee
        
        # Determine predicted outcome
        predicted = "UP" if leader_side == "LONG" else "DOWN"
        
        # Calculate PnL based on market outcome
        # Simplified Polymarket model: win = get back (1-entry_price) ratio, lose = lose bet
        if predicted == actual_outcome:
            # Win: profit based on odds
            pnl = effective_position * (1 - entry_price)
        else:
            # Loss: lose the effective position
            pnl = -effective_position
        
        # Apply exit fee on profits
        if pnl > 0:
            exit_fee = pnl * self.trading_fee
            pnl -= exit_fee
        
        # Update balance
        self.balance += pnl
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'side': leader_side,
            'amount': position_size,
            'effective_amount': effective_position,
            'entry_price': entry_price,
            'confidence': leader_confidence,
            'price_open': price_open,
            'price_close': price_close,
            'predicted': predicted,
            'actual': actual_outcome,
            'pnl': pnl,
            'balance': self.balance,
            'correct': predicted == actual_outcome
        }
        
        return trade_record
    
    def run_paper_trade(self, market_data: pd.DataFrame, leader_trades: pd.DataFrame) -> pd.DataFrame:
        """
        Run paper trading simulation (same as backtest but could add real-time features).
        
        Args:
            market_data: DataFrame with market data
            leader_trades: DataFrame with leader trades
            
        Returns:
            DataFrame with executed trades
        """
        # For now, paper trading uses same logic as backtest
        return self.run_backtest(market_data, leader_trades)
    
    def get_final_balance(self) -> float:
        """Get final account balance."""
        return self.balance
    
    def get_total_return(self) -> float:
        """Get total return as percentage."""
        return ((self.balance - self.starting_balance) / self.starting_balance) * 100
    
    def reset(self):
        """Reset simulator to initial state."""
        self.balance = self.starting_balance
        self.trades_executed = []
