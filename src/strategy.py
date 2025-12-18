"""
Copy trading strategy that mirrors the leader's positions.
"""
import pandas as pd


class CopyTradingStrategy:
    """Strategy that copies the leader trader's positions."""
    
    def __init__(self, starting_balance: float = 10000.0, max_position_size: float = 0.1,
                 trading_fee: float = 0.02, confidence_threshold: float = 0.5):
        """
        Initialize the copy trading strategy.
        
        Args:
            starting_balance: Initial capital in USD
            max_position_size: Maximum position size as fraction of balance (e.g., 0.1 = 10%)
            trading_fee: Fee per trade (as fraction, e.g., 0.02 = 2%)
            confidence_threshold: Minimum confidence to take a trade (0-1)
        """
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.max_position_size = max_position_size
        self.trading_fee = trading_fee
        self.confidence_threshold = confidence_threshold
        self.positions = []
        self.closed_positions = []
    
    def copy_trades(self, leader_trades: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute trades by copying the leader's positions with confidence-based sizing.
        
        Args:
            leader_trades: Leader's trades with columns: timestamp, side, amount, entry_price, confidence
            market_data: Market data with columns: timestamp, btc_price, actual_outcome
            
        Returns:
            DataFrame with executed trades and outcomes
        """
        trades_executed = []
        
        for idx, leader_trade in leader_trades.iterrows():
            # Skip if confidence below threshold
            if leader_trade['confidence'] < self.confidence_threshold:
                continue
            
            # Get corresponding market data
            market_row = market_data[market_data['timestamp'] == leader_trade['timestamp']]
            if market_row.empty:
                continue
            
            market_row = market_row.iloc[0]
            
            # Calculate position size: balance * confidence * max_position_size
            # No leverage - position size limited by available balance
            position_size = self.balance * leader_trade['confidence'] * self.max_position_size
            position_size = min(position_size, self.balance * 0.95)  # Keep 5% reserve
            
            if position_size < 1.0:  # Not enough balance
                continue
            
            # Apply trading fee on entry
            fee = position_size * self.trading_fee
            effective_amount = position_size - fee
            
            # Determine outcome and PnL
            predicted = "UP" if leader_trade['side'] == "LONG" else "DOWN"
            actual = market_row['actual_outcome']
            
            # Prediction market payoff model (binary contract):
            # - entry_price is the cost per $1 payoff of a contract that pays 1 if the outcome occurs, 0 otherwise.
            # - We stake `effective_amount` dollars at this entry_price.
            #   * If the prediction is correct, profit on the stake is: effective_amount * (1 - entry_price)
            #   * If the prediction is wrong, loss on the stake is:   effective_amount (full investment lost)
            if predicted == actual:
                # Win: get back investment + profit
                pnl = effective_amount * (1 - leader_trade['entry_price'])
            else:
                # Loss: lose the entire amount invested
                pnl = -effective_amount
            
            # Apply exit fee on profits
            exit_fee = abs(pnl) * self.trading_fee if pnl > 0 else 0
            pnl -= exit_fee
            
            # Update balance
            self.balance += pnl
            
            # Record trade
            trades_executed.append({
                'timestamp': leader_trade['timestamp'],
                'side': leader_trade['side'],
                'amount': position_size,
                'entry_price': leader_trade['entry_price'],
                'confidence': leader_trade['confidence'],
                'predicted': predicted,
                'actual': actual,
                'pnl': pnl,
                'balance': self.balance,
                'correct': predicted == actual
            })
        
        return pd.DataFrame(trades_executed)
    
    def get_final_balance(self) -> float:
        """Get final account balance."""
        return self.balance
    
    def get_total_pnl(self) -> float:
        """Get total profit/loss."""
        return self.balance - self.starting_balance
    
    def reset(self):
        """Reset the strategy to initial state."""
        self.balance = self.starting_balance
        self.positions = []
        self.closed_positions = []

