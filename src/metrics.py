"""
Performance metrics calculation for trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix, classification_report


class MetricsCalculator:
    """Calculate trading performance metrics."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_all_metrics(self, trades_df: pd.DataFrame, starting_balance: float) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades_df: DataFrame with executed trades
            starting_balance: Initial capital
            
        Returns:
            Dictionary with all performance metrics
        """
        if trades_df.empty:
            return self._empty_metrics()
        
        # Basic metrics
        total_pnl = trades_df['pnl'].sum()
        final_balance = starting_balance + total_pnl
        num_trades = len(trades_df)
        
        # Win rate
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = num_wins / num_trades if num_trades > 0 else 0.0
        
        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if num_wins > 0 else 0.0
        avg_loss = losing_trades['pnl'].mean() if num_losses > 0 else 0.0
        
        # Drawdown
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
        trades_df['drawdown'] = trades_df['running_max'] - trades_df['cumulative_pnl']
        max_drawdown = trades_df['drawdown'].max()
        max_drawdown_pct = (max_drawdown / starting_balance) * 100 if starting_balance > 0 else 0.0
        
        # Sharpe ratio (simplified)
        returns = trades_df['pnl'] / starting_balance
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        
        # Confusion matrix for Up/Down predictions
        y_true = trades_df['actual'].values
        y_pred = trades_df['predicted'].values
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['UP', 'DOWN'])
        
        # True Positives (predicted UP, actual UP)
        # True Negatives (predicted DOWN, actual DOWN)
        # False Positives (predicted UP, actual DOWN)
        # False Negatives (predicted DOWN, actual UP)
        tn, fp, fn, tp = 0, 0, 0, 0
        
        for true, pred in zip(y_true, y_pred):
            if true == 'UP' and pred == 'UP':
                tp += 1
            elif true == 'DOWN' and pred == 'DOWN':
                tn += 1
            elif true == 'DOWN' and pred == 'UP':
                fp += 1
            elif true == 'UP' and pred == 'DOWN':
                fn += 1
        
        # Accuracy
        accuracy = (tp + tn) / num_trades if num_trades > 0 else 0.0
        
        # Precision and Recall
        precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_down = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        recall_down = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics = {
            'total_pnl': total_pnl,
            'final_balance': final_balance,
            'return_pct': (total_pnl / starting_balance) * 100,
            'num_trades': num_trades,
            'num_wins': num_wins,
            'num_losses': num_losses,
            'win_rate': win_rate * 100,  # as percentage
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'accuracy': accuracy * 100,  # as percentage
            'confusion_matrix': {
                'TP': tp,  # True Positives (UP predicted, UP actual)
                'TN': tn,  # True Negatives (DOWN predicted, DOWN actual)
                'FP': fp,  # False Positives (UP predicted, DOWN actual)
                'FN': fn   # False Negatives (DOWN predicted, UP actual)
            },
            'precision_up': precision_up,
            'recall_up': recall_up,
            'precision_down': precision_down,
            'recall_down': recall_down
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no trades executed."""
        return {
            'total_pnl': 0.0,
            'final_balance': 0.0,
            'return_pct': 0.0,
            'num_trades': 0,
            'num_wins': 0,
            'num_losses': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'accuracy': 0.0,
            'confusion_matrix': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
            'precision_up': 0.0,
            'recall_up': 0.0,
            'precision_down': 0.0,
            'recall_down': 0.0
        }
    
    def print_metrics(self, metrics: Dict):
        """
        Print metrics in a readable format.
        
        Args:
            metrics: Dictionary of calculated metrics
        """
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\nFinancial Performance:")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"  Final Balance: ${metrics['final_balance']:.2f}")
        print(f"  Return: {metrics['return_pct']:.2f}%")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades: {metrics['num_trades']}")
        print(f"  Winning Trades: {metrics['num_wins']}")
        print(f"  Losing Trades: {metrics['num_losses']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Avg Win: ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        print(f"\nPrediction Accuracy:")
        print(f"  Overall Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Precision (UP): {metrics['precision_up']:.2f}")
        print(f"  Recall (UP): {metrics['recall_up']:.2f}")
        print(f"  Precision (DOWN): {metrics['precision_down']:.2f}")
        print(f"  Recall (DOWN): {metrics['recall_down']:.2f}")
        
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted UP  Predicted DOWN")
        print(f"  Actual UP      {cm['TP']:^12}  {cm['FN']:^14}")
        print(f"  Actual DOWN    {cm['FP']:^12}  {cm['TN']:^14}")
        print(f"\n  TP (True Positive): {cm['TP']} - Correctly predicted UP")
        print(f"  TN (True Negative): {cm['TN']} - Correctly predicted DOWN")
        print(f"  FP (False Positive): {cm['FP']} - Predicted UP, was DOWN")
        print(f"  FN (False Negative): {cm['FN']} - Predicted DOWN, was UP")
        
        print("="*60 + "\n")
