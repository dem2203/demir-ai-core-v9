import logging
import json
import os
import numpy as np
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger("PERFORMANCE_TRACKER")

class PerformanceTracker:
    """
    PERFORMANCE TRACKER - Comprehensive Trade Analytics
    
    Calculates and tracks:
    - Win Rate
    - Profit Factor
    - Sharpe Ratio
    - Maximum Drawdown
    - Best/Worst Trades
    - Equity Curve
    """
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.metrics_file = "performance_metrics.json"
        logger.info("📊 Performance Tracker initialized")
    
    def calculate_metrics(self) -> Dict:
        """Calculate all performance metrics from trade history"""
        trades = self._load_trades()
        
        if not trades:
            logger.warning("No trades found for analysis")
            return self._empty_metrics()
        
        closed_trades = [t for t in trades if t.get('status') == 'CLOSED']
        
        if not closed_trades:
            logger.warning("No closed trades found")
            return self._empty_metrics()
        
        # Extract PnL values
        pnls = [float(t.get('realized_pnl', 0)) for t in closed_trades]
        pnl_pcts = [float(t.get('pnl_pct', 0)) for t in closed_trades]
        
        # 1. Win Rate
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_trades = len(closed_trades)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        
        # 2. Profit Factor
        total_profit = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else (total_profit if total_profit > 0 else 0)
        
        # 3. Sharpe Ratio (annualized, assuming daily returns)
        if len(pnl_pcts) > 2:
            mean_return = np.mean(pnl_pcts)
            std_return = np.std(pnl_pcts)
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 4. Maximum Drawdown
        equity_curve = self._calculate_equity_curve(pnls, initial_balance=10000)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # 5. Best/Worst Trades
        best_trade = max(closed_trades, key=lambda t: float(t.get('realized_pnl', 0)))
        worst_trade = min(closed_trades, key=lambda t: float(t.get('realized_pnl', 0)))
        
        # 6. Average Win/Loss
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # 7. Total PnL
        total_pnl = sum(pnls)
        
        metrics = {
            "summary": {
                "total_trades": total_trades,
                "win_rate": round(win_rate, 2),
                "profit_factor": round(profit_factor, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "max_drawdown": round(max_drawdown, 2),
                "total_pnl": round(total_pnl, 2)
            },
            "details": {
                "wins": len(wins),
                "losses": len(losses),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "total_profit": round(total_profit, 2),
                "total_loss": round(total_loss, 2)
            },
            "extremes": {
                "best_trade": {
                    "symbol": best_trade.get('symbol', 'N/A'),
                    "pnl": round(float(best_trade.get('realized_pnl', 0)), 2),
                    "pnl_pct": round(float(best_trade.get('pnl_pct', 0)), 2),
                    "entry_time": best_trade.get('entry_time', 'N/A')
                },
                "worst_trade": {
                    "symbol": worst_trade.get('symbol', 'N/A'),
                    "pnl": round(float(worst_trade.get('realized_pnl', 0)), 2),
                    "pnl_pct": round(float(worst_trade.get('pnl_pct', 0)), 2),
                    "entry_time": worst_trade.get('entry_time', 'N/A')
                }
            },
            "equity_curve": equity_curve,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save to file
        self._save_metrics(metrics)
        
        logger.info(f"📊 Metrics calculated | Win Rate: {win_rate:.1f}% | Profit Factor: {profit_factor:.2f} | Sharpe: {sharpe_ratio:.2f}")
        
        return metrics
    
    def _load_trades(self) -> List[Dict]:
        """Load trades from portfolio.json"""
        if not os.path.exists(self.portfolio_file):
            return []
        
        try:
            with open(self.portfolio_file, 'r') as f:
                portfolio = json.load(f)
                return portfolio.get('trades', [])
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return []
    
    def _calculate_equity_curve(self, pnls: List[float], initial_balance: float = 10000) -> List[float]:
        """Calculate equity curve from PnL series"""
        equity = [initial_balance]
        current = initial_balance
        
        for pnl in pnls:
            current += pnl
            equity.append(current)
        
        return equity
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(equity_curve) < 2:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = ((peak - value) / peak * 100) if peak > 0 else 0
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _save_metrics(self, metrics: Dict):
        """Save metrics to JSON file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            "summary": {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "total_pnl": 0
            },
            "details": {
                "wins": 0,
                "losses": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "total_profit": 0,
                "total_loss": 0
            },
            "extremes": {
                "best_trade": {},
                "worst_trade": {}
            },
            "equity_curve": [10000],
            "last_updated": datetime.now().isoformat()
        }
    
    def get_latest_metrics(self) -> Dict:
        """Get latest cached metrics from file"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return self._empty_metrics()
        return self._empty_metrics()
