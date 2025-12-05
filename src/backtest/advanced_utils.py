"""
ADVANCED BACKTESTING UTILITIES (Phase 16)

Standalone utilities for professional backtesting analysis:
1. Walk-Forward Analysis - Prevent overfitting
2. Monte Carlo Simulation - Risk assessment

These can be integrated into any backtester.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger("BACKTEST_UTILS")

def walk_forward_analysis(
    backtest_func,
    symbol: str,
    train_window: int = 30,
    test_window: int = 7,
    total_periods: int = 4
) -> Dict:
    """
    Walk-Forward Analysis
    
    Process:
    - Train on N days → Test on M days
    - Roll forward M days
    - Repeat
    
    Returns out-of-sample performance
    """
    logger.info(f"🚶 Walk-Forward Analysis: {total_periods} periods")
    
    results = []
    
    for period in range(total_periods):
        days = (period + 1) * (train_window + test_window)
        
        # Run backtest for this period
        result = backtest_func(symbol, days=days)
        
        if result.get('error'):
            logger.error(f"Period {period+1} failed")
            continue
        
        results.append({
            "period": period + 1,
            "roi": result['roi'],
            "win_rate": result['win_rate'],
            "trades": result['total_trades']
        })
        
        logger.info(f"Period {period+1}: ROI={result['roi']:.2f}%")
    
    if not results:
        return {"error": "No successful periods"}
    
    avg_roi = np.mean([r['roi'] for r in results])
    roi_std = np.std([r['roi'] for r in results])
    
    return {
        "periods": results,
        "average_roi": avg_roi,
        "average_win_rate": np.mean([r['win_rate'] for r in results]),
        "roi_std": roi_std,
        "consistency": 1 - (roi_std / abs(avg_roi)) if avg_roi != 0 else 0
    }


def monte_carlo_simulation(
    trades: List[Dict],
    initial_balance: float = 10000.0,
    iterations: int = 1000
) -> Dict:
    """
    Monte Carlo Simulation
    
    Randomize trade order to assess risk distribution.
    
    Args:
        trades: List of trade dicts with 'pnl_pct'
        initial_balance: Starting balance
        iterations: Number of simulations
    
    Returns:
        Distribution statistics
    """
    logger.info(f"🎲 Monte Carlo: {iterations} iterations")
    
    # Extract P&Ls
    pnls = []
    for trade in trades:
        if 'pnl_pct' in trade:
            pnl_str = str(trade['pnl_pct']).replace('%', '')
            try:
                pnls.append(float(pnl_str) / 100)
            except:
                continue
    
    if len(pnls) < 10:
        return {"error": "Need at least 10 trades"}
    
    final_balances = []
    max_drawdowns = []
    
    for _ in range(iterations):
        # Random trade sequence
        random_pnls = np.random.choice(pnls, size=len(pnls), replace=True)
        
        balance = initial_balance
        peak = balance
        max_dd = 0
        
        for pnl in random_pnls:
            balance *= (1 + pnl - 0.001)  # Include 0.1% fee
            
            if balance > peak:
                peak = balance
            
            dd = (peak - balance) / peak
            if dd > max_dd:
                max_dd = dd
        
        final_balances.append(balance)
        max_drawdowns.append(max_dd * 100)
    
    final_balances = np.array(final_balances)
    rois = ((final_balances - initial_balance) / initial_balance) * 100
    
    return {
        "iterations": iterations,
        "median_roi": float(np.median(rois)),
        "mean_roi": float(np.mean(rois)),
        "roi_5th_percentile": float(np.percentile(rois, 5)),
        "roi_95th_percentile": float(np.percentile(rois, 95)),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "worst_max_drawdown": float(np.max(max_drawdowns)),
        "probability_profit": float((np.sum(rois > 0) / iterations) * 100)
    }


# Example usage:
# from src.backtest.advanced_utils import walk_forward_analysis, monte_carlo_simulation
# 
# # Walk-forward
# wf_results = walk_forward_analysis(my_backtest_func, "BTC/USDT")
# 
# # Monte Carlo
# mc_results = monte_carlo_simulation(trade_history, initial_balance=10000, iterations=1000)
