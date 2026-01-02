# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - 5-Year Backtest with Monte Carlo
================================================
Run comprehensive long-term strategy validation.
"""
import asyncio
import sys
import os

# Force UTF-8 on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from src.brain.backtest_engine import get_backtest_engine
from typing import Optional, Dict


async def demir_multi_indicator(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """DEMIR AI Multi-Indicator Strategy for 4h timeframe"""
    if len(df) < 50:
        return None
    
    close = df['close']
    high = df['high']
    low = df['low']
    current_price = close.iloc[-1]
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]
    macd_cross_up = macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]
    macd_cross_down = macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]
    
    # EMAs
    ema9 = close.ewm(span=9).mean()
    ema21 = close.ewm(span=21).mean()
    ema50 = close.ewm(span=50).mean()
    
    ema_bullish = ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]
    ema_bearish = ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]
    
    # ATR for SL/TP
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    # Signal logic
    buy_score = 0
    sell_score = 0
    
    if current_rsi < 35:
        buy_score += 2
    elif current_rsi > 65:
        sell_score += 2
    
    if macd_cross_up:
        buy_score += 3
    elif macd_cross_down:
        sell_score += 3
    elif macd_bullish:
        buy_score += 1
    else:
        sell_score += 1
    
    if ema_bullish:
        buy_score += 2
    elif ema_bearish:
        sell_score += 2
    
    MIN_SCORE = 4
    
    if buy_score >= MIN_SCORE and buy_score > sell_score + 1:
        return {
            'action': 'BUY',
            'confidence': min(90, 50 + buy_score * 5),
            'sl': current_price - atr * 2,
            'tp': current_price + atr * 4
        }
    elif sell_score >= MIN_SCORE and sell_score > buy_score + 1:
        return {
            'action': 'SELL',
            'confidence': min(90, 50 + sell_score * 5),
            'sl': current_price + atr * 2,
            'tp': current_price - atr * 4
        }
    
    return None


async def run_5year_test():
    print("=" * 70)
    print("[BACKTEST] DEMIR AI - 5-YEAR BACKTEST + MONTE CARLO")
    print("=" * 70)
    
    engine = get_backtest_engine()
    
    print("\n[INFO] Loading 5 years of BTCUSDT 4h data...")
    print("[INFO] This may take 2-3 minutes...\n")
    
    results = await engine.run_5year_backtest(
        symbol="BTCUSDT",
        signal_generator=demir_multi_indicator,
        initial_equity=10000,
        run_monte_carlo=True
    )
    
    print(results.summary())
    
    # Final verdict
    print("\n" + "=" * 70)
    if results.is_acceptable():
        print("[SUCCESS] Strategy meets institutional standards!")
    else:
        print("[ANALYSIS] Strategy needs optimization:")
        if results.win_rate < 45:
            print(f"   - Win Rate: {results.win_rate:.1f}% (target: >=45%)")
        if results.profit_factor < 1.2:
            print(f"   - Profit Factor: {results.profit_factor:.2f} (target: >=1.2)")
        if results.max_drawdown_pct > 25:
            print(f"   - Max Drawdown: {results.max_drawdown_pct:.1f}% (target: <=25%)")
        if results.sharpe_ratio < 0.5:
            print(f"   - Sharpe Ratio: {results.sharpe_ratio:.2f} (target: >=0.5)")
    
    if results.monte_carlo:
        mc = results.monte_carlo
        print(f"\n[MONTE CARLO] Key Insights:")
        print(f"   - Profit Probability: {mc.probability_profit:.1f}%")
        print(f"   - 20% Drawdown Risk: {mc.probability_drawdown_20:.1f}%")
        print(f"   - Expected Median P&L: ${mc.median_pnl:+,.0f}")
    
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_5year_test())
