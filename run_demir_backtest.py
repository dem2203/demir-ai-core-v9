# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - Full Strategy Backtest
Tests the actual DEMIR AI multi-indicator approach on historical data.
"""
import asyncio
import sys
import os

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from src.brain.backtest_engine import get_backtest_engine, BacktestMetrics
from typing import Optional, Dict


async def demir_ai_strategy(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """
    DEMIR AI Multi-Indicator Strategy
    
    Combines:
    1. RSI (Momentum)
    2. MACD (Trend)
    3. Bollinger Bands (Volatility/Squeeze)
    4. Volume (Confirmation)
    5. Multi-timeframe EMA alignment
    
    Only trades when multiple confirmations align.
    """
    if len(df) < 50:
        return None
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    current_price = close.iloc[-1]
    
    # === INDICATOR CALCULATIONS ===
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line
    macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]
    macd_cross_up = macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]
    macd_cross_down = macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]
    
    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid * 100
    is_squeeze = bb_width.iloc[-1] < bb_width.rolling(50).mean().iloc[-1] * 0.8
    price_near_lower = current_price < bb_lower.iloc[-1] * 1.01
    price_near_upper = current_price > bb_upper.iloc[-1] * 0.99
    
    # Volume Analysis
    avg_volume = volume.rolling(20).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    volume_spike = current_volume > avg_volume * 1.5
    
    # Multi-timeframe EMAs
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
    
    # === SIGNAL LOGIC ===
    
    buy_score = 0
    sell_score = 0
    
    # RSI signals
    if current_rsi < 35:
        buy_score += 2  # Oversold
    elif current_rsi < 45:
        buy_score += 1
    elif current_rsi > 65:
        sell_score += 2  # Overbought
    elif current_rsi > 55:
        sell_score += 1
    
    # MACD signals
    if macd_cross_up:
        buy_score += 3  # Strong bullish
    elif macd_bullish:
        buy_score += 1
    if macd_cross_down:
        sell_score += 3  # Strong bearish
    elif not macd_bullish:
        sell_score += 1
    
    # Bollinger signals
    if price_near_lower and not ema_bearish:
        buy_score += 2  # Potential bounce
    if price_near_upper and not ema_bullish:
        sell_score += 2  # Potential rejection
    if is_squeeze:
        # Squeeze = big move coming, direction based on other indicators
        if buy_score > sell_score:
            buy_score += 1
        elif sell_score > buy_score:
            sell_score += 1
    
    # EMA alignment
    if ema_bullish:
        buy_score += 2
    if ema_bearish:
        sell_score += 2
    
    # Volume confirmation
    if volume_spike:
        if buy_score > sell_score:
            buy_score += 1
        elif sell_score > buy_score:
            sell_score += 1
    
    # === DECISION ===
    
    MIN_SCORE = 5  # Need strong confirmation
    
    if buy_score >= MIN_SCORE and buy_score > sell_score + 2:
        # Calculate R/R optimized SL/TP
        sl_distance = atr * 1.5
        tp_distance = atr * 3.0  # 2:1 R/R minimum
        
        return {
            'action': 'BUY',
            'confidence': min(95, 50 + buy_score * 5),
            'sl': current_price - sl_distance,
            'tp': current_price + tp_distance
        }
    
    elif sell_score >= MIN_SCORE and sell_score > buy_score + 2:
        sl_distance = atr * 1.5
        tp_distance = atr * 3.0
        
        return {
            'action': 'SELL',
            'confidence': min(95, 50 + sell_score * 5),
            'sl': current_price + sl_distance,
            'tp': current_price - tp_distance
        }
    
    return None


async def run_demir_backtest():
    print("="*60)
    print("[BACKTEST] DEMIR AI FULL STRATEGY - 180 Days")
    print("="*60)
    
    engine = get_backtest_engine()
    
    print("\n[INFO] Testing DEMIR AI multi-indicator strategy...")
    print("[INFO] Indicators: RSI + MACD + Bollinger + Volume + EMA Alignment")
    print("[INFO] Minimum score: 5 (strong confirmation required)")
    
    strategies = {
        "DEMIR AI (Multi-Indicator)": demir_ai_strategy,
    }
    
    results = await engine.compare_strategies("BTCUSDT", strategies, days=180)
    
    metrics = results["DEMIR AI (Multi-Indicator)"]
    print(metrics.summary())
    
    # Verdict
    print("\n" + "="*60)
    if metrics.is_acceptable():
        print("[SUCCESS] Strategy meets professional standards!")
    else:
        print("[ANALYSIS] Strategy needs optimization:")
        if metrics.win_rate < 45:
            print(f"   - Win rate: {metrics.win_rate:.1f}% (target: >=45%)")
        if metrics.profit_factor < 1.2:
            print(f"   - Profit factor: {metrics.profit_factor:.2f} (target: >=1.2)")
        if metrics.max_drawdown_pct > 25:
            print(f"   - Max drawdown: {metrics.max_drawdown_pct:.1f}% (target: <=25%)")
        if metrics.sharpe_ratio < 0.5:
            print(f"   - Sharpe ratio: {metrics.sharpe_ratio:.2f} (target: >=0.5)")
    
    print("\n[NOTE] This backtest uses ONLY technical indicators.")
    print("[NOTE] Live DEMIR AI also uses: Whale data, Orderbook, LSTM, AI Council")
    print("[NOTE] These additional data sources can significantly improve performance.")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_demir_backtest())
