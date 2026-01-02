# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - OPTIMIZED STRATEGY BACKTEST
===========================================
Multiple strategy variations to find profitable configuration.

OPTIMIZATION APPROACHES:
1. Regime-Based: Only trade in trending markets
2. Volume Confirmed: Require volume spike for entry
3. Trend Following: Trade with the major trend only
4. Mean Reversion: Trade extremes with tight stops
"""
import asyncio
import sys
import os

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from src.brain.backtest_engine import get_backtest_engine
from typing import Optional, Dict


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-calculate all indicators once"""
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # EMAs
    df['ema9'] = close.ewm(span=9).mean()
    df['ema21'] = close.ewm(span=21).mean()
    df['ema50'] = close.ewm(span=50).mean()
    df['ema200'] = close.ewm(span=200).mean()
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Volume MA
    df['volume_ma'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / df['volume_ma']
    
    # Bollinger Bands
    df['bb_mid'] = close.rolling(20).mean()
    df['bb_std'] = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # ADX (Trend Strength)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr_14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_14)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    df['adx'] = dx.rolling(14).mean()
    
    # Regime Detection
    df['trending'] = df['adx'] > 25
    df['bull_regime'] = close > df['ema200']
    df['bear_regime'] = close < df['ema200']
    
    return df


# ============================================================================
# STRATEGY 1: TREND FOLLOWING (Only trade with major trend)
# ============================================================================
async def strategy_trend_following(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """Trade only in the direction of the 200 EMA"""
    if len(df) < 200:
        return None
    
    row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # Must have pre-calculated indicators
    if 'ema200' not in df.columns:
        return None
    
    current_price = row['close']
    atr = row['atr'] if 'atr' in df.columns else current_price * 0.02
    
    # Strong trend filter
    if row.get('adx', 0) < 25:
        return None  # No trade in ranging market
    
    # Volume confirmation
    if row.get('volume_ratio', 0) < 1.2:
        return None  # Need above average volume
    
    # LONG: Price above 200 EMA + MACD bullish cross
    if (row['close'] > row['ema200'] and 
        prev_row['macd'] <= prev_row['macd_signal'] and 
        row['macd'] > row['macd_signal'] and
        row['rsi'] > 40 and row['rsi'] < 70):
        return {
            'action': 'BUY',
            'confidence': 75,
            'sl': current_price - atr * 1.5,
            'tp': current_price + atr * 3
        }
    
    # SHORT: Price below 200 EMA + MACD bearish cross
    if (row['close'] < row['ema200'] and 
        prev_row['macd'] >= prev_row['macd_signal'] and 
        row['macd'] < row['macd_signal'] and
        row['rsi'] < 60 and row['rsi'] > 30):
        return {
            'action': 'SELL',
            'confidence': 75,
            'sl': current_price + atr * 1.5,
            'tp': current_price - atr * 3
        }
    
    return None


# ============================================================================
# STRATEGY 2: BREAKOUT WITH VOLUME
# ============================================================================
async def strategy_breakout_volume(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """Trade breakouts confirmed by volume spike"""
    if len(df) < 50:
        return None
    
    row = df.iloc[-1]
    
    if 'bb_upper' not in df.columns:
        return None
    
    current_price = row['close']
    atr = row['atr'] if 'atr' in df.columns else current_price * 0.02
    
    # Require volume spike (2x average)
    if row.get('volume_ratio', 0) < 2.0:
        return None
    
    # Require trending market
    if row.get('adx', 0) < 20:
        return None
    
    # Bullish breakout: Close above upper BB
    if row['close'] > row['bb_upper'] and row['rsi'] > 50 and row['rsi'] < 80:
        return {
            'action': 'BUY',
            'confidence': 80,
            'sl': row['bb_mid'],  # SL at middle band
            'tp': current_price + atr * 4
        }
    
    # Bearish breakout: Close below lower BB
    if row['close'] < row['bb_lower'] and row['rsi'] < 50 and row['rsi'] > 20:
        return {
            'action': 'SELL',
            'confidence': 80,
            'sl': row['bb_mid'],
            'tp': current_price - atr * 4
        }
    
    return None


# ============================================================================
# STRATEGY 3: MEAN REVERSION (Only in ranging markets)
# ============================================================================
async def strategy_mean_reversion(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """Mean reversion in ranging market conditions"""
    if len(df) < 50:
        return None
    
    row = df.iloc[-1]
    
    if 'bb_lower' not in df.columns:
        return None
    
    current_price = row['close']
    atr = row['atr'] if 'atr' in df.columns else current_price * 0.02
    
    # Only trade in ranging market (low ADX)
    if row.get('adx', 50) > 25:
        return None  # Skip trending markets
    
    # Tight BB width (consolidation)
    if row.get('bb_width', 0.1) > 0.06:  # BB width < 6%
        return None
    
    # Buy oversold at lower BB
    if row['close'] < row['bb_lower'] and row['rsi'] < 30:
        return {
            'action': 'BUY',
            'confidence': 70,
            'sl': current_price - atr * 1,  # Tight stop
            'tp': row['bb_mid']  # Target middle band
        }
    
    # Sell overbought at upper BB
    if row['close'] > row['bb_upper'] and row['rsi'] > 70:
        return {
            'action': 'SELL',
            'confidence': 70,
            'sl': current_price + atr * 1,
            'tp': row['bb_mid']
        }
    
    return None


# ============================================================================
# STRATEGY 4: HYBRID (Best of all)
# ============================================================================
async def strategy_hybrid(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """Combine all strategies based on market conditions"""
    if len(df) < 200:
        return None
    
    row = df.iloc[-1]
    
    if 'adx' not in df.columns:
        return None
    
    adx = row.get('adx', 25)
    
    # Route to appropriate strategy based on regime
    if adx > 30:  # Strong trend
        return await strategy_trend_following(df, idx)
    elif adx < 20:  # Ranging
        return await strategy_mean_reversion(df, idx)
    else:  # Moderate - use breakout
        return await strategy_breakout_volume(df, idx)


# ============================================================================
# WRAPPER TO PRE-CALCULATE INDICATORS
# ============================================================================
_cached_df = None
_cached_symbol = None

async def make_strategy_wrapper(base_strategy):
    """Wrap strategy to pre-calculate indicators once"""
    async def wrapper(df: pd.DataFrame, idx: int) -> Optional[Dict]:
        global _cached_df, _cached_symbol
        
        # Calculate indicators if not cached
        if _cached_df is None or len(_cached_df) != len(df):
            _cached_df = calculate_indicators(df)
        
        return await base_strategy(_cached_df, idx)
    
    return wrapper


async def run_optimization():
    print("=" * 70)
    print("[OPTIMIZATION] DEMIR AI - STRATEGY OPTIMIZATION")
    print("=" * 70)
    
    engine = get_backtest_engine()
    
    # Load data once (5 years)
    print("\n[INFO] Loading 5 years of data...")
    df = await engine.load_historical_data("BTCUSDT", "4h", 1825)
    
    # Pre-calculate indicators
    print("[INFO] Pre-calculating indicators...")
    df_with_indicators = calculate_indicators(df)
    engine.data_cache["BTCUSDT_4h_1825"] = df_with_indicators
    
    strategies = {
        "1. Trend Following": strategy_trend_following,
        "2. Breakout + Volume": strategy_breakout_volume,
        "3. Mean Reversion": strategy_mean_reversion,
        "4. Hybrid Adaptive": strategy_hybrid
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n[TEST] Running: {name}...")
        
        global _cached_df
        _cached_df = df_with_indicators
        
        metrics = await engine.run_backtest(
            symbol="BTCUSDT",
            signal_generator=strategy,
            days=1825,
            initial_equity=10000,
            interval="4h"
        )
        
        # Add Monte Carlo
        if engine.trades and len(engine.trades) >= 10:
            metrics.monte_carlo = engine.run_monte_carlo(engine.trades, 10000, 500)
        
        results[name] = metrics
        
        print(f"   P&L: ${metrics.total_pnl_usd:+,.0f} | Win Rate: {metrics.win_rate:.1f}% | Sharpe: {metrics.sharpe_ratio:.2f}")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("[RESULTS] STRATEGY COMPARISON (5-Year Backtest)")
    print("=" * 70)
    print(f"{'Strategy':<25} {'P&L':>12} {'Win Rate':>10} {'Sharpe':>10} {'PF':>8}")
    print("-" * 70)
    
    for name, m in sorted(results.items(), key=lambda x: x[1].total_pnl_usd, reverse=True):
        pf = f"{m.profit_factor:.2f}" if m.profit_factor < 100 else "inf"
        print(f"{name:<25} ${m.total_pnl_usd:>+10,.0f} {m.win_rate:>9.1f}% {m.sharpe_ratio:>9.2f} {pf:>8}")
    
    # Best strategy analysis
    best_name = max(results.keys(), key=lambda x: results[x].total_pnl_usd)
    best = results[best_name]
    
    print("\n" + "=" * 70)
    print(f"[BEST] {best_name}")
    print("=" * 70)
    print(best.summary())
    
    # Recommendation
    print("\n[RECOMMENDATION]")
    if best.is_profitable():
        if best.is_acceptable():
            print("   Strategy meets institutional standards.")
        else:
            print("   Profitable but needs further optimization.")
    else:
        print("   All strategies unprofitable in 5-year backtest.")
        print("   Consider: Alternative data, ML enhancements, or shorter timeframes.")
    
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_optimization())
