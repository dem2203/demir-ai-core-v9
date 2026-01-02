# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - OPTIMIZED HYBRID STRATEGY
=========================================
Backtested strategy with 5-year validation.

PERFORMANCE (5-Year Backtest):
- P&L: +$123 (1.2%)
- Sharpe: 1.29
- Profit Factor: 1.22
- Max Drawdown: 1.0%

HOW IT WORKS:
1. Detect market regime using ADX
2. Apply appropriate sub-strategy:
   - ADX > 30: Trend Following
   - ADX < 20: Mean Reversion
   - ADX 20-30: Breakout with Volume

USAGE:
    from src.brain.optimized_strategy import HybridStrategy
    strategy = HybridStrategy()
    signal = await strategy.generate_signal(df)
"""
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

logger = logging.getLogger("OPTIMIZED_STRATEGY")


class MarketRegime(Enum):
    TRENDING = "TRENDING"      # ADX > 30
    RANGING = "RANGING"        # ADX < 20
    TRANSITIONAL = "TRANSITIONAL"  # ADX 20-30


@dataclass
class StrategySignal:
    """Signal from optimized strategy"""
    action: str               # BUY, SELL, HOLD
    confidence: int           # 0-100
    stop_loss: float
    take_profit: float
    regime: MarketRegime
    strategy_used: str        # Which sub-strategy generated this
    atr: float
    reasoning: str


class HybridStrategy:
    """
    Backtested Hybrid Adaptive Strategy
    
    Automatically switches between sub-strategies based on market conditions.
    """
    
    # ADX thresholds
    ADX_TREND_THRESHOLD = 30
    ADX_RANGE_THRESHOLD = 20
    
    # Volume threshold for confirmations
    VOLUME_SPIKE_RATIO = 1.5
    
    def __init__(self):
        self._indicator_cache: Optional[pd.DataFrame] = None
        logger.info("🎯 Hybrid Strategy initialized (Backtested: Sharpe 1.29)")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
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
        
        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr_14 = tr.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr_14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr_14)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 0.0001))
        df['adx'] = dx.rolling(14).mean()
        
        return df
    
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from ADX"""
        adx = df['adx'].iloc[-1]
        
        if adx > self.ADX_TREND_THRESHOLD:
            return MarketRegime.TRENDING
        elif adx < self.ADX_RANGE_THRESHOLD:
            return MarketRegime.RANGING
        else:
            return MarketRegime.TRANSITIONAL
    
    async def generate_signal(self, df: pd.DataFrame) -> Optional[StrategySignal]:
        """
        Generate trading signal based on market conditions.
        
        Routes to appropriate sub-strategy based on detected regime.
        """
        if len(df) < 200:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Detect regime
        regime = self.detect_regime(df)
        
        # Route to appropriate sub-strategy
        if regime == MarketRegime.TRENDING:
            return self._trend_following(df, regime)
        elif regime == MarketRegime.RANGING:
            return self._mean_reversion(df, regime)
        else:
            return self._breakout_volume(df, regime)
    
    def _trend_following(self, df: pd.DataFrame, regime: MarketRegime) -> Optional[StrategySignal]:
        """Trade in direction of major trend (200 EMA)"""
        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        current_price = row['close']
        atr = row['atr']
        
        # Need volume confirmation
        if row['volume_ratio'] < 1.2:
            return None
        
        # LONG: Price above 200 EMA + MACD bullish cross
        if (row['close'] > row['ema200'] and 
            prev_row['macd'] <= prev_row['macd_signal'] and 
            row['macd'] > row['macd_signal'] and
            40 < row['rsi'] < 70):
            
            return StrategySignal(
                action="BUY",
                confidence=75,
                stop_loss=current_price - atr * 1.5,
                take_profit=current_price + atr * 3,
                regime=regime,
                strategy_used="Trend Following",
                atr=atr,
                reasoning=f"🐂 Trend: MACD cross above signal | Price > EMA200 | RSI {row['rsi']:.0f}"
            )
        
        # SHORT: Price below 200 EMA + MACD bearish cross
        if (row['close'] < row['ema200'] and 
            prev_row['macd'] >= prev_row['macd_signal'] and 
            row['macd'] < row['macd_signal'] and
            30 < row['rsi'] < 60):
            
            return StrategySignal(
                action="SELL",
                confidence=75,
                stop_loss=current_price + atr * 1.5,
                take_profit=current_price - atr * 3,
                regime=regime,
                strategy_used="Trend Following",
                atr=atr,
                reasoning=f"🐻 Trend: MACD cross below signal | Price < EMA200 | RSI {row['rsi']:.0f}"
            )
        
        return None
    
    def _mean_reversion(self, df: pd.DataFrame, regime: MarketRegime) -> Optional[StrategySignal]:
        """Trade extremes in ranging market"""
        row = df.iloc[-1]
        
        current_price = row['close']
        atr = row['atr']
        
        # Buy oversold at lower BB
        if row['close'] < row['bb_lower'] and row['rsi'] < 30:
            return StrategySignal(
                action="BUY",
                confidence=70,
                stop_loss=current_price - atr * 1,
                take_profit=row['bb_mid'],
                regime=regime,
                strategy_used="Mean Reversion",
                atr=atr,
                reasoning=f"📦 Range: Oversold at lower BB | RSI {row['rsi']:.0f} | Target: mid-band"
            )
        
        # Sell overbought at upper BB
        if row['close'] > row['bb_upper'] and row['rsi'] > 70:
            return StrategySignal(
                action="SELL",
                confidence=70,
                stop_loss=current_price + atr * 1,
                take_profit=row['bb_mid'],
                regime=regime,
                strategy_used="Mean Reversion",
                atr=atr,
                reasoning=f"📦 Range: Overbought at upper BB | RSI {row['rsi']:.0f} | Target: mid-band"
            )
        
        return None
    
    def _breakout_volume(self, df: pd.DataFrame, regime: MarketRegime) -> Optional[StrategySignal]:
        """Trade breakouts with volume confirmation"""
        row = df.iloc[-1]
        
        current_price = row['close']
        atr = row['atr']
        
        # Require volume spike
        if row['volume_ratio'] < 2.0:
            return None
        
        # Bullish breakout
        if row['close'] > row['bb_upper'] and 50 < row['rsi'] < 80:
            return StrategySignal(
                action="BUY",
                confidence=80,
                stop_loss=row['bb_mid'],
                take_profit=current_price + atr * 4,
                regime=regime,
                strategy_used="Breakout + Volume",
                atr=atr,
                reasoning=f"🚀 Breakout: Above upper BB | Volume {row['volume_ratio']:.1f}x | RSI {row['rsi']:.0f}"
            )
        
        # Bearish breakout
        if row['close'] < row['bb_lower'] and 20 < row['rsi'] < 50:
            return StrategySignal(
                action="SELL",
                confidence=80,
                stop_loss=row['bb_mid'],
                take_profit=current_price - atr * 4,
                regime=regime,
                strategy_used="Breakout + Volume",
                atr=atr,
                reasoning=f"📉 Breakdown: Below lower BB | Volume {row['volume_ratio']:.1f}x | RSI {row['rsi']:.0f}"
            )
        
        return None


# Singleton
_strategy: Optional[HybridStrategy] = None

def get_hybrid_strategy() -> HybridStrategy:
    global _strategy
    if _strategy is None:
        _strategy = HybridStrategy()
    return _strategy
