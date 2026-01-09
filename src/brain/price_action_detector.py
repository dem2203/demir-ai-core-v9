import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger("PRICE_ACTION")

class PriceActionDetector:
    """
    Detect bullish/bearish price action BEFORE big moves
    Professional microstructure analysis
    """
    
    def __init__(self):
        self.alerts = []
        
    def analyze_price_action(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Detect early signals of major moves
        
        Key indicators:
        1. Higher lows/Lower highs (trend building)
        2. Volume accumulation (smart money)
        3. Tight consolidation → expansion (coiling spring)
        4. Bullish/Bearish engulfing patterns
        """
        signals = []
        strength = 0
        
        # 1. Trend Building Detection
        trend_signal = self._detect_trend_building(df)
        if trend_signal:
            signals.append(trend_signal)
            strength += 2
        
        # 2. Volume Accumulation
        volume_signal = self._detect_volume_accumulation(df)
        if volume_signal:
            signals.append(volume_signal)
            strength += 3
        
        # 3. Consolidation Breakout Setup
        consolidation = self._detect_consolidation_breakout(df)
        if consolidation:
            signals.append(consolidation)
            strength += 4
        
        # 4. Candlestick Patterns
        pattern = self._detect_reversal_patterns(df)
        if pattern:
            signals.append(pattern)
            strength += 2
        
        
        # Overall signal determination
        # 0-3: Strong directional
        # 4-6: NEUTRAL (no clear direction)
        # 7-10: Very strong directional
        
        bullish_count = sum(1 for s in signals if 'BULLISH' in str(s).upper() or 'Higher Lows' in str(s) or 'Accumulation' in str(s))
        bearish_count = sum(1 for s in signals if 'BEARISH' in str(s).upper() or 'Lower Highs' in str(s) or 'Distribution' in str(s))
        
        if strength >= 8:
            # Very strong signal
            overall = "STRONG_BULLISH" if bullish_count > bearish_count else "STRONG_BEARISH"
        elif strength >= 3:
            # Mid-range = NEUTRAL (don't force a direction on weak signals!)
            overall = "NEUTRAL"
        elif strength >= 1:
            # Weak but present
            overall = "WEAK_BULLISH" if bullish_count > bearish_count else "WEAK_BEARISH"
        else:
            overall = "NEUTRAL"
        
        return {
            'signal': overall,
            'strength': min(strength, 10),
            'indicators': signals if signals else ['No strong signals detected'],
            'current_price': df['close'].iloc[-1]
        }
    
    def _detect_trend_building(self, df: pd.DataFrame) -> str:
        """
        Higher lows = bullish trend building
        Lower highs = bearish trend building
        """
        lows = df['low'].tail(10).values
        highs = df['high'].tail(10).values
        
        # Check for higher lows (bullish)
        higher_lows = 0
        for i in range(1, len(lows)):
            if lows[i] > lows[i-1]:
                higher_lows += 1
        
        if higher_lows >= 6:
            return "📈 Higher Lows - Bullish Trend Building"
        
        # Check for lower highs (bearish)
        lower_highs = 0
        for i in range(1, len(highs)):
            if highs[i] < highs[i-1]:
                lower_highs += 1
        
        if lower_highs >= 6:
            return "📉 Lower Highs - Bearish Trend Building"
        
        return None
    
    def _detect_volume_accumulation(self, df: pd.DataFrame) -> str:
        """
        Smart money accumulation/distribution
        Rising volume + tight price = accumulation
        """
        volume = df['volume'].tail(20)
        close = df['close'].tail(20)
        
        # Calculate volume trend
        volume_ma = volume.rolling(10).mean()
        recent_vol = volume.tail(5).mean()
        
        # Calculate price volatility
        price_range = close.max() - close.min()
        price_std = close.std()
        
        # Accumulation: Rising volume + low volatility
        if recent_vol > volume_ma.iloc[-1] * 1.3 and price_std < close.mean() * 0.02:
            return "🐋 Smart Money Accumulation (Volume ↑ / Price Stable)"
        
        # Distribution: Rising volume + high volatility + declining price
        if recent_vol > volume_ma.iloc[-1] * 1.3 and close.iloc[-1] < close.iloc[-10]:
            return "⚠️ Smart Money Distribution (Volume ↑ / Price ↓)"
        
        return None
    
    def _detect_consolidation_breakout(self, df: pd.DataFrame) -> str:
        """
        Tight range → imminent breakout
        """
        close = df['close'].tail(20)
        high = df['high'].tail(20)
        low = df['low'].tail(20)
        
        # Calculate ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Price range in last 10 candles
        recent_range = high.tail(10).max() - low.tail(10).min()
        
        # Consolidation: range < 1.5 * ATR
        if recent_range < atr * 1.5:
            # Check which way it's leaning
            mid_price = (high.tail(10).max() + low.tail(10).min()) / 2
            current_price = close.iloc[-1]
            
            if current_price > mid_price:
                return "🎯 Tight Consolidation - Bullish Breakout Setup"
            else:
                return "🎯 Tight Consolidation - Bearish Breakdown Setup"
        
        return None
    
    def _detect_reversal_patterns(self, df: pd.DataFrame) -> str:
        """
        Bullish/Bearish engulfing patterns
        """
        if len(df) < 2:
            return None
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        # Bullish Engulfing
        if (prev['close'] < prev['open'] and  # Previous bearish
            curr['close'] > curr['open'] and  # Current bullish
            curr['open'] < prev['close'] and  # Opens below prev close
            curr['close'] > prev['open']):    # Closes above prev open
            return "🟢 Bullish Engulfing Pattern"
        
        # Bearish Engulfing
        if (prev['close'] > prev['open'] and  # Previous bullish
            curr['close'] < curr['open'] and  # Current bearish
            curr['open'] > prev['close'] and  # Opens above prev close
            curr['close'] < prev['open']):    # Closes below prev open
            return "🔴 Bearish Engulfing Pattern"
        
        return None
