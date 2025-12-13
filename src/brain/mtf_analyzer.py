"""
Multi-Timeframe (MTF) Confluence Analyzer

Analyzes trend alignment across multiple timeframes:
1. 1H (Short-term momentum)
2. 4H (Medium-term trend)  
3. 1D (Long-term direction)

Confluence = All timeframes agree = High probability setup

ALL DATA FROM BINANCE OHLCV - NO MOCKS!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import ccxt

logger = logging.getLogger("MTF_ANALYZER")


@dataclass
class TimeframeTrend:
    """Trend analysis for a single timeframe"""
    timeframe: str
    trend: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: int  # 0-100
    ema_fast: float
    ema_slow: float
    rsi: float
    structure: str  # 'HIGHER_HIGHS', 'LOWER_LOWS', 'RANGING'


class MTFAnalyzer:
    """
    Multi-Timeframe Confluence Analyzer
    
    Fetches 1H, 4H, 1D data from Binance and determines trend alignment.
    Zero external dependencies - all calculated from price action.
    """
    
    TIMEFRAMES = ['1h', '4h', '1d']
    
    def __init__(self):
        self.exchange = ccxt.binance()
        self.trends: Dict[str, TimeframeTrend] = {}
    
    def analyze(self, symbol: str) -> Dict:
        """
        Full MTF analysis for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Dict with trend analysis per timeframe and confluence score
        """
        try:
            # Fetch data for all timeframes
            data = {}
            for tf in self.TIMEFRAMES:
                ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=100)
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    data[tf] = df
            
            if not data:
                logger.error(f"Failed to fetch data for {symbol}")
                return self._empty_result()
            
            # Analyze each timeframe
            for tf, df in data.items():
                self.trends[tf] = self._analyze_timeframe(df, tf)
            
            # Calculate confluence
            confluence = self._calculate_confluence()
            
            # Determine entry quality
            entry_quality = self._assess_entry_quality()
            
            # Generate MTF signal
            mtf_signal = self._generate_mtf_signal()
            
            return {
                'trends': {tf: self._trend_to_dict(trend) for tf, trend in self.trends.items()},
                'confluence': confluence,
                'confluence_score': confluence['score'],
                'confluence_type': confluence['type'],
                'entry_quality': entry_quality,
                'mtf_signal': mtf_signal,
                'recommended_action': mtf_signal['action'],
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"MTF analysis error: {e}")
            return self._empty_result()
    
    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> TimeframeTrend:
        """Analyze trend for a single timeframe"""
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema_slow = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        
        # Calculate RSI
        rsi = self._calculate_rsi(df['close'], 14)
        
        # Determine trend
        current_price = df['close'].iloc[-1]
        
        # EMA-based trend
        if current_price > ema_fast > ema_slow:
            trend = 'BULLISH'
        elif current_price < ema_fast < ema_slow:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
        
        # Calculate trend strength
        ema_spread = abs(ema_fast - ema_slow) / ema_slow * 100
        strength = min(100, int(ema_spread * 20))
        
        # Adjust strength with RSI
        if trend == 'BULLISH' and rsi > 50:
            strength = min(100, strength + 10)
        elif trend == 'BEARISH' and rsi < 50:
            strength = min(100, strength + 10)
        
        # Market structure
        structure = self._analyze_structure(df)
        
        return TimeframeTrend(
            timeframe=timeframe,
            trend=trend,
            strength=strength,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi=rsi,
            structure=structure
        )
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _analyze_structure(self, df: pd.DataFrame) -> str:
        """Analyze market structure (HH/HL vs LH/LL)"""
        highs = df['high'].rolling(window=5).max()
        lows = df['low'].rolling(window=5).min()
        
        # Get recent swing points
        recent_highs = highs.tail(20).values
        recent_lows = lows.tail(20).values
        
        # Check for higher highs
        hh_count = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i] > recent_highs[i-1]:
                hh_count += 1
        
        # Check for lower lows
        ll_count = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i] < recent_lows[i-1]:
                ll_count += 1
        
        if hh_count > ll_count + 5:
            return 'HIGHER_HIGHS'
        elif ll_count > hh_count + 5:
            return 'LOWER_LOWS'
        else:
            return 'RANGING'
    
    def _calculate_confluence(self) -> Dict:
        """Calculate confluence across timeframes"""
        if not self.trends:
            return {'type': 'NO_DATA', 'score': 0, 'direction': 'NEUTRAL'}
        
        bullish_count = sum(1 for t in self.trends.values() if t.trend == 'BULLISH')
        bearish_count = sum(1 for t in self.trends.values() if t.trend == 'BEARISH')
        total = len(self.trends)
        
        # Perfect confluence
        if bullish_count == total:
            return {
                'type': 'STRONG_CONFLUENCE',
                'score': 100,
                'direction': 'BULLISH'
            }
        elif bearish_count == total:
            return {
                'type': 'STRONG_CONFLUENCE',
                'score': 100,
                'direction': 'BEARISH'
            }
        
        # Moderate confluence (2/3 agree)
        elif bullish_count >= 2:
            return {
                'type': 'MODERATE_CONFLUENCE',
                'score': 67,
                'direction': 'BULLISH'
            }
        elif bearish_count >= 2:
            return {
                'type': 'MODERATE_CONFLUENCE',
                'score': 67,
                'direction': 'BEARISH'
            }
        
        # Conflict
        else:
            return {
                'type': 'CONFLICT',
                'score': 33,
                'direction': 'NEUTRAL'
            }
    
    def _assess_entry_quality(self) -> Dict:
        """Assess entry quality based on MTF alignment"""
        confluence = self._calculate_confluence()
        
        # Check if lower timeframe is pulling back to higher timeframe trend
        tf_1h = self.trends.get('1h')
        tf_4h = self.trends.get('4h')
        tf_1d = self.trends.get('1d')
        
        quality = {
            'rating': 'POOR',
            'score': 0,
            'reason': ''
        }
        
        if not tf_1h or not tf_4h or not tf_1d:
            return quality
        
        # Best entry: HTF bullish, LTF pullback (neutral/slight bearish)
        if tf_4h.trend == 'BULLISH' and tf_1d.trend == 'BULLISH':
            if tf_1h.trend == 'NEUTRAL' or tf_1h.rsi < 40:
                quality = {
                    'rating': 'OPTIMAL',
                    'score': 95,
                    'reason': 'HTF bullish + LTF pullback opportunity'
                }
            elif tf_1h.trend == 'BULLISH':
                quality = {
                    'rating': 'GOOD',
                    'score': 75,
                    'reason': 'Full alignment, may be extended'
                }
        
        elif tf_4h.trend == 'BEARISH' and tf_1d.trend == 'BEARISH':
            if tf_1h.trend == 'NEUTRAL' or tf_1h.rsi > 60:
                quality = {
                    'rating': 'OPTIMAL',
                    'score': 95,
                    'reason': 'HTF bearish + LTF pullback opportunity'
                }
            elif tf_1h.trend == 'BEARISH':
                quality = {
                    'rating': 'GOOD',
                    'score': 75,
                    'reason': 'Full alignment, may be oversold'
                }
        
        elif confluence['type'] == 'MODERATE_CONFLUENCE':
            quality = {
                'rating': 'FAIR',
                'score': 50,
                'reason': 'Partial alignment'
            }
        
        return quality
    
    def _generate_mtf_signal(self) -> Dict:
        """Generate trading signal based on MTF analysis"""
        confluence = self._calculate_confluence()
        entry = self._assess_entry_quality()
        
        signal = {
            'action': 'WAIT',
            'confidence': 0,
            'timeframe_alignment': {},
            'reason': ''
        }
        
        # Map trends
        for tf, trend in self.trends.items():
            signal['timeframe_alignment'][tf] = trend.trend
        
        # Generate action
        if confluence['score'] >= 67:
            if confluence['direction'] == 'BULLISH':
                signal['action'] = 'BUY'
                signal['confidence'] = entry['score']
                signal['reason'] = f"MTF Bullish ({confluence['type']}) - {entry['reason']}"
            elif confluence['direction'] == 'BEARISH':
                signal['action'] = 'SELL'
                signal['confidence'] = entry['score']
                signal['reason'] = f"MTF Bearish ({confluence['type']}) - {entry['reason']}"
        else:
            signal['action'] = 'WAIT'
            signal['confidence'] = 25
            signal['reason'] = f"MTF Conflict - No clear direction"
        
        return signal
    
    def _trend_to_dict(self, trend: TimeframeTrend) -> Dict:
        """Convert TimeframeTrend to dictionary"""
        return {
            'timeframe': trend.timeframe,
            'trend': trend.trend,
            'strength': trend.strength,
            'ema_fast': trend.ema_fast,
            'ema_slow': trend.ema_slow,
            'rsi': trend.rsi,
            'structure': trend.structure
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result when analysis fails"""
        return {
            'trends': {},
            'confluence': {'type': 'NO_DATA', 'score': 0, 'direction': 'NEUTRAL'},
            'confluence_score': 0,
            'confluence_type': 'NO_DATA',
            'entry_quality': {'rating': 'UNKNOWN', 'score': 0, 'reason': 'No data'},
            'mtf_signal': {'action': 'WAIT', 'confidence': 0, 'reason': 'No data'},
            'recommended_action': 'WAIT',
            'symbol': ''
        }


# Quick test
if __name__ == "__main__":
    print("Testing MTF Analyzer...")
    
    mtf = MTFAnalyzer()
    result = mtf.analyze('BTC/USDT')
    
    print(f"\n[MTF] Analysis for BTC/USDT:")
    print(f"1H Trend: {result['trends'].get('1h', {}).get('trend', 'N/A')}")
    print(f"4H Trend: {result['trends'].get('4h', {}).get('trend', 'N/A')}")
    print(f"1D Trend: {result['trends'].get('1d', {}).get('trend', 'N/A')}")
    print(f"Confluence: {result['confluence_type']} ({result['confluence_score']}%)")
    print(f"Entry Quality: {result['entry_quality']}")
    print(f"Signal: {result['mtf_signal']}")
