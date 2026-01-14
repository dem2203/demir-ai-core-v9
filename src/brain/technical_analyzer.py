import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger("TECHNICAL_ANALYZER")

class TechnicalAnalyzer:
    """
    Professional Technical Analysis with 9 indicators
    - RSI, MACD, Bollinger Bands
    - ATR, Stochastic, Volume Profile
    - OBV, ADX, Support/Resistance
    """
    def __init__(self):
        pass
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Full technical analysis using 9 professional indicators
        
        Args:
            df: OHLCV DataFrame
            symbol: Trading pair
            
        Returns:
            Analysis with trend, strength, signals, all indicators
        """
        try:
            if df.empty or len(df) < 50:
                return {"trend": "UNKNOWN", "analysis": "Insufficient data"}
            
            # Calculate ALL indicators
            rsi = self._calculate_rsi(df['close'])
            macd, signal, hist = self._calculate_macd(df['close'])
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
            atr = self._calculate_atr(df)
            stoch_k, stoch_d = self._calculate_stochastic(df)
            obv = self._calculate_obv(df)
            adx = self._calculate_adx(df)
            
            # Current values
            current_rsi = rsi.iloc[-1]
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            current_price = df['close'].iloc[-1]
            current_atr = atr.iloc[-1]
            current_stoch_k = stoch_k.iloc[-1]
            current_stoch_d = stoch_d.iloc[-1]
            current_obv = obv.iloc[-1]
            current_adx = adx.iloc[-1]
            
            # PROFESSIONAL trend with 9 indicators
            trend = self._determine_trend_professional(
                current_rsi, current_macd, current_signal, 
                current_price, bb_middle.iloc[-1],
                current_stoch_k, current_adx, current_obv, obv.iloc[-5]
            )
            
            # Strength with all indicators
            strength = self._calculate_strength_professional(
                current_rsi, current_macd, current_signal,
                current_stoch_k, current_adx
            )
            
            # Generate analysis
            analysis = f"""{symbol} Professional TA:
• RSI {current_rsi:.1f} ({self._rsi_interpretation(current_rsi)})
• MACD {'Positive' if current_macd > current_signal else 'Negative'} ({'Bullish' if current_macd > current_signal else 'Bearish'} momentum)
• Stochastic {current_stoch_k:.1f} ({self._stoch_interpretation(current_stoch_k)})
• ADX {current_adx:.1f} (Trend strength: {self._adx_interpretation(current_adx)})
• ATR ${current_atr:.2f} (Volatility)
• OBV {'Rising' if current_obv > obv.iloc[-5] else 'Falling'} (Volume pressure)
• Price within Bollinger Bands (${bb_lower.iloc[-1]:.2f} - ${bb_upper.iloc[-1]:.2f})
• Trend: {trend} (Strength: {strength:.0%})"""
            
            logger.info(f"📊 TA: {symbol} → {trend} (Strength: {strength:.0%})")
            
            return {
                "trend": trend,
                "strength": strength,
                "analysis": analysis,
                "indicators": {
                    "rsi": current_rsi,
                    "macd": current_macd,
                    "macd_signal": current_signal,
                    "bb_upper": bb_upper.iloc[-1],
                    "bb_lower": bb_lower.iloc[-1],
                    "atr": current_atr,
                    "stoch_k": current_stoch_k,
                    "stoch_d": current_stoch_d,
                    "obv": current_obv,
                    "adx": current_adx
                }
            }
            
        except Exception as e:
            logger.error(f"TA error: {e}")
            return {"trend": "ERROR", "analysis": str(e)}
    
    # ========== CORE INDICATORS (Existing) ==========
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, series: pd.Series, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, series: pd.Series, period=20, std_dev=2):
        """Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    # ========== NEW PROFESSIONAL INDICATORS ==========
    
    def _calculate_atr(self, df: pd.DataFrame, period=14) -> pd.Series:
        """Average True Range - Volatility"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, period=14, smooth=3):
        """Stochastic Oscillator - Momentum"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=smooth).mean()
        return stoch_k, stoch_d
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume - Volume Pressure"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def _calculate_adx(self, df: pd.DataFrame, period=14) -> pd.Series:
        """Average Directional Index - Trend Strength"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self._calculate_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx
    
    # ========== PROFESSIONAL DECISION LOGIC ==========
    
    def _determine_trend_professional(self, rsi, macd, signal, price, bb_middle, 
                                       stoch_k, adx, obv, obv_prev) -> str:
        """Professional trend with 9 indicators"""
        bullish_score = 0
        bearish_score = 0
        
        # RSI (weight: 2 - REFERENCE ONLY, NOT BLOCKER)
        if rsi < 30: bullish_score += 2  # Oversold = Opportunity
        elif rsi > 70: bearish_score += 1  # Mild caution, but DON'T block trend
        elif rsi > 50: bullish_score += 1
        elif rsi < 50: bearish_score += 1
        
        # MACD (weight: 3)
        if macd > signal: 
            bullish_score += 3
        else: 
            bearish_score += 3
        
        # Bollinger (weight: 1)
        if price > bb_middle: bullish_score += 1
        else: bearish_score += 1
        
        # Stochastic (weight: 1 - REDUCED from 2)
        if stoch_k < 20: bullish_score += 1
        elif stoch_k > 80: bearish_score += 1  # REMOVED extreme penalty
        elif stoch_k > 50: bullish_score += 1
        else: bearish_score += 1
        
        # ADX Trend Strength (weight: 2 - INCREASED for trend following)
        if adx > 25:
            # Strong trend - FOLLOW IT
            if bullish_score > bearish_score: bullish_score += 2
            else: bearish_score += 2
        
        # OBV Volume Pressure (weight: 3 - INCREASED, volume is king)
        if obv > obv_prev: bullish_score += 3
        else: bearish_score += 3
        
        # FINAL DECISION - Simple majority
        if bullish_score > bearish_score: return "BULLISH"
        elif bearish_score > bullish_score: return "BEARISH"
        else: return "NEUTRAL"
    
    def _calculate_strength_professional(self, rsi, macd, signal, stoch_k, adx):
        """Strength based on all indicators"""
        # RSI strength
        rsi_distance = abs(rsi - 50)
        if rsi_distance > 30: rsi_strength = 1.0
        elif rsi_distance > 20: rsi_strength = 0.8
        elif rsi_distance > 10: rsi_strength = 0.5
        else: rsi_strength = 0.3
        
        # MACD strength
        macd_diff = abs(macd - signal)
        macd_strength = min(macd_diff / 50, 1.0)
        
        # Stochastic strength
        stoch_distance = abs(stoch_k - 50)
        if stoch_distance > 30: stoch_strength = 1.0
        elif stoch_distance > 20: stoch_strength = 0.7
        else: stoch_strength = 0.4
        
        # ADX trend strength
        if adx > 50: adx_strength = 1.0
        elif adx > 25: adx_strength = 0.7
        elif adx > 15: adx_strength = 0.4
        else: adx_strength = 0.2
        
        # Weighted average (MACD and ADX more important)
        total_strength = (
            rsi_strength * 0.2 +
            macd_strength * 0.3 +
            stoch_strength * 0.2 +
            adx_strength * 0.3
        )
        
        # Minimum floor 30% (less aggressive than before)
        return max(total_strength, 0.3)
    
    # ========== HELPER INTERPRETATIONS ==========
    
    def _rsi_interpretation(self, rsi):
        if rsi < 30: return "Oversold"
        elif rsi > 70: return "Overbought"
        else: return "Normal"
    
    def _stoch_interpretation(self, stoch):
        if stoch < 20: return "Oversold"
        elif stoch > 80: return "Overbought"
        else: return "Normal"
    
    def _adx_interpretation(self, adx):
        if adx > 50: return "Very Strong"
        elif adx > 25: return "Strong"
        elif adx > 15: return "Moderate"
        else: return "Weak/Ranging"
