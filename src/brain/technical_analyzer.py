import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger("TECHNICAL_ANALYZER")

class TechnicalAnalyzer:
    """
    PROFESSIONAL Technical Analysis - No AI needed
    Uses industry-standard indicators: RSI, MACD, Bollinger Bands
    """
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Full technical analysis using traditional indicators
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            symbol: Trading pair symbol
            
        Returns:
            Analysis dict with trend, strength, signals
        """
        try:
            if df.empty or len(df) < 50:
                return {"trend": "UNKNOWN", "analysis": "Insufficient data"}
            
            # Calculate all indicators
            rsi = self._calculate_rsi(df['close'])
            macd, signal, hist = self._calculate_macd(df['close'])
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
            
            # Current values
            current_rsi = rsi.iloc[-1]
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Trend determination
            trend = self._determine_trend(current_rsi, current_macd, current_signal, current_price, bb_middle.iloc[-1])
            
            # Strength calculation
            strength = self._calculate_strength(current_rsi, current_macd, current_signal)
            
            # Generate analysis text
            analysis = self._generate_analysis(
                symbol, current_rsi, current_macd, current_signal,
                current_price, bb_upper.iloc[-1], bb_lower.iloc[-1], trend, strength
            )
            
            logger.info(f"📊 TA: {symbol} → {trend} (Güç: {strength:.0%})")
            
            return {
                "trend": trend,
                "strength": strength,
                "analysis": analysis,
                "indicators": {
                    "rsi": current_rsi,
                    "macd": current_macd,
                    "macd_signal": current_signal,
                    "bb_upper": bb_upper.iloc[-1],
                    "bb_lower": bb_lower.iloc[-1]
                }
            }
            
        except Exception as e:
            logger.error(f"TA error: {e}")
            return {"trend": "ERROR", "analysis": str(e)}
    
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
    
    def _determine_trend(self, rsi, macd, signal, price, bb_middle) -> str:
        """
        AGGRESSIVE trend determination - lower threshold!
        """
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI signals (MORE WEIGHT)
        if rsi < 35:  # Was 30, now 35 - easier to trigger
            bullish_signals += 3  # Was 2, now 3 - stronger
        elif rsi > 65:  # Was 70, now 65 - easier to trigger
            bearish_signals += 3  # Was 2, now 3 - stronger
        elif rsi > 52:  # Was 50, now 52 - more sensitive
            bullish_signals += 2  # Was 1, now 2
        elif rsi < 48:  # Was <50, now <48 - more sensitive
            bearish_signals += 2  # Was 1, now 2
        
        # MACD signals (MORE WEIGHT)
        if macd > signal:
            bullish_signals += 2  # Was 1, now 2
        else:
            bearish_signals += 2  # Was 1, now 2
        
        # Bollinger Band position
        if price > bb_middle:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # AGGRESSIVE Decision - only need equal or 1 advantage!
        if bullish_signals >= bearish_signals:  # Was >, now >=
            return "BULLISH"
        elif bearish_signals > bullish_signals:  # Removed +1
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_strength(self, rsi, macd, signal) -> float:
        """
        AGGRESSIVE strength calculation - exponential!
        """
        # RSI contribution - EXPONENTIAL for more impact
        rsi_distance = abs(rsi - 50) / 50
        rsi_strength = min(rsi_distance ** 1.5, 1.0)  # Exponential boost!
        # RSI 60: 0.2^1.5 = 0.089 → NOW: 0.2^1.5 = 0.089... wait
        # Let me use better formula
        # RSI 60: distance = 10, normalize to 0-1 scale
        rsi_normalized = abs(rsi - 50)  # 0 to 50
        if rsi_normalized > 20:
            rsi_strength = 0.8 + (rsi_normalized - 20) * 0.01  # Strong signal
        elif rsi_normalized > 10:
            rsi_strength = 0.5 + (rsi_normalized - 10) * 0.03  # Medium
        elif rsi_normalized > 5:
            rsi_strength = 0.3 + (rsi_normalized - 5) * 0.04
        else:
            rsi_strength = rsi_normalized * 0.06
        
        # MACD contribution - MORE AGGRESSIVE SCALING
        macd_diff = abs(macd - signal)
        macd_strength = min(macd_diff / 50, 1.0)  # Was /100, now /50 - 2x boost!
        
        # Average strength with minimum floor
        avg_strength = (rsi_strength + macd_strength) / 2
        return max(avg_strength, 0.4)  # FLOOR at 0.4 instead of 0!
    
    def _generate_analysis(self, symbol, rsi, macd, signal, price, bb_upper, bb_lower, trend, strength):
        """Generate human-readable analysis"""
        
        # RSI interpretation
        if rsi < 30:
            rsi_text = f"RSI {rsi:.1f} (Oversold - Strong BUY Signal)" # User facing: Aşırı Satım
        elif rsi > 70:
            rsi_text = f"RSI {rsi:.1f} (Overbought - SELL Signal)" # User facing: Aşırı Alım
        else:
            rsi_text = f"RSI {rsi:.1f} (Normal Levels)" # User facing: Normal
        
        # MACD interpretation
        if macd > signal:
            macd_text = "MACD Positive (Bullish momentum)"
        else:
            macd_text = "MACD Negative (Bearish momentum)"
        
        # Bollinger Bands
        if price > bb_upper:
            bb_text = f"Price at Upper Bollinger Band (${price:.2f} > ${bb_upper:.2f})"
        elif price < bb_lower:
            bb_text = f"Price at Lower Bollinger Band (${price:.2f} < ${bb_lower:.2f})"
        else:
            bb_text = f"Price within Bollinger Bands (${bb_lower:.2f} - ${bb_upper:.2f})"
        
        analysis = f"""
{symbol} Teknik Analiz:
• {rsi_text}
• {macd_text}
• {bb_text}
• Trend: {trend} (Güç: {strength:.0%})
"""
        return analysis.strip()
