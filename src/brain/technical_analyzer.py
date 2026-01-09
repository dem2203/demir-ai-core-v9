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
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> Dict:
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
        Determine overall trend from indicators
        """
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI signals
        if rsi < 30:
            bullish_signals += 2  # Oversold = strong buy
        elif rsi > 70:
            bearish_signals += 2  # Overbought = strong sell
        elif rsi > 50:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # MACD signals
        if macd > signal:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Bollinger Band position
        if price > bb_middle:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Decision
        if bullish_signals > bearish_signals + 1:
            return "BULLISH"
        elif bearish_signals > bullish_signals + 1:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_strength(self, rsi, macd, signal) -> float:
        """
        Calculate trend strength (0-1)
        """
        # RSI contribution (distance from 50)
        rsi_strength = abs(rsi - 50) / 50
        
        # MACD contribution (histogram strength)
        macd_strength = min(abs(macd - signal) / 100, 1.0)
        
        # Average strength
        return (rsi_strength + macd_strength) / 2
    
    def _generate_analysis(self, symbol, rsi, macd, signal, price, bb_upper, bb_lower, trend, strength):
        """Generate human-readable analysis"""
        
        # RSI interpretation
        if rsi < 30:
            rsi_text = f"RSI {rsi:.1f} (AŞırı Satım - Güçlü Al Sinyali)"
        elif rsi > 70:
            rsi_text = f"RSI {rsi:.1f} (Aşırı Alım - Sat Sinyali)"
        else:
            rsi_text = f"RSI {rsi:.1f} (Normal Seviyeler)"
        
        # MACD interpretation
        if macd > signal:
            macd_text = "MACD Pozitif (Bullish momentum)"
        else:
            macd_text = "MACD Negatif (Bearish momentum)"
        
        # Bollinger Bands
        if price > bb_upper:
            bb_text = f"Fiyat Bollinger üst bandında (${price:.2f} > ${bb_upper:.2f})"
        elif price < bb_lower:
            bb_text = f"Fiyat Bollinger alt bandında (${price:.2f} < ${bb_lower:.2f})"
        else:
            bb_text = f"Fiyat Bollinger bandları içinde (${bb_lower:.2f} - ${bb_upper:.2f})"
        
        analysis = f"""
{symbol} Teknik Analiz:
• {rsi_text}
• {macd_text}
• {bb_text}
• Trend: {trend} (Güç: {strength:.0%})
"""
        return analysis.strip()
