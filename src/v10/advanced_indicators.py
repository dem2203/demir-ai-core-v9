# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - ADVANCED TECHNICAL INDICATORS
=============================================
7 Yeni gösterge + Multi-timeframe analizi:
1. Bollinger Bands (squeeze detection)
2. VWAP (Volume Weighted Average Price)
3. ATR (Average True Range - volatility)
4. Stochastic RSI
5. ADX (Trend Strength)
6. Volume Profile (key levels)
7. Ichimoku Cloud (multi-signal)

KULLANIM:
    analyzer = AdvancedIndicators()
    signals = await analyzer.analyze(symbol, klines)
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger("ADVANCED_INDICATORS")


@dataclass
class IndicatorSignal:
    """Tek gösterge sinyali"""
    name: str
    value: float
    signal: str  # "BUY", "SELL", "NEUTRAL"
    strength: float  # 0-1
    description: str


class AdvancedIndicators:
    """
    7 gelişmiş teknik gösterge hesaplama ve analiz.
    Tüm hesaplamalar kline verilerinden yapılır.
    """
    
    def __init__(self):
        logger.info("📊 Advanced Indicators initialized")
    
    def analyze_all(self, klines: List, current_price: float) -> Dict:
        """
        Tüm göstergeleri hesapla ve analiz et.
        
        Args:
            klines: [[open_time, open, high, low, close, volume, ...], ...]
            current_price: Mevcut fiyat
        
        Returns:
            Dict with all indicator signals and combined score
        """
        if len(klines) < 50:
            return {'error': 'Insufficient data', 'signals': []}
        
        # Parse klines
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        opens = [float(k[1]) for k in klines]
        
        signals = []
        
        # 1. Bollinger Bands
        bb_signal = self._bollinger_bands(closes, current_price)
        signals.append(bb_signal)
        
        # 2. VWAP
        vwap_signal = self._vwap(closes, highs, lows, volumes, current_price)
        signals.append(vwap_signal)
        
        # 3. ATR (volatility)
        atr_signal = self._atr(closes, highs, lows, current_price)
        signals.append(atr_signal)
        
        # 4. Stochastic RSI
        stoch_rsi_signal = self._stochastic_rsi(closes)
        signals.append(stoch_rsi_signal)
        
        # 5. ADX (Trend Strength)
        adx_signal = self._adx(closes, highs, lows)
        signals.append(adx_signal)
        
        # 6. Volume Profile
        vol_profile_signal = self._volume_profile(closes, volumes, current_price)
        signals.append(vol_profile_signal)
        
        # 7. Ichimoku Cloud
        ichimoku_signal = self._ichimoku(closes, highs, lows, current_price)
        signals.append(ichimoku_signal)
        
        # Calculate combined score
        bullish_score = sum(s.strength for s in signals if s.signal == "BUY")
        bearish_score = sum(s.strength for s in signals if s.signal == "SELL")
        
        if bullish_score > bearish_score:
            combined_direction = "BULLISH"
        elif bearish_score > bullish_score:
            combined_direction = "BEARISH"
        else:
            combined_direction = "NEUTRAL"
        
        return {
            'signals': signals,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'combined_direction': combined_direction,
            'confidence': max(bullish_score, bearish_score) / 7 * 100  # Max 7 indicators
        }
    
    # =========================================
    # 1. BOLLINGER BANDS
    # =========================================
    
    def _bollinger_bands(self, closes: List[float], current_price: float, 
                         period: int = 20, std_dev: float = 2.0) -> IndicatorSignal:
        """
        Bollinger Bands - Squeeze detection ve breakout sinyalleri
        """
        if len(closes) < period:
            return IndicatorSignal("Bollinger Bands", 0, "NEUTRAL", 0, "Yetersiz veri")
        
        # SMA
        sma = sum(closes[-period:]) / period
        
        # Standard Deviation
        variance = sum((x - sma) ** 2 for x in closes[-period:]) / period
        std = math.sqrt(variance)
        
        # Bands
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        # Bandwidth (squeeze detection)
        bandwidth = (upper_band - lower_band) / sma * 100
        
        # Signal
        if current_price < lower_band:
            signal = "BUY"
            strength = min(1.0, (lower_band - current_price) / lower_band * 20)
            desc = f"Fiyat alt bandın altında (${lower_band:.0f})"
        elif current_price > upper_band:
            signal = "SELL"
            strength = min(1.0, (current_price - upper_band) / upper_band * 20)
            desc = f"Fiyat üst bandın üstünde (${upper_band:.0f})"
        elif bandwidth < 2:  # Squeeze
            signal = "NEUTRAL"
            strength = 0.5
            desc = f"Squeeze tespit! Bandwidth: {bandwidth:.1f}%"
        else:
            signal = "NEUTRAL"
            strength = 0
            desc = f"Normal aralık (BW: {bandwidth:.1f}%)"
        
        return IndicatorSignal("Bollinger Bands", bandwidth, signal, strength, desc)
    
    # =========================================
    # 2. VWAP
    # =========================================
    
    def _vwap(self, closes: List[float], highs: List[float], lows: List[float],
              volumes: List[float], current_price: float) -> IndicatorSignal:
        """
        VWAP - Volume Weighted Average Price
        Fiyat VWAP üstünde/altında?
        """
        if len(closes) < 24:  # 24 bar minimum
            return IndicatorSignal("VWAP", 0, "NEUTRAL", 0, "Yetersiz veri")
        
        # Typical price = (H + L + C) / 3
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs[-24:], lows[-24:], closes[-24:])]
        vols = volumes[-24:]
        
        # VWAP = sum(TP * Vol) / sum(Vol)
        tp_vol = sum(tp * v for tp, v in zip(typical_prices, vols))
        total_vol = sum(vols)
        vwap = tp_vol / total_vol if total_vol > 0 else closes[-1]
        
        # Distance from VWAP
        distance_pct = (current_price - vwap) / vwap * 100
        
        if distance_pct > 1.5:
            signal = "SELL"
            strength = min(1.0, distance_pct / 5)
            desc = f"Fiyat VWAP üstünde (+{distance_pct:.1f}%)"
        elif distance_pct < -1.5:
            signal = "BUY"
            strength = min(1.0, abs(distance_pct) / 5)
            desc = f"Fiyat VWAP altında ({distance_pct:.1f}%)"
        else:
            signal = "NEUTRAL"
            strength = 0
            desc = f"Fiyat VWAP seviyesinde (${vwap:.0f})"
        
        return IndicatorSignal("VWAP", vwap, signal, strength, desc)
    
    # =========================================
    # 3. ATR (Volatility)
    # =========================================
    
    def _atr(self, closes: List[float], highs: List[float], lows: List[float],
             current_price: float, period: int = 14) -> IndicatorSignal:
        """
        ATR - Average True Range (volatility measure)
        """
        if len(closes) < period + 1:
            return IndicatorSignal("ATR", 0, "NEUTRAL", 0, "Yetersiz veri")
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        atr = sum(true_ranges[-period:]) / period
        atr_pct = atr / current_price * 100
        
        # High volatility = risk / opportunity
        if atr_pct > 3:
            signal = "NEUTRAL"  # High vol = risky
            strength = 0.5
            desc = f"Yüksek volatilite! ATR: {atr_pct:.2f}%"
        elif atr_pct < 1:
            signal = "NEUTRAL"  # Low vol = breakout coming
            strength = 0.3
            desc = f"Düşük volatilite (sıkışma). ATR: {atr_pct:.2f}%"
        else:
            signal = "NEUTRAL"
            strength = 0
            desc = f"Normal volatilite. ATR: {atr_pct:.2f}%"
        
        return IndicatorSignal("ATR", atr_pct, signal, strength, desc)
    
    # =========================================
    # 4. STOCHASTIC RSI
    # =========================================
    
    def _stochastic_rsi(self, closes: List[float], period: int = 14) -> IndicatorSignal:
        """
        Stochastic RSI - RSI'ın RSI'ı (daha hassas)
        """
        if len(closes) < period * 2:
            return IndicatorSignal("Stoch RSI", 0, "NEUTRAL", 0, "Yetersiz veri")
        
        # Calculate RSI first
        rsi_values = []
        for i in range(period, len(closes)):
            changes = [closes[j] - closes[j-1] for j in range(i-period+1, i+1)]
            gains = [c if c > 0 else 0 for c in changes]
            losses = [-c if c < 0 else 0 for c in changes]
            
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        if len(rsi_values) < period:
            return IndicatorSignal("Stoch RSI", 0, "NEUTRAL", 0, "Yetersiz veri")
        
        # Stochastic of RSI
        recent_rsi = rsi_values[-period:]
        min_rsi = min(recent_rsi)
        max_rsi = max(recent_rsi)
        current_rsi = rsi_values[-1]
        
        if max_rsi == min_rsi:
            stoch_rsi = -1  # -1 = veri yok (ESKİ: 50 YANLIŞ!)
        else:
            stoch_rsi = (current_rsi - min_rsi) / (max_rsi - min_rsi) * 100
        
        # Signal
        if stoch_rsi < 20:
            signal = "BUY"
            strength = (20 - stoch_rsi) / 20
            desc = f"Aşırı satım (Stoch RSI: {stoch_rsi:.0f})"
        elif stoch_rsi > 80:
            signal = "SELL"
            strength = (stoch_rsi - 80) / 20
            desc = f"Aşırı alım (Stoch RSI: {stoch_rsi:.0f})"
        else:
            signal = "NEUTRAL"
            strength = 0
            desc = f"Normal (Stoch RSI: {stoch_rsi:.0f})"
        
        return IndicatorSignal("Stoch RSI", stoch_rsi, signal, strength, desc)
    
    # =========================================
    # 5. ADX (Trend Strength)
    # =========================================
    
    def _adx(self, closes: List[float], highs: List[float], lows: List[float],
             period: int = 14) -> IndicatorSignal:
        """
        ADX - Average Directional Index (trend strength)
        """
        if len(closes) < period * 2:
            return IndicatorSignal("ADX", 0, "NEUTRAL", 0, "Yetersiz veri")
        
        # +DM and -DM
        plus_dm = []
        minus_dm = []
        tr_list = []
        
        for i in range(1, len(closes)):
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            
            if up > down and up > 0:
                plus_dm.append(up)
            else:
                plus_dm.append(0)
            
            if down > up and down > 0:
                minus_dm.append(down)
            else:
                minus_dm.append(0)
            
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        # Smoothed averages
        atr = sum(tr_list[-period:]) / period
        plus_di = (sum(plus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0
        minus_di = (sum(minus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0
        
        # ADX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = (di_diff / di_sum * 100) if di_sum > 0 else 0
        adx = dx  # Simplified (should be smoothed)
        
        # Signal
        if adx > 25 and plus_di > minus_di:
            signal = "BUY"
            strength = min(1.0, (adx - 25) / 25)
            desc = f"Güçlü YUKARI trend (ADX: {adx:.0f}, +DI > -DI)"
        elif adx > 25 and minus_di > plus_di:
            signal = "SELL"
            strength = min(1.0, (adx - 25) / 25)
            desc = f"Güçlü AŞAĞI trend (ADX: {adx:.0f}, -DI > +DI)"
        else:
            signal = "NEUTRAL"
            strength = 0
            desc = f"Zayıf trend (ADX: {adx:.0f})"
        
        return IndicatorSignal("ADX", adx, signal, strength, desc)
    
    # =========================================
    # 6. VOLUME PROFILE
    # =========================================
    
    def _volume_profile(self, closes: List[float], volumes: List[float],
                        current_price: float, bins: int = 10) -> IndicatorSignal:
        """
        Volume Profile - En yoğun işlem bölgelerini bul (POC)
        """
        if len(closes) < 24:
            return IndicatorSignal("Volume Profile", 0, "NEUTRAL", 0, "Yetersiz veri")
        
        # Price range
        min_price = min(closes[-24:])
        max_price = max(closes[-24:])
        bin_size = (max_price - min_price) / bins
        
        if bin_size == 0:
            return IndicatorSignal("Volume Profile", 0, "NEUTRAL", 0, "Fiyat değişmedi")
        
        # Volume at each level
        vol_profile = [0] * bins
        for i, (price, vol) in enumerate(zip(closes[-24:], volumes[-24:])):
            bin_idx = min(int((price - min_price) / bin_size), bins - 1)
            vol_profile[bin_idx] += vol
        
        # POC (Point of Control) - highest volume level
        poc_idx = vol_profile.index(max(vol_profile))
        poc_price = min_price + (poc_idx + 0.5) * bin_size
        
        # Distance from POC
        distance_pct = (current_price - poc_price) / poc_price * 100
        
        if distance_pct > 2:
            signal = "SELL"
            strength = min(1.0, distance_pct / 5)
            desc = f"POC üstünde (${poc_price:.0f}), ortalamaya dönüş olabilir"
        elif distance_pct < -2:
            signal = "BUY"
            strength = min(1.0, abs(distance_pct) / 5)
            desc = f"POC altında (${poc_price:.0f}), ortalamaya dönüş olabilir"
        else:
            signal = "NEUTRAL"
            strength = 0
            desc = f"POC seviyesinde işlem (${poc_price:.0f})"
        
        return IndicatorSignal("Volume Profile", poc_price, signal, strength, desc)
    
    # =========================================
    # 7. ICHIMOKU CLOUD
    # =========================================
    
    def _ichimoku(self, closes: List[float], highs: List[float], lows: List[float],
                  current_price: float) -> IndicatorSignal:
        """
        Ichimoku Cloud - Multi-signal indicator
        Tenkan, Kijun, Senkou Span A/B, Chikou
        """
        if len(closes) < 52:
            return IndicatorSignal("Ichimoku", 0, "NEUTRAL", 0, "Yetersiz veri")
        
        # Tenkan-sen (Conversion Line): (9-high + 9-low) / 2
        tenkan = (max(highs[-9:]) + min(lows[-9:])) / 2
        
        # Kijun-sen (Base Line): (26-high + 26-low) / 2
        kijun = (max(highs[-26:]) + min(lows[-26:])) / 2
        
        # Senkou Span A: (Tenkan + Kijun) / 2
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B: (52-high + 52-low) / 2
        senkou_b = (max(highs[-52:]) + min(lows[-52:])) / 2
        
        # Cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Signals
        bullish_signals = 0
        bearish_signals = 0
        
        # 1. Price above/below cloud
        if current_price > cloud_top:
            bullish_signals += 2
        elif current_price < cloud_bottom:
            bearish_signals += 2
        
        # 2. Tenkan cross Kijun
        if tenkan > kijun:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # 3. Senkou A > Senkou B (green cloud)
        if senkou_a > senkou_b:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Combined signal
        if bullish_signals > bearish_signals + 1:
            signal = "BUY"
            strength = min(1.0, bullish_signals / 4)
            desc = f"Bulutun üstünde + TK>KJ + Yeşil bulut"
        elif bearish_signals > bullish_signals + 1:
            signal = "SELL"
            strength = min(1.0, bearish_signals / 4)
            desc = f"Bulutun altında + TK<KJ + Kırmızı bulut"
        else:
            signal = "NEUTRAL"
            strength = 0.3
            desc = f"Bulut içinde veya karışık sinyaller"
        
        return IndicatorSignal("Ichimoku", cloud_top, signal, strength, desc)


# Singleton
_indicators: Optional[AdvancedIndicators] = None

def get_advanced_indicators() -> AdvancedIndicators:
    global _indicators
    if _indicators is None:
        _indicators = AdvancedIndicators()
    return _indicators
