# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - BREAKOUT HUNTER
===============================
Ani yükselişleri ÖNCEDEN tespit eder.

Ana Prensip: Sıkışma → Patlama (Squeeze → Breakout)

Tespit Kriterleri:
1. Bollinger Squeeze (BB width < 2%)
2. Volume Building (hacim sessizce artıyor)
3. RSI Neutral (40-60 arası = hareket alanı var)
4. Order Book Imbalance (alım tarafı ağır)
5. Whale Accumulation (büyük oyuncular birikiyor)

Bu kombinasyon = BREAKOUT YAKLAŞIYOR 🚀
"""

import logging
import asyncio
import aiohttp
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("BREAKOUT_HUNTER")


@dataclass
class BreakoutSignal:
    """Breakout sinyal sonucu"""
    symbol: str
    is_squeeze: bool              # Bollinger sıkışması var mı?
    squeeze_duration: int         # Kaç mum sıkışmada?
    squeeze_tightness: float      # BB width (düşük = sıkı)
    volume_building: bool         # Hacim artıyor mu?
    volume_ratio: float           # Son hacim / ortalama hacim
    direction: str                # "BULLISH", "BEARISH", "NEUTRAL"
    breakout_probability: float   # 0-100
    trigger_price_up: float       # Bu fiyatı geçerse LONG
    trigger_price_down: float     # Bu fiyatın altına düşerse SHORT
    current_price: float
    reasoning: str
    is_imminent: bool = False     # Breakout çok yakın mı?
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'is_squeeze': self.is_squeeze,
            'squeeze_duration': self.squeeze_duration,
            'squeeze_tightness': self.squeeze_tightness,
            'volume_building': self.volume_building,
            'volume_ratio': self.volume_ratio,
            'direction': self.direction,
            'breakout_probability': self.breakout_probability,
            'trigger_price_up': self.trigger_price_up,
            'trigger_price_down': self.trigger_price_down,
            'is_imminent': self.is_imminent,
            'reasoning': self.reasoning
        }


class BreakoutHunter:
    """
    Breakout (Patlama) Avcısı
    
    Sıkışma → Patlama pattern'larını önceden tespit eder.
    Bu pattern'lar genellikle büyük hareketlerin habercisidir.
    """
    
    # Parametreler (tuning için)
    SQUEEZE_THRESHOLD = 0.025      # BB width < %2.5 = sıkışma
    VOLUME_RATIO_THRESHOLD = 1.3   # Son hacim > ortalama * 1.3
    MIN_SQUEEZE_DURATION = 3       # Minimum 3 mum sıkışmada
    BREAKOUT_PROB_THRESHOLD = 65   # %65+ = imminent
    
    def __init__(self):
        self.cache = {}
        self.last_signals = {}
        logger.info("🎯 Breakout Hunter initialized")
    
    async def analyze(self, symbol: str) -> BreakoutSignal:
        """
        Breakout potansiyelini analiz et
        """
        try:
            # Kline verisi al (15m timeframe - daha hassas)
            klines = await self._fetch_klines(symbol, interval="15m", limit=50)
            if not klines or len(klines) < 30:
                return self._empty_signal(symbol, "Yetersiz veri")
            
            # Fiyat ve hacim verileri
            closes = np.array([float(k[4]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            volumes = np.array([float(k[5]) for k in klines])
            current_price = closes[-1]
            
            # 1. BOLLINGER SQUEEZE TESPİTİ
            is_squeeze, squeeze_duration, bb_width, upper_band, lower_band, middle_band = self._detect_squeeze(closes)
            
            # 2. VOLUME BUILDING TESPİTİ
            volume_building, volume_ratio = self._detect_volume_buildup(volumes)
            
            # 3. BREAKOUT YÖNÜ TAHMİNİ
            direction, direction_score = await self._predict_direction(
                symbol, closes, highs, lows, volumes, upper_band, lower_band, current_price
            )
            
            # 4. BREAKOUT OLASILIK HESABI
            probability = self._calculate_probability(
                is_squeeze, squeeze_duration, volume_building, volume_ratio, direction_score
            )
            
            # 5. TETİKLEYİCİ FİYATLAR
            trigger_up = upper_band if upper_band else current_price * 1.005
            trigger_down = lower_band if lower_band else current_price * 0.995
            
            # 6. IMMINENT (ÇOK YAKIN) KONTROLÜ
            is_imminent = probability >= self.BREAKOUT_PROB_THRESHOLD and is_squeeze
            
            # Reasoning oluştur
            reasons = []
            if is_squeeze:
                reasons.append(f"🔥 Bollinger Squeeze ({squeeze_duration} mum, width: {bb_width*100:.1f}%)")
            if volume_building:
                reasons.append(f"📈 Volume Building ({volume_ratio:.1f}x ortalamanın üstünde)")
            if direction != "NEUTRAL":
                reasons.append(f"🎯 Yön: {direction} (skor: {direction_score:.0f})")
            if is_imminent:
                reasons.append(f"🚨 BREAKOUT YAKIN! (olasılık: %{probability:.0f})")
            
            reasoning = " | ".join(reasons) if reasons else "Normal piyasa koşulları"
            
            signal = BreakoutSignal(
                symbol=symbol,
                is_squeeze=is_squeeze,
                squeeze_duration=squeeze_duration,
                squeeze_tightness=bb_width,
                volume_building=volume_building,
                volume_ratio=volume_ratio,
                direction=direction,
                breakout_probability=probability,
                trigger_price_up=trigger_up,
                trigger_price_down=trigger_down,
                current_price=current_price,
                reasoning=reasoning,
                is_imminent=is_imminent
            )
            
            if is_imminent:
                logger.warning(f"🚨 BREAKOUT IMMINENT: {symbol} | {direction} | Prob: {probability:.0f}%")
            elif is_squeeze:
                logger.info(f"🔥 Squeeze detected: {symbol} | Duration: {squeeze_duration} | Width: {bb_width*100:.2f}%")
            
            self.last_signals[symbol] = signal
            return signal
            
        except Exception as e:
            logger.error(f"Breakout analysis error for {symbol}: {e}")
            return self._empty_signal(symbol, str(e))
    
    def _detect_squeeze(self, closes: np.ndarray) -> Tuple[bool, int, float, float, float, float]:
        """
        Bollinger Squeeze tespiti
        
        Squeeze = Bollinger bantları daralıyor = Büyük hareket yaklaşıyor
        """
        # Bollinger Bands hesapla (20 periyot, 2 std)
        period = 20
        std_dev = 2
        
        if len(closes) < period:
            return False, 0, 0.1, 0, 0, 0
        
        # SMA ve std hesapla
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        # BB Width (normalize edilmiş)
        bb_width = (upper - lower) / sma
        
        # Squeeze kontrolü
        is_squeeze = bb_width < self.SQUEEZE_THRESHOLD
        
        # Kaç mumdur sıkışmada?
        squeeze_duration = 0
        if is_squeeze:
            for i in range(min(20, len(closes) - period)):
                idx = -(i + 1)
                temp_sma = np.mean(closes[idx-period:idx if idx != -1 else None])
                temp_std = np.std(closes[idx-period:idx if idx != -1 else None])
                temp_width = (2 * temp_std * 2) / temp_sma
                if temp_width < self.SQUEEZE_THRESHOLD:
                    squeeze_duration += 1
                else:
                    break
        
        return is_squeeze, squeeze_duration, bb_width, upper, lower, sma
    
    def _detect_volume_buildup(self, volumes: np.ndarray) -> Tuple[bool, float]:
        """
        Hacim birikimi tespiti
        
        Sessizce artan hacim = Büyük oyuncular pozisyon alıyor
        """
        if len(volumes) < 20:
            return False, 1.0
        
        # Ortalama hacim (son 20 mum, son 3 hariç)
        avg_volume = np.mean(volumes[-20:-3])
        
        # Son 3 mumun hacmi
        recent_volume = np.mean(volumes[-3:])
        
        # Oran
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Building kontrolü
        volume_building = volume_ratio > self.VOLUME_RATIO_THRESHOLD
        
        return volume_building, volume_ratio
    
    async def _predict_direction(
        self, 
        symbol: str,
        closes: np.ndarray, 
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        upper_band: float,
        lower_band: float,
        current_price: float
    ) -> Tuple[str, float]:
        """
        Breakout yönünü tahmin et
        
        Faktörler:
        1. Fiyatın Bollinger içindeki pozisyonu
        2. Son mumların yönü
        3. Order book imbalance
        4. RSI momentum
        """
        score = 0  # Pozitif = BULLISH, Negatif = BEARISH
        
        # 1. Fiyat pozisyonu (Bollinger içinde)
        if upper_band and lower_band:
            bb_range = upper_band - lower_band
            if bb_range > 0:
                position = (current_price - lower_band) / bb_range
                if position > 0.7:
                    score += 20  # Üst banda yakın - bullish pressure
                elif position < 0.3:
                    score -= 20  # Alt banda yakın - bearish pressure
        
        # 2. Son 5 mumun yönü
        if len(closes) >= 5:
            recent_change = (closes[-1] - closes[-5]) / closes[-5] * 100
            if recent_change > 0.3:
                score += 15
            elif recent_change < -0.3:
                score -= 15
        
        # 3. Higher Lows (bullish) veya Lower Highs (bearish)
        if len(lows) >= 3 and len(highs) >= 3:
            if lows[-1] > lows[-2] > lows[-3]:
                score += 25  # Higher lows = bullish structure
            if highs[-1] < highs[-2] < highs[-3]:
                score -= 25  # Lower highs = bearish structure
        
        # 4. Volume on up vs down candles
        if len(closes) >= 5 and len(volumes) >= 5:
            up_volume = sum(volumes[i] for i in range(-5, 0) if closes[i] > closes[i-1])
            down_volume = sum(volumes[i] for i in range(-5, 0) if closes[i] < closes[i-1])
            if up_volume > down_volume * 1.5:
                score += 20
            elif down_volume > up_volume * 1.5:
                score -= 20
        
        # 5. Order book kontrolü (varsa)
        try:
            ob_score = await self._get_orderbook_bias(symbol)
            score += ob_score  # -30 ile +30 arası
        except:
            pass
        
        # Yön belirleme
        if score >= 25:
            direction = "BULLISH"
        elif score <= -25:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        return direction, abs(score)
    
    async def _get_orderbook_bias(self, symbol: str) -> float:
        """Order book bias hesapla"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://fapi.binance.com/fapi/v1/depth"
                params = {"symbol": symbol, "limit": 20}
                async with session.get(url, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        bids = sum(float(b[1]) for b in data.get('bids', [])[:10])
                        asks = sum(float(a[1]) for a in data.get('asks', [])[:10])
                        total = bids + asks
                        if total > 0:
                            imbalance = (bids - asks) / total
                            return imbalance * 30  # -30 ile +30 arası normalize
        except:
            pass
        return 0
    
    def _calculate_probability(
        self,
        is_squeeze: bool,
        squeeze_duration: int,
        volume_building: bool,
        volume_ratio: float,
        direction_score: float
    ) -> float:
        """
        Breakout olasılığı hesapla
        """
        probability = 30  # Base
        
        # Squeeze bonusu
        if is_squeeze:
            probability += 25
            # Uzun squeeze = daha büyük patlama
            if squeeze_duration >= 5:
                probability += 10
            elif squeeze_duration >= 3:
                probability += 5
        
        # Volume bonusu
        if volume_building:
            probability += 15
            if volume_ratio > 2.0:
                probability += 10
        
        # Direction clarity bonusu
        if direction_score >= 50:
            probability += 15
        elif direction_score >= 30:
            probability += 10
        
        return min(100, probability)
    
    async def _fetch_klines(self, symbol: str, interval: str = "15m", limit: int = 50) -> List:
        """Binance kline verisi al"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://fapi.binance.com/fapi/v1/klines"
                params = {"symbol": symbol, "interval": interval, "limit": limit}
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.error(f"Kline fetch error: {e}")
        return []
    
    def _empty_signal(self, symbol: str, reason: str) -> BreakoutSignal:
        """Boş sinyal döndür"""
        return BreakoutSignal(
            symbol=symbol,
            is_squeeze=False,
            squeeze_duration=0,
            squeeze_tightness=0.1,
            volume_building=False,
            volume_ratio=1.0,
            direction="NEUTRAL",
            breakout_probability=0,
            trigger_price_up=0,
            trigger_price_down=0,
            current_price=0,
            reasoning=reason,
            is_imminent=False
        )


# Singleton
_breakout_hunter: Optional[BreakoutHunter] = None


def get_breakout_hunter() -> BreakoutHunter:
    """Get or create Breakout Hunter instance"""
    global _breakout_hunter
    if _breakout_hunter is None:
        _breakout_hunter = BreakoutHunter()
    return _breakout_hunter


async def check_breakout(symbol: str) -> BreakoutSignal:
    """Quick access function"""
    hunter = get_breakout_hunter()
    return await hunter.analyze(symbol)
