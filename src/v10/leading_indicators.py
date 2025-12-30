# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - Leading Indicators Module
=========================================
Hareket BAŞLAMADAN ÖNCE sinyal veren öncü göstergeler.

Bu göstergeler gecikmeli (lagging) değil, öncü (leading) göstergelerdir:
- Whale Accumulation: Büyük oyuncuların pozisyon alması
- Order Book Imbalance: Alım/satım dengesizliği
- OI + Price Divergence: Gizli pozisyon birikimi
- Funding Rate Spike: Aşırı long/short yığılması
- Volume Precursor: Hareket öncesi hacim artışı
"""
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("LEADING_INDICATORS")


class SignalDirection(Enum):
    """Sinyal yönü"""
    STRONG_BULLISH = "STRONG_BULLISH"   # Güçlü alım sinyali
    BULLISH = "BULLISH"                  # Alım sinyali
    NEUTRAL = "NEUTRAL"                  # Nötr
    BEARISH = "BEARISH"                  # Satış sinyali
    STRONG_BEARISH = "STRONG_BEARISH"   # Güçlü satış sinyali


@dataclass
class IndicatorResult:
    """Tek bir göstergenin sonucu"""
    name: str
    value: float           # -100 to +100 (negatif = bearish, pozitif = bullish)
    confidence: float      # 0-100 güven skoru
    direction: SignalDirection
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LeadingSignal:
    """Birleşik leading sinyal"""
    symbol: str
    direction: SignalDirection
    strength: float        # 0-100
    confidence: float      # 0-100
    indicators: List[IndicatorResult] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Individual indicator scores for AI Brain
    orderbook_score: float = 0.0   # -100 to +100
    whale_score: float = 0.0       # -100 to +100
    funding_score: float = 0.0     # -100 to +100
    oi_divergence_score: float = 0.0  # -100 to +100
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'strength': round(self.strength, 1),
            'confidence': round(self.confidence, 1),
            'reasoning': self.reasoning,
            'indicators': {i.name: i.value for i in self.indicators},
            'timestamp': self.timestamp.isoformat()
        }


class LeadingIndicators:
    """
    Leading Indicators Calculator
    
    5 öncü göstergeyi hesaplar ve birleştirir.
    """
    
    # Gösterge ağırlıkları
    WEIGHTS = {
        'whale': 0.25,
        'orderbook': 0.25,
        'oi_divergence': 0.20,
        'funding': 0.15,
        'volume': 0.15
    }
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._cache_ttl = 30  # 30 saniye cache
        
        logger.info("📊 Leading Indicators initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': 'Mozilla/5.0'}
            )
        return self._session
    
    async def close(self):
        if self._session:
            await self._session.close()
    
    async def calculate_all(self, symbol: str = "BTCUSDT") -> LeadingSignal:
        """
        Tüm leading göstergeleri hesapla ve birleştir.
        """
        logger.info(f"🔍 Calculating leading indicators for {symbol}...")
        
        # Paralel olarak tüm göstergeleri hesapla
        results = await asyncio.gather(
            self._calculate_whale_accumulation(symbol),
            self._calculate_orderbook_imbalance(symbol),
            self._calculate_oi_divergence(symbol),
            self._calculate_funding_spike(symbol),
            self._calculate_volume_precursor(symbol),
            return_exceptions=True
        )
        
        # Hataları filtrele
        indicators = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Indicator calculation failed: {result}")
                continue
            if result:
                indicators.append(result)
        
        # Birleşik sinyal hesapla
        signal = self._combine_signals(symbol, indicators)
        
        return signal
    
    # =========================================
    # 1. WHALE ACCUMULATION
    # =========================================
    
    async def _calculate_whale_accumulation(self, symbol: str) -> Optional[IndicatorResult]:
        """
        Whale (balina) birikimi tespit et.
        
        Bakılan veriler:
        - Büyük işlemler (>$100K)
        - Net alım/satım
        - Exchange inflow/outflow
        """
        try:
            session = await self._get_session()
            
            # Binance büyük işlemler (son 1 saat)
            base_symbol = symbol.replace("USDT", "").lower()
            
            # Aggr trades al
            url = f"https://fapi.binance.com/fapi/v1/aggTrades"
            params = {'symbol': symbol, 'limit': 1000}
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                trades = await resp.json()
            
            if not trades:
                return None
            
            # Büyük işlemleri analiz et
            big_buys = 0
            big_sells = 0
            threshold = 50000  # $50K üstü işlemler
            
            for trade in trades:
                qty = float(trade['q'])
                price = float(trade['p'])
                value = qty * price
                
                if value >= threshold:
                    if trade['m']:  # Maker (satıcı)
                        big_sells += value
                    else:
                        big_buys += value
            
            # PHASE 15: Whale Tracker integration (with safe import)
            try:
                from src.brain.whale_tracker import get_whale_tracker
                tracker = get_whale_tracker()
                if not tracker.running:
                    # Lazy start if not running
                    try:
                        await tracker.start()
                        await asyncio.sleep(1)
                    except:
                        pass
            except Exception as whale_err:
                logger.debug(f"Whale tracker unavailable: {whale_err}")
            
            # PHASE 15: DYNAMIC THRESHOLD
            # Calculate threshold based on 24h volume
            try:
                # Fetch 24h ticker for volume
                session = await self._get_session()
                async with session.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}") as resp:
                    ticker = await resp.json()
                    quote_volume = float(ticker.get('quoteVolume', 0))
                    
                    # Rule: Threshold = 0.05% of 5-min volume estimate? 
                    # Better: Threshold = 0.1% of 1m volume?
                    # Simple heuristic: If 24h vol > $1B -> threshold $250k
                    # If 24h vol < $10M -> threshold $20k
                    # Formula: quote_volume / 20000 clipped between 20k and 500k
                    
                    calc_threshold = max(20000, min(500000, quote_volume / 20000))
                    if quote_volume > 0:
                        tracker.set_threshold(calc_threshold)
            except Exception as e:
                logger.debug(f"Dynamic threshold error: {e}")

            summary = tracker.get_whale_summary()
            # Net whale flow
            total = big_buys + big_sells
            if total == 0:
                return IndicatorResult(
                    name='whale',
                    value=0,
                    confidence=30,
                    direction=SignalDirection.NEUTRAL,
                    details={'big_buys': 0, 'big_sells': 0}
                )
            
            # -100 to +100 skalası
            net_ratio = (big_buys - big_sells) / total
            value = net_ratio * 100
            
            # Confidence: Toplam hacme göre
            confidence = min(80, (total / 1000000) * 20)  # Her $1M için 20 puan
            
            if value > 30:
                direction = SignalDirection.STRONG_BULLISH
            elif value > 10:
                direction = SignalDirection.BULLISH
            elif value < -30:
                direction = SignalDirection.STRONG_BEARISH
            elif value < -10:
                direction = SignalDirection.BEARISH
            else:
                direction = SignalDirection.NEUTRAL
            
            return IndicatorResult(
                name='whale',
                value=value,
                confidence=confidence,
                direction=direction,
                details={
                    'big_buys_usd': big_buys,
                    'big_sells_usd': big_sells,
                    'net_flow': big_buys - big_sells
                }
            )
            
        except Exception as e:
            logger.error(f"Whale calculation error: {e}")
            return None
    
    # =========================================
    # 2. ORDER BOOK IMBALANCE
    # =========================================
    
    async def _calculate_orderbook_imbalance(self, symbol: str) -> Optional[IndicatorResult]:
        """
        Order book dengesizliği.
        
        Bid (alım) tarafı > Ask (satım) tarafı = Bullish
        """
        try:
            session = await self._get_session()
            
            url = f"https://fapi.binance.com/fapi/v1/depth"
            params = {'symbol': symbol, 'limit': 20}  # İlk 20 seviye
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            
            # Toplam bid ve ask hacmi
            bid_volume = sum(float(b[1]) for b in data['bids'])
            ask_volume = sum(float(a[1]) for a in data['asks'])
            
            total = bid_volume + ask_volume
            if total == 0:
                return None
            
            # Imbalance ratio
            imbalance = (bid_volume - ask_volume) / total
            value = imbalance * 100
            
            # Wall detection (büyük emirler)
            bid_prices = [float(b[0]) for b in data['bids']]
            ask_prices = [float(a[0]) for a in data['asks']]
            bid_sizes = [float(b[1]) for b in data['bids']]
            ask_sizes = [float(a[1]) for a in data['asks']]
            
            max_bid_wall = max(bid_sizes) if bid_sizes else 0
            max_ask_wall = max(ask_sizes) if ask_sizes else 0
            
            # Confidence
            confidence = min(70, abs(value) * 2)
            
            if value > 25:
                direction = SignalDirection.STRONG_BULLISH
            elif value > 10:
                direction = SignalDirection.BULLISH
            elif value < -25:
                direction = SignalDirection.STRONG_BEARISH
            elif value < -10:
                direction = SignalDirection.BEARISH
            else:
                direction = SignalDirection.NEUTRAL
            
            return IndicatorResult(
                name='orderbook',
                value=value,
                confidence=confidence,
                direction=direction,
                details={
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'imbalance_pct': value,
                    'max_bid_wall': max_bid_wall,
                    'max_ask_wall': max_ask_wall
                }
            )
            
        except Exception as e:
            logger.error(f"Orderbook calculation error: {e}")
            return None
    
    # =========================================
    # 3. OI + PRICE DIVERGENCE
    # =========================================
    
    async def _calculate_oi_divergence(self, symbol: str) -> Optional[IndicatorResult]:
        """
        Open Interest ve Fiyat Divergence.
        
        OI artıyor + Fiyat düşüyor = Gizli long birikimi (Bullish)
        OI artıyor + Fiyat yükseliyor = Gizli short birikimi (Bearish)
        """
        try:
            session = await self._get_session()
            
            # Son 4 saatlik OI verisi
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            params = {'symbol': symbol, 'period': '15m', 'limit': 16}  # 4 saat
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                oi_data = await resp.json()
            
            if len(oi_data) < 4:
                return None
            
            # OI değişimi
            oldest_oi = float(oi_data[0]['sumOpenInterest'])
            latest_oi = float(oi_data[-1]['sumOpenInterest'])
            oi_change_pct = ((latest_oi - oldest_oi) / oldest_oi) * 100 if oldest_oi > 0 else 0
            
            # Fiyat değişimi (klines'dan)
            kline_url = f"https://fapi.binance.com/fapi/v1/klines"
            kline_params = {'symbol': symbol, 'interval': '15m', 'limit': 16}
            
            async with session.get(kline_url, params=kline_params) as resp:
                if resp.status != 200:
                    return None
                klines = await resp.json()
            
            oldest_price = float(klines[0][4])  # Close
            latest_price = float(klines[-1][4])
            price_change_pct = ((latest_price - oldest_price) / oldest_price) * 100
            
            # Divergence analizi
            # OI↑ + Price↓ = Hidden longs = Bullish
            # OI↑ + Price↑ = Hidden shorts = Bearish
            # OI↓ = Position closing = Neutral
            
            if oi_change_pct > 2:  # OI artıyor
                if price_change_pct < -1:
                    # Hidden longs
                    value = min(50, oi_change_pct * 10)
                    direction = SignalDirection.BULLISH
                elif price_change_pct > 1:
                    # Hidden shorts
                    value = max(-50, -oi_change_pct * 10)
                    direction = SignalDirection.BEARISH
                else:
                    value = 0
                    direction = SignalDirection.NEUTRAL
            elif oi_change_pct < -2:  # OI düşüyor
                value = 0  # Position closing, less informative
                direction = SignalDirection.NEUTRAL
            else:
                value = 0
                direction = SignalDirection.NEUTRAL
            
            confidence = min(60, abs(oi_change_pct) * 5)
            
            return IndicatorResult(
                name='oi_divergence',
                value=value,
                confidence=confidence,
                direction=direction,
                details={
                    'oi_change_pct': round(oi_change_pct, 2),
                    'price_change_pct': round(price_change_pct, 2),
                    'divergence_type': 'hidden_longs' if value > 0 else 'hidden_shorts' if value < 0 else 'none'
                }
            )
            
        except Exception as e:
            logger.error(f"OI divergence calculation error: {e}")
            return None
    
    # =========================================
    # 4. FUNDING RATE SPIKE
    # =========================================
    
    async def _calculate_funding_spike(self, symbol: str) -> Optional[IndicatorResult]:
        """
        Funding rate spike tespiti.
        
        Aşırı pozitif funding = Çok long = Short squeeze beklentisi
        Aşırı negatif funding = Çok short = Long squeeze beklentisi
        """
        try:
            session = await self._get_session()
            
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {'symbol': symbol, 'limit': 8}  # Son 24 saat
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            
            if not data:
                return None
            
            # Son funding rate
            latest_funding = float(data[-1]['fundingRate']) * 100  # Yüzdeye çevir
            
            # Ortalama funding
            avg_funding = sum(float(d['fundingRate']) for d in data) / len(data) * 100
            
            # Spike detection
            # Normal range: -0.01% to +0.01%
            # Elevated: ±0.03%
            # Extreme: ±0.05%+
            
            if latest_funding > 0.05:
                # Aşırı long pozisyon = Short squeeze riski
                value = -min(50, latest_funding * 500)  # Bearish (çünkü overextended)
                direction = SignalDirection.BEARISH
            elif latest_funding > 0.03:
                value = -20
                direction = SignalDirection.BEARISH
            elif latest_funding < -0.03:
                # Aşırı short pozisyon = Long squeeze riski
                value = min(50, abs(latest_funding) * 500)  # Bullish
                direction = SignalDirection.BULLISH
            elif latest_funding < -0.05:
                value = 50
                direction = SignalDirection.STRONG_BULLISH
            else:
                value = 0
                direction = SignalDirection.NEUTRAL
            
            confidence = min(60, abs(latest_funding) * 1000)
            
            return IndicatorResult(
                name='funding',
                value=value,
                confidence=confidence,
                direction=direction,
                details={
                    'current_funding': round(latest_funding, 4),
                    'avg_funding_24h': round(avg_funding, 4),
                    'signal': 'short_squeeze_risk' if latest_funding > 0.03 else 'long_squeeze_risk' if latest_funding < -0.03 else 'normal'
                }
            )
            
        except Exception as e:
            logger.error(f"Funding calculation error: {e}")
            return None
    
    # =========================================
    # 5. VOLUME PRECURSOR
    # =========================================
    
    async def _calculate_volume_precursor(self, symbol: str) -> Optional[IndicatorResult]:
        """
        Hacim öncüsü tespiti.
        
        Fiyat sabitken hacim artışı = Hareket öncesi birikim
        """
        try:
            session = await self._get_session()
            
            url = f"https://fapi.binance.com/fapi/v1/klines"
            params = {'symbol': symbol, 'interval': '15m', 'limit': 24}  # 6 saat
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                klines = await resp.json()
            
            if len(klines) < 10:
                return None
            
            # Son 4 mum vs önceki 20 mum
            recent_volume = sum(float(k[5]) for k in klines[-4:]) / 4
            historical_volume = sum(float(k[5]) for k in klines[:-4]) / (len(klines) - 4)
            
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1
            
            # Fiyat değişimi (son 4 mum)
            price_start = float(klines[-4][1])  # Open
            price_end = float(klines[-1][4])  # Close
            price_change = abs((price_end - price_start) / price_start * 100)
            
            # Buy/Sell hacmi analizi
            buy_volume = 0
            sell_volume = 0
            for k in klines[-4:]:
                taker_buy = float(k[9])  # Taker buy volume
                total_vol = float(k[5])
                buy_volume += taker_buy
                sell_volume += (total_vol - taker_buy)
            
            buy_ratio = buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.5
            
            # Volume spike + Low price movement = Accumulation
            if volume_ratio > 1.5 and price_change < 1:
                # Birikim var
                if buy_ratio > 0.55:
                    value = min(40, (volume_ratio - 1) * 30)
                    direction = SignalDirection.BULLISH
                elif buy_ratio < 0.45:
                    value = max(-40, -(volume_ratio - 1) * 30)
                    direction = SignalDirection.BEARISH
                else:
                    value = 0
                    direction = SignalDirection.NEUTRAL
            else:
                value = 0
                direction = SignalDirection.NEUTRAL
            
            confidence = min(50, (volume_ratio - 1) * 50) if volume_ratio > 1 else 30
            
            return IndicatorResult(
                name='volume',
                value=value,
                confidence=confidence,
                direction=direction,
                details={
                    'volume_ratio': round(volume_ratio, 2),
                    'price_change_pct': round(price_change, 2),
                    'buy_ratio': round(buy_ratio, 2),
                    'accumulation': volume_ratio > 1.3 and price_change < 1.5
                }
            )
            
        except Exception as e:
            logger.error(f"Volume calculation error: {e}")
            return None
    
    # =========================================
    # SIGNAL COMBINATION
    # =========================================
    
    def _combine_signals(self, symbol: str, indicators: List[IndicatorResult]) -> LeadingSignal:
        """
        Tüm göstergeleri ağırlıklı olarak birleştir.
        """
        if not indicators:
            return LeadingSignal(
                symbol=symbol,
                direction=SignalDirection.NEUTRAL,
                strength=0,
                confidence=0,
                reasoning="Gösterge verisi alınamadı"
            )
        
        # Ağırlıklı değer hesapla
        total_weight = 0
        weighted_value = 0
        weighted_confidence = 0
        
        for indicator in indicators:
            weight = self.WEIGHTS.get(indicator.name, 0.1)
            weighted_value += indicator.value * weight
            weighted_confidence += indicator.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            final_value = weighted_value / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_value = 0
            final_confidence = 0
        
        # Yön belirleme
        if final_value > 30:
            direction = SignalDirection.STRONG_BULLISH
        elif final_value > 15:
            direction = SignalDirection.BULLISH
        elif final_value < -30:
            direction = SignalDirection.STRONG_BEARISH
        elif final_value < -15:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL
        
        # Güç (strength) = |value|
        strength = min(100, abs(final_value))
        
        # Reasoning oluştur
        bullish_signals = [i.name for i in indicators if i.value > 10]
        bearish_signals = [i.name for i in indicators if i.value < -10]
        
        if bullish_signals:
            reasoning = f"Bullish: {', '.join(bullish_signals)}"
        elif bearish_signals:
            reasoning = f"Bearish: {', '.join(bearish_signals)}"
        else:
            reasoning = "No strong signals"
        
        # Extract individual scores for AI Brain
        orderbook_val = next((i.value for i in indicators if i.name == 'orderbook'), 0.0)
        whale_val = next((i.value for i in indicators if i.name == 'whale'), 0.0)
        funding_val = next((i.value for i in indicators if i.name == 'funding'), 0.0)
        oi_div_val = next((i.value for i in indicators if i.name == 'oi_divergence'), 0.0)
        
        return LeadingSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=final_confidence,
            indicators=indicators,
            reasoning=reasoning,
            orderbook_score=orderbook_val,
            whale_score=whale_val,
            funding_score=funding_val,
            oi_divergence_score=oi_div_val
        )


# Global instance
_leading_indicators: Optional[LeadingIndicators] = None


async def get_leading_indicators() -> LeadingIndicators:
    """Get or create leading indicators instance"""
    global _leading_indicators
    if _leading_indicators is None:
        _leading_indicators = LeadingIndicators()
    return _leading_indicators
