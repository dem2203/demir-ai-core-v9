# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - MULTI-TIMEFRAME ANALYZER
========================================
5 farklı zaman dilimini analiz ederek confluence sinyali üretir:
- 5m: Scalp signals
- 15m: Intraday
- 1h: Swing (ana)
- 4h: Position
- 1d: Trend direction

Confluence: 3+ timeframe aynı yönde = güçlü sinyal
"""
import logging
import aiohttp
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("MULTI_TIMEFRAME")


@dataclass
class TimeframeTrend:
    """Tek timeframe trend bilgisi"""
    timeframe: str
    trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    rsi: float
    ema_trend: str
    strength: float  # 0-1


@dataclass
class MTFAnalysis:
    """Multi-timeframe analiz sonucu"""
    timeframes: List[TimeframeTrend]
    confluence_direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confluence_score: int  # Kaç TF aynı yönde (0-5)
    confidence: float  # 0-100
    primary_signals: List[str]  # Ana sinyaller
    is_valid: bool


class MultiTimeframeAnalyzer:
    """
    5 timeframe'i paralel analiz et ve confluence bul.
    """
    
    TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']
    FUTURES_BASE = "https://fapi.binance.com"
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("📊 Multi-Timeframe Analyzer initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers=self.HEADERS
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def analyze(self, symbol: str) -> MTFAnalysis:
        """
        Tek coin için 5 timeframe analizi yap.
        """
        import asyncio
        
        # Paralel kline çekme
        tasks = [self._fetch_and_analyze_tf(symbol, tf) for tf in self.TIMEFRAMES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        timeframes = []
        bullish_count = 0
        bearish_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"⚠️ {symbol} {self.TIMEFRAMES[i]}: {result}")
                continue
            
            if result:
                timeframes.append(result)
                if result.trend == "BULLISH":
                    bullish_count += 1
                elif result.trend == "BEARISH":
                    bearish_count += 1
        
        # Confluence
        if bullish_count >= 3:
            confluence_direction = "BULLISH"
        elif bearish_count >= 3:
            confluence_direction = "BEARISH"
        else:
            confluence_direction = "NEUTRAL"
        
        confluence_score = max(bullish_count, bearish_count)
        confidence = confluence_score / 5 * 100
        
        # Primary signals
        primary_signals = []
        if confluence_score >= 4:
            primary_signals.append(f"🎯 GÜÇLÜ CONFLUENCE: {confluence_score}/5 TF {confluence_direction}")
        elif confluence_score >= 3:
            primary_signals.append(f"✅ Orta confluence: {confluence_score}/5 TF {confluence_direction}")
        
        return MTFAnalysis(
            timeframes=timeframes,
            confluence_direction=confluence_direction,
            confluence_score=confluence_score,
            confidence=confidence,
            primary_signals=primary_signals,
            is_valid=len(timeframes) >= 3
        )
    
    async def _fetch_and_analyze_tf(self, symbol: str, timeframe: str) -> Optional[TimeframeTrend]:
        """
        Tek timeframe için veri çek ve analiz et.
        """
        try:
            session = await self._get_session()
            
            # Kline sayısı timeframe'e göre
            limit = 100 if timeframe in ['5m', '15m'] else 200
            
            url = f"{self.FUTURES_BASE}/fapi/v1/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"API error: {resp.status}")
                klines = await resp.json()
            
            if len(klines) < 50:
                return None
            
            # Parse
            closes = [float(k[4]) for k in klines]
            current = closes[-1]
            
            # RSI
            rsi = self._calculate_rsi(closes[-14:])
            
            # EMA Trend
            ema_20 = self._calculate_ema(closes, 20)
            ema_50 = self._calculate_ema(closes, 50)
            
            if current > ema_20 > ema_50:
                ema_trend = "BULLISH"
            elif current < ema_20 < ema_50:
                ema_trend = "BEARISH"
            else:
                ema_trend = "NEUTRAL"
            
            # Combined trend
            bullish_points = 0
            bearish_points = 0
            
            if rsi < 40:
                bullish_points += 1
            elif rsi > 60:
                bearish_points += 1
            
            if ema_trend == "BULLISH":
                bullish_points += 2
            elif ema_trend == "BEARISH":
                bearish_points += 2
            
            # Last 3 candles momentum
            if closes[-1] > closes[-3]:
                bullish_points += 1
            elif closes[-1] < closes[-3]:
                bearish_points += 1
            
            if bullish_points > bearish_points:
                trend = "BULLISH"
                strength = min(1.0, bullish_points / 4)
            elif bearish_points > bullish_points:
                trend = "BEARISH"
                strength = min(1.0, bearish_points / 4)
            else:
                trend = "NEUTRAL"
                strength = 0.5
            
            return TimeframeTrend(
                timeframe=timeframe,
                trend=trend,
                rsi=rsi,
                ema_trend=ema_trend,
                strength=strength
            )
            
        except Exception as e:
            logger.error(f"❌ MTF {symbol} {timeframe}: {e}")
            return None
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [c if c > 0 else 0 for c in changes[-period:]]
        losses = [-c if c < 0 else 0 for c in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema


# Singleton
_mtf_analyzer: Optional[MultiTimeframeAnalyzer] = None

def get_mtf_analyzer() -> MultiTimeframeAnalyzer:
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MultiTimeframeAnalyzer()
    return _mtf_analyzer
