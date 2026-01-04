# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - ADVANCED SIGNAL ENHANCER
=======================================
Whale, Liquidation ve Sentiment verilerini sinyal üretimine entegre eder.

İşleyiş:
1. Mevcut teknik sinyal > %80 güven
2. Advanced modüller ek boost/engel verir
3. Final karar: Boost varsa eşik düşer, Engel varsa sinyal iptal

Author: DEMIR AI Team
Date: 2026-01-04
"""
import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger("ADVANCED_ENHANCER")

class AdvancedSignalEnhancer:
    """
    Gelişmiş Sinyal Güçlendirici.
    
    Whale, Liquidation ve Sentiment modüllerini kullanarak:
    - Sinyal güvenini artır (boost)
    - Tehlikeli durumlarda sinyal engelle (block)
    """
    
    def __init__(self):
        self.whale_tracker = None
        self.liq_hunter = None
        self.sentiment_analyzer = None
        self.initialized = False
        
        # Cache (API çağrılarını azaltmak için)
        self.cache = {}
        self.cache_ttl = 60  # 60 saniye
        
    async def initialize(self):
        """Modülleri başlat."""
        if self.initialized:
            return
            
        try:
            # Whale Tracker
            from src.brain.whale_tracker import get_whale_tracker
            self.whale_tracker = get_whale_tracker()
            await self.whale_tracker.start()
            logger.info("🐋 Whale Tracker: ACTIVE")
        except Exception as e:
            logger.warning(f"⚠️ Whale Tracker başlatılamadı: {e}")
            
        try:
            # Liquidation Hunter
            from src.brain.liquidation_hunter import LiquidationHunter
            self.liq_hunter = LiquidationHunter()
            logger.info("💥 Liquidation Hunter: ACTIVE")
        except Exception as e:
            logger.warning(f"⚠️ Liquidation Hunter başlatılamadı: {e}")
            
        try:
            # Sentiment Analyzer
            from src.brain.sentiment_analyzer import SentimentAnalyzer
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("📊 Sentiment Analyzer: ACTIVE")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment Analyzer başlatılamadı: {e}")
            
        self.initialized = True
        logger.info("✅ Advanced Signal Enhancer initialized")
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Cache'den veri al."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Cache'e veri yaz."""
        self.cache[key] = (data, datetime.now())
    
    async def get_whale_score(self, symbol: str) -> float:
        """
        Whale aktivitesinden güven puanı üret.
        
        Returns:
            -0.10 to 0.10 arası skor
            Pozitif = Whale BUY yapıyor (Bullish)
            Negatif = Whale SELL yapıyor (Bearish)
        """
        if not self.whale_tracker:
            return 0.0
            
        cache_key = f"whale_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached.get('score', 0.0)
            
        try:
            summary = self.whale_tracker.get_whale_summary()
            
            if not summary.get('available'):
                return 0.0
            
            score = 0.0
            
            # Order Book Imbalance
            imbalance = summary.get('imbalance_ratio', 1.0)
            if imbalance > 1.5:  # Çok fazla alıcı
                score += 0.05
            elif imbalance < 0.6:  # Çok fazla satıcı
                score -= 0.05
                
            # Net Flow
            net_flow = summary.get('net_flow_usd', 0)
            if net_flow > 5_000_000:  # $5M+ net alım
                score += 0.05
            elif net_flow < -5_000_000:  # $5M+ net satım
                score -= 0.05
                
            # Status override
            status = summary.get('status', 'NEUTRAL')
            if status == 'BULLISH':
                score = max(score, 0.10)
            elif status == 'BEARISH':
                score = min(score, -0.10)
                
            self._set_cache(cache_key, {'score': score})
            logger.debug(f"🐋 Whale Score [{symbol}]: {score:.2f}")
            return score
            
        except Exception as e:
            logger.error(f"Whale score error: {e}")
            return 0.0
    
    async def get_liquidation_score(self, symbol: str) -> float:
        """
        Likiditation durumundan güven puanı üret.
        
        Returns:
            -0.10 to 0.10 arası skor
            Pozitif = Long likidasyonu yakın (short squeeze potansiyeli → BUY)
            Negatif = Short likidasyonu yakın (long squeeze potansiyeli → SELL)
        """
        if not self.liq_hunter:
            return 0.0
            
        cache_key = f"liq_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached.get('score', 0.0)
            
        try:
            # Get L/S Ratio and Funding
            ls_data = await self.liq_hunter.get_long_short_ratio(symbol)
            funding_data = await self.liq_hunter.get_funding_extremes(symbol)
            
            score = 0.0
            
            # Long/Short Ratio Analysis
            if ls_data.get('available'):
                ratio = ls_data.get('ratio', 1.0)
                if ratio > 2.0:  # Çok fazla long → crash riski
                    score -= 0.05
                elif ratio < 0.5:  # Çok fazla short → squeeze potansiyeli
                    score += 0.05
                    
            # Funding Rate Analysis
            if funding_data.get('available'):
                current_funding = funding_data.get('current_funding', 0)
                if current_funding > 0.05:  # Aşırı pozitif → short fırsatı
                    score -= 0.05
                elif current_funding < -0.03:  # Aşırı negatif → long fırsatı
                    score += 0.05
            
            self._set_cache(cache_key, {'score': score})
            logger.debug(f"💥 Liquidation Score [{symbol}]: {score:.2f}")
            return score
            
        except Exception as e:
            logger.error(f"Liquidation score error: {e}")
            return 0.0
    
    async def get_sentiment_score(self, symbol: str) -> float:
        """
        Sentiment'tan güven puanı üret.
        
        Returns:
            -0.10 to 0.10 arası skor
            Pozitif = Piyasa iyimser (BUY destekle)
            Negatif = Piyasa korkulu (dikkatli ol)
        """
        if not self.sentiment_analyzer:
            return 0.0
            
        cache_key = f"sent_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached.get('score', 0.0)
            
        try:
            # Symbol mapping
            symbol_map = {
                'BTCUSDT': 'BTC',
                'ETHUSDT': 'ETH'
            }
            sym = symbol_map.get(symbol, 'BTC')
            
            sentiment = self.sentiment_analyzer.get_sentiment(sym)
            
            if not sentiment:
                return 0.0
                
            # Composite score: -1 to 1 → scale to -0.10 to 0.10
            composite = sentiment.get('composite_score', 0)
            score = composite * 0.10
            
            self._set_cache(cache_key, {'score': score})
            logger.debug(f"📊 Sentiment Score [{symbol}]: {score:.2f}")
            return score
            
        except Exception as e:
            logger.error(f"Sentiment score error: {e}")
            return 0.0
    
    async def get_boost_score(self, symbol: str) -> float:
        """
        Toplam boost skoru hesapla.
        
        Returns:
            -0.15 to 0.15 arası toplam boost
            Pozitif = Eşiği düşür (daha kolay sinyal)
            Negatif = Eşiği yükselt (daha zor sinyal)
        """
        whale = await self.get_whale_score(symbol)
        liq = await self.get_liquidation_score(symbol)
        sent = await self.get_sentiment_score(symbol)
        
        # Weighted average: Whale en önemli
        total = whale * 0.5 + liq * 0.3 + sent * 0.2
        
        # Cap at ±0.15
        total = max(-0.15, min(0.15, total))
        
        logger.info(f"🎯 Boost Score [{symbol}]: {total:.3f} (W:{whale:.2f} L:{liq:.2f} S:{sent:.2f})")
        return total
    
    async def should_block_signal(self, symbol: str, side: str) -> tuple[bool, str]:
        """
        Sinyal engellenmelimi kontrol et.
        
        Args:
            symbol: Trading pair
            side: 'BUY' veya 'SELL'
            
        Returns:
            (block: bool, reason: str)
        """
        # 1. Whale ters yönde hareket ediyorsa
        whale_score = await self.get_whale_score(symbol)
        if side == "BUY" and whale_score < -0.08:
            return True, "Whale SELL yapıyor, BUY engellendi"
        if side == "SELL" and whale_score > 0.08:
            return True, "Whale BUY yapıyor, SELL engellendi"
        
        # 2. Aşırı sentiment (contrarian)
        sent_score = await self.get_sentiment_score(symbol)
        if side == "BUY" and sent_score > 0.08:  # Herkes çok iyimser → tehlikeli
            return True, "Aşırı Greed, BUY riski yüksek"
        if side == "SELL" and sent_score < -0.08:  # Herkes çok korkmuş → dip olabilir
            return True, "Aşırı Fear, SELL riski yüksek"
        
        return False, ""
    
    async def get_enhanced_signal_data(self, symbol: str, side: str, base_confidence: float) -> Dict:
        """
        Gelişmiş sinyal verisi üret.
        
        Args:
            symbol: Trading pair
            side: 'BUY' veya 'SELL'
            base_confidence: Teknik modelden gelen güven (0-1)
            
        Returns:
            {
                'blocked': bool,
                'block_reason': str,
                'boost': float,
                'final_confidence': float,
                'whale_status': str,
                'sentiment': str
            }
        """
        # Engel kontrolü
        blocked, reason = await self.should_block_signal(symbol, side)
        
        # Boost hesapla
        boost = await self.get_boost_score(symbol)
        
        # Final güven (boost sinyal yönüne uyumluysa ekle)
        final_confidence = base_confidence
        if side == "BUY" and boost > 0:
            final_confidence += boost
        elif side == "SELL" and boost < 0:
            final_confidence += abs(boost)
        elif (side == "BUY" and boost < 0) or (side == "SELL" and boost > 0):
            final_confidence -= abs(boost) * 0.5  # Ters yön hafif düşür
        
        # Cap at 1.0
        final_confidence = min(1.0, final_confidence)
        
        return {
            'blocked': blocked,
            'block_reason': reason,
            'boost': boost,
            'final_confidence': final_confidence,
            'whale_score': await self.get_whale_score(symbol),
            'liq_score': await self.get_liquidation_score(symbol),
            'sentiment_score': await self.get_sentiment_score(symbol)
        }
    
    async def close(self):
        """Temizlik."""
        if self.whale_tracker:
            await self.whale_tracker.stop()
        if self.liq_hunter:
            await self.liq_hunter.close()
        logger.info("🛑 Advanced Signal Enhancer closed")


# Singleton
_enhancer = None

def get_advanced_enhancer() -> AdvancedSignalEnhancer:
    """Get or create singleton enhancer."""
    global _enhancer
    if _enhancer is None:
        _enhancer = AdvancedSignalEnhancer()
    return _enhancer
