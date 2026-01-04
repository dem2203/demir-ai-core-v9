# -*- coding: utf-8 -*-
"""
DEMIR AI v11.1 - AI COUNCIL V2
==============================
4 AI + ML Model birlikte piyasa analizi yapar.

Üyeler:
1. GPT-4o Vision - Chart görsel analizi
2. Gemini 1.5 Vision - Chart görsel analizi  
3. Claude 3 - Teknik veri analizi
4. DEMIR v11 - LightGBM + Whale + Liq + Sentiment

Author: DEMIR AI Team
Date: 2026-01-04
"""
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("AI_COUNCIL")


@dataclass
class AIVote:
    """Tek bir AI'ın oyu."""
    model_name: str
    direction: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    reasoning: str
    emoji: str


@dataclass
class CouncilDecision:
    """Konsey kararı."""
    consensus: str  # BUY, SELL, HOLD
    agreement_pct: float  # 0-100
    votes: List[AIVote]
    timestamp: datetime
    symbol: str


class AICouncilV2:
    """
    AI Konseyi - 4 Farklı Kaynak
    
    Her kaynak bağımsız analiz yapar, sonuçlar birleştirilir.
    """
    
    def __init__(self):
        self.gpt4o = None
        self.gemini = None
        self.claude = None
        self.demir_v11 = None
        
        self._init_models()
    
    def _init_models(self):
        """Modelleri başlat."""
        # GPT-4o + Gemini (mevcut vision_analyst kullanılacak)
        try:
            from src.brain.vision_analyst import VisionAnalyst
            self.vision = VisionAnalyst()
            logger.info("✅ Vision Analyst (GPT-4o + Gemini) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Vision Analyst failed: {e}")
            self.vision = None
        
        # Claude 3
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.claude = anthropic.Anthropic(api_key=api_key)
                logger.info("✅ Claude 3 initialized")
            else:
                logger.warning("⚠️ ANTHROPIC_API_KEY not found")
        except Exception as e:
            logger.warning(f"⚠️ Claude init failed: {e}")
        
        # DEMIR v11
        try:
            from src.execution.signal_generator import SignalGenerator
            from src.execution.advanced_signals import get_advanced_enhancer
            self.demir_generator = SignalGenerator(["BTCUSDT", "ETHUSDT"], use_advanced=False)
            self.demir_enhancer = get_advanced_enhancer()
            logger.info("✅ DEMIR v11 initialized")
        except Exception as e:
            logger.warning(f"⚠️ DEMIR v11 init failed: {e}")
            self.demir_generator = None
            self.demir_enhancer = None
    
    async def analyze(self, symbol: str, df=None) -> CouncilDecision:
        """
        Tüm AI'ları paralel çalıştır ve konsensüs hesapla.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            df: Optional DataFrame with OHLCV data
            
        Returns:
            CouncilDecision with consensus and all votes
        """
        logger.info(f"🧠 AI Council analyzing {symbol}...")
        
        # Veriyi hazırla (eğer verilmediyse)
        if df is None:
            df = await self._fetch_data(symbol)
        
        # Paralel analiz
        tasks = []
        
        # 1. Vision AI (GPT-4o + Gemini birlikte)
        if self.vision and df is not None:
            tasks.append(self._query_vision(symbol, df))
        
        # 2. Claude 3
        if self.claude and df is not None:
            tasks.append(self._query_claude(symbol, df))
        
        # 3. DEMIR v11
        if self.demir_generator:
            tasks.append(self._query_demir(symbol, df))
        
        # Sonuçları topla
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Hataları filtrele ve oyları topla
        votes = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"AI query error: {r}")
            elif isinstance(r, list):
                votes.extend(r)  # Vision birden fazla oy döndürebilir
            elif isinstance(r, AIVote):
                votes.append(r)
        
        # Konsensüs hesapla
        consensus = self._calculate_consensus(votes)
        
        return CouncilDecision(
            consensus=consensus['direction'],
            agreement_pct=consensus['agreement'],
            votes=votes,
            timestamp=datetime.now(),
            symbol=symbol
        )
    
    async def _fetch_data(self, symbol: str):
        """Binance'dan veri çek - Robust Implementation."""
        try:
            from src.data_pipeline.collector import get_data_collector
            collector = get_data_collector()
            
            # 1. Mevcut veriyi güncelle
            df = await collector.update_symbol(symbol, interval="1m")
            
            # 2. Eğer veri yetersizse, zorla son 3 günü indir
            if df is None or len(df) < 100:
                logger.warning(f"⚠️ Insufficient data ({len(df) if df is not None else 0} rows), forcing new download...")
                df = await collector.download_symbol(symbol, interval="1m", days=3)
            
            return df
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return None
    
    async def _query_vision(self, symbol: str, df) -> List[AIVote]:
        """GPT-4o ve Gemini'den görsel analiz al."""
        votes = []
        
        try:
            result = await asyncio.to_thread(self.vision.analyze_chart, symbol, df)
            
            if not result or not result.get('available'):
                return votes
            
            # Her model için ayrı oy
            for model_result in result.get('models', []):
                model_name = model_result.get('model', 'Unknown')
                direction = model_result.get('direction', 'HOLD').upper()
                confidence = model_result.get('confidence', 50) / 100
                reasoning = model_result.get('reasoning', '')[:200]
                
                emoji = "🤖" if "gpt" in model_name.lower() else "🔮"
                
                votes.append(AIVote(
                    model_name=model_name,
                    direction=direction,
                    confidence=confidence,
                    reasoning=reasoning,
                    emoji=emoji
                ))
                
        except Exception as e:
            logger.error(f"Vision query error: {e}")
        
        return votes
    
    async def _query_claude(self, symbol: str, df) -> AIVote:
        """Claude 3 ile teknik analiz."""
        try:
            # Son 100 mum için özet istatistikler
            recent = df.tail(100)
            
            # Teknik özet oluştur
            current_price = float(recent['close'].iloc[-1])
            
            # Safe metrics
            if len(recent) >= 60:
                price_change_1h = (current_price / float(recent['close'].iloc[-60]) - 1) * 100
            else:
                price_change_1h = 0.0
                
            high_24h = float(recent['high'].max())
            low_24h = float(recent['low'].min())
            
            # Safe volume trend
            if len(recent) >= 30:
                vol_recent = float(recent['volume'].tail(15).mean())
                vol_past = float(recent['volume'].head(15).mean())
                volume_trend = "Artıyor" if vol_recent > vol_past else "Azalıyor"
            else:
                volume_trend = "Nötr"
            
            prompt = f"""Sen profesyonel bir kripto trader'sın. Aşağıdaki verileri analiz et ve kısa bir değerlendirme yap.

{symbol} Teknik Özet:
- Güncel Fiyat: ${current_price:,.2f}
- Son 1 Saat Değişim: %{price_change_1h:.2f}
- 24 Saat En Yüksek: ${high_24h:,.2f}
- 24 Saat En Düşük: ${low_24h:,.2f}
- Hacim Trendi: {volume_trend}
- Fiyat Aralığı: ${high_24h - low_24h:,.2f}
- Mevcut Pozisyon: Aralığın %{(current_price - low_24h) / (high_24h - low_24h) * 100 if high_24h != low_24h else 50:.0f}'inde

JSON formatında yanıt ver:
{{"direction": "BUY/SELL/HOLD", "confidence": 0-100, "reasoning": "kısa açıklama"}}"""

            response = self.claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            import json
            text = response.content[0].text
            
            # JSON'ı bul
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return AIVote(
                    model_name="Claude 3.5 Sonnet",
                    direction=data.get('direction', 'HOLD').upper(),
                    confidence=min(1.0, data.get('confidence', 50) / 100),
                    reasoning=data.get('reasoning', '')[:200],
                    emoji="🧪"
                )
                
        except Exception as e:
            logger.error(f"Claude query error: {e}")
        
        return AIVote(
            model_name="Claude 3.5 Sonnet",
            direction="HOLD",
            confidence=0.5,
            reasoning="Analiz yapılamadı",
            emoji="🧪"
        )
    
    async def _query_demir(self, symbol: str, df=None) -> AIVote:
        """DEMIR v11 ML modeli ile analiz."""
        try:
            # Enhancer initialize et
            if self.demir_enhancer and not self.demir_enhancer.initialized:
                await self.demir_enhancer.initialize()
            
            # Veri kontrol
            if df is None:
                df = await self._fetch_data(symbol)
            
            if df is None or len(df) < 100:
                raise Exception(f"Yetersiz veri ({len(df) if df is not None else 0} satır)")
            
            # Feature hesapla
            from src.features.technical import TechnicalFeatures
            fe = TechnicalFeatures()
            df_features = fe.calculate_all(df)
            
            # Model yükle ve tahmin yap
            from src.models.trainer import QuantModelTrainer
            model_name = f"quant_{symbol.lower().replace('usdt', '')}"
            trainer = QuantModelTrainer(model_name)
            trainer.load_model()
            
            # Son satır
            last_row = df_features.iloc[[-1]]
            feature_cols = trainer.feature_columns
            
            if not feature_cols:
                exclude = ['timestamp', 'label_1h', 'label_4h', 'label_4h_triple', 
                           'future_return_60', 'future_return_240', 'symbol', 'close']
                feature_cols = [c for c in df_features.columns if c not in exclude]
            
            # Eksik kolonları doldur
            for c in set(feature_cols) - set(last_row.columns):
                last_row[c] = 0
            
            X = last_row[feature_cols]
            probs = trainer.model.predict_proba(X)
            prob_buy = probs[0][1] if len(probs[0]) > 1 else probs[0][0]
            
            # Yön belirle
            direction = "HOLD"
            if prob_buy > 0.65:
                direction = "BUY"
            elif prob_buy < 0.35:
                direction = "SELL"
            
            # Advanced skorları al
            reasoning_parts = [f"ML: {prob_buy*100:.0f}%"]
            
            if self.demir_enhancer:
                whale = await self.demir_enhancer.get_whale_score(symbol)
                liq = await self.demir_enhancer.get_liquidation_score(symbol)
                sent = await self.demir_enhancer.get_sentiment_score(symbol)
                
                if whale != 0:
                    reasoning_parts.append(f"🐋 Whale: {whale:+.2f}")
                if liq != 0:
                    reasoning_parts.append(f"💥 Liq: {liq:+.2f}")
                if sent != 0:
                    reasoning_parts.append(f"📊 Sent: {sent:+.2f}")
            
            return AIVote(
                model_name="DEMIR v11 ML",
                direction=direction,
                confidence=prob_buy if direction == "BUY" else (1 - prob_buy if direction == "SELL" else 0.5),
                reasoning=" | ".join(reasoning_parts),
                emoji="🦾"
            )
            
        except Exception as e:
            logger.error(f"DEMIR query error: {e}")
            return AIVote(
                model_name="DEMIR v11 ML",
                direction="HOLD",
                confidence=0.5,
                reasoning=f"Hata: {str(e)[:50]}",
                emoji="🦾"
            )
    
    def _calculate_consensus(self, votes: List[AIVote]) -> Dict:
        """Oylardan konsensüs hesapla."""
        if not votes:
            return {'direction': 'HOLD', 'agreement': 0}
        
        # Weighted voting - güven puanına göre ağırlıklı
        buy_weight = sum(v.confidence for v in votes if v.direction == "BUY")
        sell_weight = sum(v.confidence for v in votes if v.direction == "SELL")
        hold_weight = sum(v.confidence for v in votes if v.direction == "HOLD")
        
        total = buy_weight + sell_weight + hold_weight
        if total == 0:
            return {'direction': 'HOLD', 'agreement': 0}
        
        # En yüksek ağırlık kazanır
        if buy_weight >= sell_weight and buy_weight >= hold_weight:
            direction = "BUY"
            agreement = (buy_weight / total) * 100
        elif sell_weight >= buy_weight and sell_weight >= hold_weight:
            direction = "SELL"
            agreement = (sell_weight / total) * 100
        else:
            direction = "HOLD"
            agreement = (hold_weight / total) * 100
        
        return {
            'direction': direction,
            'agreement': round(agreement, 1)
        }
    
    def format_report(self, decision: CouncilDecision) -> str:
        """Telegram için rapor formatla."""
        emoji = "🟢" if decision.consensus == "BUY" else "🔴" if decision.consensus == "SELL" else "⚪"
        
        report = (
            f"🧠 **AI COUNCIL ANALİZİ** - {decision.symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{emoji} **KONSENSÜS: {decision.consensus}** (%{decision.agreement_pct:.0f} anlaşma)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )
        
        # Her AI'ın oyunu ekle
        for vote in decision.votes:
            vote_emoji = "🟢" if vote.direction == "BUY" else "🔴" if vote.direction == "SELL" else "⚪"
            
            report += (
                f"{vote.emoji} **{vote.model_name}:**\n"
                f"   {vote_emoji} {vote.direction} (%{vote.confidence*100:.0f} güven)\n"
                f"   _{vote.reasoning}_\n\n"
            )
        
        # Özet ve öneri
        buy_count = sum(1 for v in decision.votes if v.direction == "BUY")
        sell_count = sum(1 for v in decision.votes if v.direction == "SELL")
        total = len(decision.votes)
        
        if decision.consensus == "BUY" and decision.agreement_pct >= 70:
            advice = f"💡 **ÖNERİ:** {buy_count}/{total} AI 'BUY' diyor. Güçlü sinyal!"
        elif decision.consensus == "SELL" and decision.agreement_pct >= 70:
            advice = f"💡 **ÖNERİ:** {sell_count}/{total} AI 'SELL' diyor. Dikkatli ol!"
        else:
            advice = f"💡 **ÖNERİ:** Konsensüs zayıf (%{decision.agreement_pct:.0f}). Bekle."
        
        report += (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{advice}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ _Analiz zamanı: {decision.timestamp.strftime('%H:%M:%S')}_"
        )
        
        return report


# Singleton
_council = None

def get_ai_council() -> AICouncilV2:
    """Get or create singleton AI Council."""
    global _council
    if _council is None:
        _council = AICouncilV2()
    return _council
