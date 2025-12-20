# -*- coding: utf-8 -*-
"""
DEMIR AI - THINKING BRAIN
==========================
Gercek dusunen yapay zeka.

Kural tabanli degil, dusunce tabanli:
- Senaryo analizi yapar
- Gelecegi kurgular
- Hikayelestirir
- Karar verir

Author: DEMIR AI Core Team
Date: 2024-12
"""
import logging
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import requests

logger = logging.getLogger("THINKING_BRAIN")

# Hafiza dizini
MEMORY_DIR = Path("src/thinking_brain/memory_data")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Observation:
    """Piyasa gozlemi."""
    symbol: str
    price: float
    fear_greed: int
    funding_rate: float
    whale_flow: str  # 'BUYING', 'SELLING', 'NEUTRAL'
    whale_volume: float
    volume_ratio: float
    rsi: float
    ema_trend: str  # 'BULLISH', 'BEARISH', 'MIXED'
    squeeze: bool
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Scenario:
    """Gelecek senaryosu."""
    name: str
    probability: float
    description: str
    invalidation_point: float
    confirmation_point: float

@dataclass
class Decision:
    """Nihai karar."""
    action: str  # LONG, SHORT, WAIT
    symbol: str
    confidence: float
    narrative: str  # Hikaye/Gerekce
    scenarios: List[Scenario]
    entry_price: float = 0
    target_price: float = 0
    stop_loss: float = 0
    conditions: List[str] = field(default_factory=list)

@dataclass 
class Memory:
    """Gecmis tahmin hafizasi."""
    id: str
    symbol: str
    prediction: str
    reasoning: str
    result: str  # 'WIN', 'LOSS', 'PENDING'
    pnl_percent: float
    lesson: str
    timestamp: str


# =============================================================================
# THINKING BRAIN
# =============================================================================

class ThinkingBrain:
    """
    GERCEK DUSUNEN AI BEYNI v3.0
    
    Kural tabanli degil, SENARYO ve HIKAYE tabanli calisir.
    Bir insan trader gibi:
    1. Hikayeyi kurar (Narrative Construction)
    2. Senaryolari oynatir (Scenario Simulation)
    3. Risk/Odul hesabi yapar (Risk Assessment)
    4. Karar verir (Execution)
    """
    
    def __init__(self):
        self.memory_path = Path(__file__).parent / 'memory_data'
        self.memory_path.mkdir(exist_ok=True)
        self.predictions_file = self.memory_path / 'predictions.json'
        self.logger = logging.getLogger("THINKING_BRAIN")
        self.memories: List[Memory] = []
        self._load_memory()
        
    async def think(self, symbol: str = 'BTCUSDT') -> Decision:
        """
        INSAN GIBI DUSUNME SURECI
        """
        self.logger.info(f"Thinking started for {symbol}")
        
        # 1. GOZLEM (Veri Toplama)
        obs = await self._observe(symbol)
        
        # 2. KARAKTER ANALIZI (Piyasa Ruh Hali)
        mood = self._analyze_mood(obs)
        
        # 3. SENARYO OLUSTURMA (Gelecegi Hayal Etme)
        bull_case = self._construct_bull_case(obs)
        bear_case = self._construct_bear_case(obs)
        
        # 4. SENARYO CARPISTIRMA (Hangisi daha mantikli?)
        winner, narrative = self._synthesize_narrative(obs, mood, bull_case, bear_case)
        
        # 5. KARAR (Eylem Plani)
        decision = self._formulate_plan(symbol, winner, narrative, obs)
        
        # Hafizaya at
        self._save_prediction(decision)
        
        self.logger.info(f"Decision made: {decision.action}")
        return decision

    def _analyze_mood(self, obs: Observation) -> str:
        """Piyasanin ruh halini anla."""
        factors = []
        if obs.fear_greed <= 25: factors.append("PANIK")
        elif obs.fear_greed >= 75: factors.append("COSKU")
        else: factors.append("NOTR")
        
        if obs.funding_rate > 0.05: factors.append("ACGOZLU FONLAMA")
        elif obs.funding_rate < 0: factors.append("KORKAK FONLAMA")
        
        if obs.whale_flow == 'BUYING' and obs.whale_volume > 50: factors.append("BALINA TOPLUYOR")
        elif obs.whale_flow == 'SELLING' and obs.whale_volume > 50: factors.append("BALINA BOSALTIYOR")
        
        if obs.squeeze: factors.append("PATLAMAYA HAZIR")
        
        return ", ".join(factors)

    def _construct_bull_case(self, obs: Observation) -> Scenario:
        """Yukselis senaryosunu kurgula."""
        prob_score = 10 # Base score
        reasons = []
        
        # Teknik
        if obs.ema_trend == 'BULLISH': 
            prob_score += 30
            reasons.append("Trend guclu (EMA)")
        if obs.rsi < 35: 
            prob_score += 20
            reasons.append("Asiri satim tepkisi (RSI)")
        
        # Whale
        if obs.whale_flow == 'BUYING':
            prob_score += 25
            reasons.append("Balinalar aliyor")
            
        # Sentiment (Contrarian)
        if obs.fear_greed < 20: # Extreme fear
            prob_score += 15
            reasons.append("Herkes korkarken al (Extreme Fear)")
            
        # Funding
        if obs.funding_rate < 0:
            prob_score += 10
            reasons.append("Short squeeze potansiyeli (Negatif Funding)")
            
        return Scenario(
            name="BOĞA SENARYOSU 🚀",
            probability=min(0.95, prob_score / 100),
            description=", ".join(reasons),
            invalidation_point=obs.price * 0.985,
            confirmation_point=obs.price * 1.005
        )

    def _construct_bear_case(self, obs: Observation) -> Scenario:
        """Dusus senaryosunu kurgula."""
        prob_score = 10
        reasons = []
        
        # Teknik
        if obs.ema_trend == 'BEARISH':
            prob_score += 30
            reasons.append("Trend zayif (EMA)")
        if obs.rsi > 65:
            prob_score += 20
            reasons.append("Asiri alim doygunlugu (RSI)")
            
        # Whale
        if obs.whale_flow == 'SELLING':
            prob_score += 25
            reasons.append("Balinalar satiyor")
            
        # Sentiment
        if obs.fear_greed > 80:
            prob_score += 15
            reasons.append("Herkes coskuluyken sat (Extreme Greed)")
            
        return Scenario(
            name="AYI SENARYOSU 📉",
            probability=min(0.95, prob_score / 100),
            description=", ".join(reasons),
            invalidation_point=obs.price * 1.015,
            confirmation_point=obs.price * 0.995
        )

    def _synthesize_narrative(self, obs: Observation, mood: str, bull: Scenario, bear: Scenario) -> Tuple[Scenario, str]:
        """Iki senaryoyu carpistir ve hikayeyi yaz."""
        
        # Probabilite farki
        diff = abs(bull.probability - bear.probability)
        is_confusing = diff < 0.15 # %15 farktan az ise kafa karisik
        
        winner = bull if bull.probability > bear.probability else bear
        
        # Hikayelestirme
        narrative = f"Piyasa su an {mood} modunda. "
        
        if is_confusing:
            narrative += f"Acikcasi kafam karisik. Bir yanda {bull.description} var, ama ote yanda {bear.description}. "
            narrative += "Buyuk oyuncular ile teknik gostergeler kavga ediyor veya sinyaller net degil. "
            narrative += "Bu ortamda islem acmak yazi tura atmak gibidir. En iyisi kenarda beklemek."
        else:
            narrative += f"Bence yon {winner.name.split(' ')[0]}. "
            narrative += f"Neden bu kadar eminim? Cunku {winner.description}. "
            loser = bear if winner == bull else bull
            narrative += f"Karsi senaryo ({loser.name.split(' ')[0]}) su an cok daha zayif ({int(loser.probability*100)}%)."
            
        return winner, narrative

    def _formulate_plan(self, symbol: str, winner: Scenario, narrative: str, obs: Observation) -> Decision:
        """Nihai karari ver."""
        
        # Eger fark az ise BEKLE
        if "kafam karisik" in narrative.lower() or winner.probability < 0.55:
            return Decision(
                action="WAIT",
                symbol=symbol,
                confidence=winner.probability,
                narrative=narrative,
                scenarios=[winner],
                conditions=[f"Net kirilim bekliyorum (${winner.confirmation_point:,.0f} ustu)"]
            )
            
        action = "LONG" if "BOĞA" in winner.name else "SHORT"
        
        # Dinamik TP/SL
        atr_pct = 0.02 # Basitlik icin sabit, normalde ATR'den gelmeli
        
        target = obs.price * (1 + (atr_pct * 1.5)) if action == 'LONG' else obs.price * (1 - (atr_pct * 1.5))
        stop = winner.invalidation_point
        
        return Decision(
            action=action,
            symbol=symbol,
            confidence=winner.probability,
            narrative=narrative,
            scenarios=[winner],
            entry_price=obs.price,
            target_price=target,
            stop_loss=stop
        )

    def format_telegram(self, d: Decision) -> str:
        """Dogal dil ciktisi - Insan gibi konusur."""
        emoji = "🟢" if d.action == "LONG" else "🔴" if d.action == "SHORT" else "👀"
        
        msg = f"{emoji} **DEMIR AI GÜNLÜĞÜ** - {d.symbol}\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # Hikaye kismi
        msg += f"💭 **Düşüncelerim:**\n_{d.narrative}_\n\n"
        
        if d.action == "WAIT":
            msg += f"⏳ **Kararım:** Şu an kenarda bekliyorum.\n"
            msg += f"📊 **Güven:** %{int(d.confidence*100)} (Yetersiz)\n"
            if d.conditions:
                msg += f"🎯 **İzlediğim Seviye:** {d.conditions[0]}\n"
        else:
            msg += f"🚀 **Kararım:** {d.action} yönünde pozisyon alıyorum.\n"
            msg += f"📊 **Güven:** %{int(d.confidence*100)}\n"
            msg += f"💰 **Giriş:** ${d.entry_price:,.2f}\n"
            msg += f"🎯 **Hedef:** ${d.target_price:,.2f}\n"
            msg += f"🛑 **Stop:** ${d.stop_loss:,.2f}\n\n"
            msg += f"📉 **Oyun Planı:** Eger fiyat ${d.stop_loss:,.2f} seviyesini ihlal ederse senaryom geçersiz olur ve çıkarım."
            
        msg += "\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"⏰ {datetime.now().strftime('%H:%M')}"
        return msg

    # =========================================================================
    # YARDIMCI FONKSIYONLAR - DATA FETCHING
    # =========================================================================
    
    async def _observe(self, symbol: str) -> Observation:
        """Piyasayi gozlemle ve veri topla."""
        try:
            # Fiyat ve teknik veriler
            price_data = await self._get_price_data(symbol)
            
            # Fear & Greed
            fear_greed = await self._get_fear_greed()
            
            # Funding
            funding = await self._get_funding(symbol)
            
            # Whale aktivitesi
            whale_flow, whale_volume = await self._get_whale_activity(symbol)
            
            # RSI hesapla
            rsi = self._calculate_rsi(price_data['closes'])
            
            # EMA trend
            ema_trend = self._calculate_ema_trend(price_data['closes'])
            
            # Squeeze tespit
            squeeze = self._detect_squeeze(price_data['closes'])
            
            # Volume ratio
            volume_ratio = self._calculate_volume_ratio(price_data['volumes'])
            
            return Observation(
                symbol=symbol,
                price=price_data['current_price'],
                fear_greed=fear_greed,
                funding_rate=funding,
                whale_flow=whale_flow,
                whale_volume=whale_volume,
                volume_ratio=volume_ratio,
                rsi=rsi,
                ema_trend=ema_trend,
                squeeze=squeeze
            )
            
        except Exception as e:
            logger.error(f"Observation failed: {e}")
            # Default gozlem
            return Observation(
                symbol=symbol,
                price=0,
                fear_greed=50,
                funding_rate=0,
                whale_flow='NEUTRAL',
                whale_volume=0,
                volume_ratio=1.0,
                rsi=50,
                ema_trend='MIXED',
                squeeze=False
            )
            
    async def _get_price_data(self, symbol: str) -> Dict:
        """Fiyat verileri al."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1h', 'limit': 50},
                timeout=10
            )
            if resp.status_code == 200:
                klines = resp.json()
                return {
                    'closes': np.array([float(k[4]) for k in klines]),
                    'volumes': np.array([float(k[5]) for k in klines]),
                    'current_price': float(klines[-1][4])
                }
        except:
            pass
        return {'closes': np.array([0]), 'volumes': np.array([0]), 'current_price': 0}
    
    async def _get_fear_greed(self) -> int:
        """Fear & Greed index al."""
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            if resp.status_code == 200:
                return int(resp.json()['data'][0]['value'])
        except:
            pass
        return 50
    
    async def _get_funding(self, symbol: str) -> float:
        """Funding rate al."""
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200 and resp.json():
                return float(resp.json()[0]['fundingRate']) * 100
        except:
            pass
        return 0
    
    async def _get_whale_activity(self, symbol: str) -> Tuple[str, float]:
        """Whale aktivitesi (orderbook'tan)."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/depth",
                params={'symbol': symbol, 'limit': 100},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                bid_vol = sum(float(b[1]) for b in data['bids'])
                ask_vol = sum(float(a[1]) for a in data['asks'])
                
                ratio = bid_vol / (ask_vol + 0.001)
                
                if ratio > 1.5:
                    return 'BUYING', bid_vol
                elif ratio < 0.67:
                    return 'SELLING', ask_vol
        except:
            pass
        return 'NEUTRAL', 0
    
    def _calculate_rsi(self, closes: np.ndarray) -> float:
        """RSI hesapla."""
        if len(closes) < 15:
            return 50
        
        delta = np.diff(closes)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        rs = avg_gain / (avg_loss + 0.001)
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema_trend(self, closes: np.ndarray) -> str:
        """EMA trend belirle."""
        if len(closes) < 21:
            return 'MIXED'
        
        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        current = closes[-1]
        
        if current > ema9 > ema21:
            return 'BULLISH'
        elif current < ema9 < ema21:
            return 'BEARISH'
        return 'MIXED'
    
    def _detect_squeeze(self, closes: np.ndarray) -> bool:
        """Bollinger squeeze tespit."""
        if len(closes) < 20:
            return False
        
        sma = np.mean(closes[-20:])
        std = np.std(closes[-20:])
        bandwidth = (2 * std / sma) * 100 if sma > 0 else 5
        
        return bandwidth < 2.0
    
    def _calculate_volume_ratio(self, volumes: np.ndarray) -> float:
        """Volume ratio hesapla."""
        if len(volumes) < 10:
            return 1.0
        
        recent = np.mean(volumes[-3:])
        avg = np.mean(volumes[-20:-3])
        
        return recent / (avg + 0.001)
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """EMA hesapla."""
        if len(data) < period:
            return data[-1]
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def _load_memory(self):
        """Hafizayi yukle."""
        mem_file = MEMORY_DIR / "predictions.json"
        if mem_file.exists():
            try:
                with open(mem_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memories = [Memory(**m) for m in data]
            except:
                self.memories = []

    def _save_prediction(self, decision: Decision):
        """Tahmini kaydet."""
        mem = Memory(
            id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=decision.symbol,
            prediction=decision.action,
            reasoning=decision.narrative,
            result='PENDING',
            pnl_percent=0,
            lesson='',
            timestamp=datetime.now().isoformat()
        )
        
        self.memories.append(mem)
        # Sadece son 50yi sakla
        if len(self.memories) > 50:
            self.memories = self.memories[-50:]
            
        try:
            with open(self.predictions_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(m) for m in self.memories], f, indent=2)
        except Exception as e:
            logger.error(f"Memory save failed: {e}")

# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_thinking_brain: Optional[ThinkingBrain] = None

def get_thinking_brain() -> ThinkingBrain:
    """Singleton instance dondur."""
    global _thinking_brain
    if _thinking_brain is None:
        _thinking_brain = ThinkingBrain()
    return _thinking_brain

# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import sys
    import io
    
    # Windows terminal encoding fix
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        brain = get_thinking_brain()
        print("BEYIN BASLATILDI.")
        print("DUSUNUYOR...")
        
        decision = await brain.think("BTCUSDT")
        
        print("\n" + "="*50)
        print(brain.format_telegram(decision))
        print("="*50)
    
    asyncio.run(test())
