# -*- coding: utf-8 -*-
"""
DEMIR AI - THINKING BRAIN
==========================
Gercek dusunen yapay zeka.

Kural tabanli degil, dusunce tabanli:
- Gozlemler
- Hatirlar
- Celiskileri tespit eder
- Mantik yurutur
- Karar verir
- Ogrendiklerini uygular

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
class Thought:
    """Tek bir dusunce adimi."""
    step: str  # 'OBSERVE', 'REMEMBER', 'ANALYZE', 'CONFLICT', 'DECIDE'
    content: str
    confidence: float = 0.0


@dataclass
class Decision:
    """Nihai karar."""
    action: str  # 'LONG', 'SHORT', 'WAIT'
    symbol: str
    reasoning: List[Thought]
    confidence: float
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0
    conditions: List[str] = field(default_factory=list)  # "Eger X olursa Y yap"
    timestamp: datetime = field(default_factory=datetime.now)


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
    Dusunen Yapay Zeka Beyni
    
    Kural takip etmez, DUSUNUR:
    1. GOZLEM: Piyasayi gozlemle
    2. HAFIZA: Gecmisi hatirla
    3. ANALIZ: Faktorleri analiz et
    4. CELISKI: Celiskileri tespit et
    5. KARAR: Dusunup karar ver
    """
    
    def __init__(self):
        self.memories: List[Memory] = []
        self.learnings: Dict = {}
        self.factor_weights: Dict = {
            'fear_greed': 1.0,
            'whale_flow': 1.0,
            'funding': 1.0,
            'squeeze': 1.0,
            'ema': 1.0,
            'rsi': 1.0,
            'volume': 1.0
        }
        self._load_memory()
        self._load_learnings()
        logger.info("Thinking Brain initialized")
    
    # =========================================================================
    # ANA FONKSIYON: DUSUN
    # =========================================================================
    
    async def think(self, symbol: str = 'BTCUSDT') -> Decision:
        """
        ANA DUSUNME FONKSIYONU
        
        Bir insan trader gibi dusunur ve karar verir.
        """
        thoughts: List[Thought] = []
        
        # 1. GOZLEM - Piyasayi gozlemle
        thoughts.append(Thought(step='OBSERVE', content='Piyasayi gozlemliyorum...'))
        observation = await self._observe(symbol)
        
        obs_text = self._describe_observation(observation)
        thoughts.append(Thought(step='OBSERVE', content=obs_text, confidence=1.0))
        
        # 2. HAFIZA - Gecmisi hatirla
        thoughts.append(Thought(step='REMEMBER', content='Gecmis deneyimlerimi hatirliyorum...'))
        memories = self._remember_similar(observation)
        
        if memories:
            mem_text = self._describe_memories(memories)
            thoughts.append(Thought(step='REMEMBER', content=mem_text, confidence=0.8))
        else:
            thoughts.append(Thought(step='REMEMBER', content='Bu duruma benzer gecmis deneyimim yok.', confidence=0.5))
        
        # 3. ANALIZ - Faktorleri analiz et
        thoughts.append(Thought(step='ANALYZE', content='Faktorleri analiz ediyorum...'))
        signals = self._analyze_factors(observation)
        
        analysis_text = self._describe_analysis(signals)
        thoughts.append(Thought(step='ANALYZE', content=analysis_text, confidence=0.9))
        
        # 4. CELISKI - Celiskileri tespit et
        conflicts = self._detect_conflicts(signals)
        
        if conflicts:
            thoughts.append(Thought(
                step='CONFLICT', 
                content=f"DIKKAT: Celiskili sinyaller var! {conflicts}",
                confidence=0.7
            ))
        else:
            thoughts.append(Thought(
                step='CONFLICT',
                content='Sinyaller tutarli, celiski yok.',
                confidence=0.9
            ))
        
        # 5. KARAR - Mantik yuruterek karar ver
        decision = self._make_decision(observation, signals, conflicts, memories)
        decision.reasoning = thoughts
        
        # Karari acikla
        decision_text = self._describe_decision(decision)
        thoughts.append(Thought(step='DECIDE', content=decision_text, confidence=decision.confidence))
        
        # Hafizaya kaydet
        self._save_prediction(decision)
        
        return decision
    
    # =========================================================================
    # GOZLEM
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
    
    def _describe_observation(self, obs: Observation) -> str:
        """Gozlemi dogal dilde acikla."""
        lines = [f"{obs.symbol} su an ${obs.price:,.0f} seviyesinde."]
        
        # Fear & Greed
        if obs.fear_greed <= 20:
            lines.append(f"Piyasa ASIRI KORKU modunda (Fear: {obs.fear_greed}).")
        elif obs.fear_greed <= 35:
            lines.append(f"Piyasada korku hakim (Fear: {obs.fear_greed}).")
        elif obs.fear_greed >= 80:
            lines.append(f"Piyasa ASIRI ACGOZLULUK modunda (Greed: {obs.fear_greed}).")
        elif obs.fear_greed >= 65:
            lines.append(f"Piyasada acgozluluk hakim (Greed: {obs.fear_greed}).")
        else:
            lines.append(f"Piyasa duygusu notr (Index: {obs.fear_greed}).")
        
        # Whale
        if obs.whale_flow == 'BUYING':
            lines.append(f"Whale'ler ALICI pozisyonunda ({obs.whale_volume:.0f} BTC).")
        elif obs.whale_flow == 'SELLING':
            lines.append(f"Whale'ler SATICI pozisyonunda ({obs.whale_volume:.0f} BTC).")
        
        # Squeeze
        if obs.squeeze:
            lines.append("Volatilite sikismis, patlama bekleniyor.")
        
        # EMA
        if obs.ema_trend == 'BULLISH':
            lines.append("EMA'lar yukari trendde.")
        elif obs.ema_trend == 'BEARISH':
            lines.append("EMA'lar asagi trendde.")
        
        return " ".join(lines)
    
    # =========================================================================
    # HAFIZA
    # =========================================================================
    
    def _remember_similar(self, obs: Observation) -> List[Memory]:
        """Benzer gecmis durumları hatirla."""
        similar = []
        
        for mem in self.memories[-50:]:  # Son 50 hafiza
            # Benzerlik skoru
            similarity = 0
            
            # Ayni sembol
            if mem.symbol == obs.symbol:
                similarity += 1
            
            # Benzer fear/greed (+-10)
            # (Bu bilgi memory'de kayitli degil, basitlestirdik)
            
            if similarity > 0:
                similar.append(mem)
        
        return similar[-5:]  # Son 5 benzer
    
    def _describe_memories(self, memories: List[Memory]) -> str:
        """Hatiralari acikla."""
        if not memories:
            return "Benzer gecmis deneyimim yok."
        
        wins = sum(1 for m in memories if m.result == 'WIN')
        losses = sum(1 for m in memories if m.result == 'LOSS')
        
        lines = [f"Benzer {len(memories)} durum hatirliyorum:"]
        lines.append(f"- {wins} kazanc, {losses} kayip")
        
        # Son ders
        lessons = [m.lesson for m in memories if m.lesson]
        if lessons:
            lines.append(f"- Ogrendigim: {lessons[-1]}")
        
        return " ".join(lines)
    
    # =========================================================================
    # ANALIZ
    # =========================================================================
    
    def _analyze_factors(self, obs: Observation) -> Dict[str, str]:
        """Her faktoru analiz et ve sinyal uret."""
        signals = {}
        
        # Fear & Greed (contrarian)
        if obs.fear_greed <= 25:
            signals['fear_greed'] = 'LONG'
        elif obs.fear_greed >= 75:
            signals['fear_greed'] = 'SHORT'
        else:
            signals['fear_greed'] = 'NEUTRAL'
        
        # Whale
        if obs.whale_flow == 'BUYING':
            signals['whale'] = 'LONG'
        elif obs.whale_flow == 'SELLING':
            signals['whale'] = 'SHORT'
        else:
            signals['whale'] = 'NEUTRAL'
        
        # Funding (contrarian)
        if obs.funding_rate > 0.05:
            signals['funding'] = 'SHORT'  # Cok long var
        elif obs.funding_rate < -0.03:
            signals['funding'] = 'LONG'  # Cok short var
        else:
            signals['funding'] = 'NEUTRAL'
        
        # EMA
        if obs.ema_trend == 'BULLISH':
            signals['ema'] = 'LONG'
        elif obs.ema_trend == 'BEARISH':
            signals['ema'] = 'SHORT'
        else:
            signals['ema'] = 'NEUTRAL'
        
        # RSI
        if obs.rsi < 30:
            signals['rsi'] = 'LONG'
        elif obs.rsi > 70:
            signals['rsi'] = 'SHORT'
        else:
            signals['rsi'] = 'NEUTRAL'
        
        # Volume
        if obs.volume_ratio > 2.0:
            signals['volume'] = 'ATTENTION'  # Yuksek hacim = dikkat
        else:
            signals['volume'] = 'NEUTRAL'
        
        return signals
    
    def _describe_analysis(self, signals: Dict[str, str]) -> str:
        """Analizi acikla."""
        long_factors = [k for k, v in signals.items() if v == 'LONG']
        short_factors = [k for k, v in signals.items() if v == 'SHORT']
        
        lines = []
        
        if long_factors:
            lines.append(f"YUKARI isaret eden faktorler: {', '.join(long_factors)}")
        
        if short_factors:
            lines.append(f"ASAGI isaret eden faktorler: {', '.join(short_factors)}")
        
        if not long_factors and not short_factors:
            lines.append("Tum faktorler notr durumda.")
        
        return " ".join(lines)
    
    # =========================================================================
    # CELISKI TESPITI
    # =========================================================================
    
    def _detect_conflicts(self, signals: Dict[str, str]) -> Optional[str]:
        """Celiskileri tespit et."""
        long_factors = [k for k, v in signals.items() if v == 'LONG']
        short_factors = [k for k, v in signals.items() if v == 'SHORT']
        
        # Kritik celiski: Fear vs Whale
        fear_long = signals.get('fear_greed') == 'LONG'
        whale_short = signals.get('whale') == 'SHORT'
        
        if fear_long and whale_short:
            return "Fear LONG diyor ama Whale'ler SATIYOR - Bu ciddi bir celiski!"
        
        # Kritik celiski: EMA vs RSI
        ema_long = signals.get('ema') == 'LONG'
        rsi_short = signals.get('rsi') == 'SHORT'
        
        if ema_long and rsi_short:
            return "EMA yukari ama RSI asiri alim bolgisinde - Dikkatli ol!"
        
        # Genel celiski
        if long_factors and short_factors:
            return f"{len(long_factors)} faktor LONG, {len(short_factors)} faktor SHORT diyor."
        
        return None
    
    # =========================================================================
    # KARAR VERME
    # =========================================================================
    
    def _make_decision(
        self, 
        obs: Observation, 
        signals: Dict[str, str],
        conflicts: Optional[str],
        memories: List[Memory]
    ) -> Decision:
        """Dusunup karar ver."""
        
        long_count = sum(1 for v in signals.values() if v == 'LONG')
        short_count = sum(1 for v in signals.values() if v == 'SHORT')
        
        # Agirlikli skorlar (ogrenmeden)
        long_score = sum(
            self.factor_weights.get(k.split('_')[0], 1.0)
            for k, v in signals.items() if v == 'LONG'
        )
        short_score = sum(
            self.factor_weights.get(k.split('_')[0], 1.0)
            for k, v in signals.items() if v == 'SHORT'
        )
        
        # Celiski varsa ve ciddi ise BEKLE
        whale_against = (signals.get('whale') == 'SHORT' and long_count > short_count) or \
                       (signals.get('whale') == 'LONG' and short_count > long_count)
        
        if conflicts and whale_against:
            # Whale'e karsi gitme - BEKLE
            conditions = []
            
            if signals.get('whale') == 'SHORT':
                conditions.append("Whale'ler aliciya donerse LONG dusunurum")
            else:
                conditions.append("Whale'ler saticiya donerse SHORT dusunurum")
            
            conditions.append(f"${obs.price * 0.98:,.0f} altina duserse stop")
            
            return Decision(
                action='WAIT',
                symbol=obs.symbol,
                reasoning=[],
                confidence=0.6,
                conditions=conditions
            )
        
        # Karar ver
        if long_score > short_score and long_count >= 2:
            action = 'LONG'
            confidence = min(0.85, 0.5 + (long_count * 0.1))
            target = obs.price * 1.02  # %2 hedef
            stop = obs.price * 0.985   # %1.5 stop
        elif short_score > long_score and short_count >= 2:
            action = 'SHORT'
            confidence = min(0.85, 0.5 + (short_count * 0.1))
            target = obs.price * 0.98
            stop = obs.price * 1.015
        else:
            action = 'WAIT'
            confidence = 0.5
            target = 0
            stop = 0
        
        return Decision(
            action=action,
            symbol=obs.symbol,
            reasoning=[],
            confidence=confidence,
            entry_price=obs.price,
            target_price=target,
            stop_loss=stop,
            conditions=[]
        )
    
    def _describe_decision(self, decision: Decision) -> str:
        """Karari acikla."""
        if decision.action == 'WAIT':
            lines = [f"KARARIM: BEKLIYORUM ({decision.confidence*100:.0f}% guven)"]
            if decision.conditions:
                lines.append("Sartlarim:")
                for cond in decision.conditions:
                    lines.append(f"  - {cond}")
        else:
            dir_tr = 'YUKARI (LONG)' if decision.action == 'LONG' else 'ASAGI (SHORT)'
            lines = [f"KARARIM: {dir_tr} ({decision.confidence*100:.0f}% guven)"]
            lines.append(f"Giris: ${decision.entry_price:,.0f}")
            lines.append(f"Hedef: ${decision.target_price:,.0f}")
            lines.append(f"Stop: ${decision.stop_loss:,.0f}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # YARDIMCI FONKSIYONLAR - VERI TOPLAMA
    # =========================================================================
    
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
    
    # =========================================================================
    # HAFIZA YONETIMI
    # =========================================================================
    
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
    
    def _load_learnings(self):
        """Ogrenimleri yukle."""
        learn_file = MEMORY_DIR / "learnings.json"
        if learn_file.exists():
            try:
                with open(learn_file, 'r', encoding='utf-8') as f:
                    self.learnings = json.load(f)
                    # Agirlik guncellemesi
                    if 'factor_weights' in self.learnings:
                        self.factor_weights.update(self.learnings['factor_weights'])
            except:
                self.learnings = {}
    
    def _save_prediction(self, decision: Decision):
        """Tahmini kaydet."""
        mem = Memory(
            id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=decision.symbol,
            prediction=decision.action,
            reasoning=decision.reasoning[-1].content if decision.reasoning else '',
            result='PENDING',
            pnl_percent=0,
            lesson='',
            timestamp=datetime.now().isoformat()
        )
        
        self.memories.append(mem)
        
        # Kaydet
        mem_file = MEMORY_DIR / "predictions.json"
        with open(mem_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(m) for m in self.memories[-100:]], f, indent=2)
    
    # =========================================================================
    # TELEGRAM FORMATI
    # =========================================================================
    
    def format_telegram(self, decision: Decision) -> str:
        """Telegram icin formatla."""
        lines = ["DEMIR AI DUSUNUYOR...", ""]
        
        # Dusunce zinciri
        for thought in decision.reasoning:
            if thought.step == 'OBSERVE':
                lines.append(thought.content)
            elif thought.step == 'REMEMBER' and 'yok' not in thought.content.lower():
                lines.append("")
                lines.append(thought.content)
            elif thought.step == 'ANALYZE':
                lines.append("")
                lines.append(thought.content)
            elif thought.step == 'CONFLICT' and 'Celiski' in thought.content:
                lines.append("")
                lines.append(thought.content)
            elif thought.step == 'DECIDE':
                lines.append("")
                lines.append("---")
                lines.append(thought.content)
        
        return "\n".join(lines)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_brain: Optional[ThinkingBrain] = None

def get_thinking_brain() -> ThinkingBrain:
    """Get or create brain instance."""
    global _brain
    if _brain is None:
        _brain = ThinkingBrain()
    return _brain


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        brain = get_thinking_brain()
        decision = await brain.think('BTCUSDT')
        print(brain.format_telegram(decision))
    
    asyncio.run(test())
