# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - Early Signal Engine
===================================
ML-Enhanced Leading Indicator System.

Bu motor:
1. Leading indicators'ı hesaplar
2. Feature vector oluşturur
3. ML modeli ile tahmin yapar
4. Sinyal üretir

FARK: Eski sistem fiyat geçmişine bakardı.
      Bu sistem "büyük oyuncular ne yapıyor?" sorusuna bakar.
"""
import logging
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from src.v10.leading_indicators import (
    LeadingIndicators, 
    LeadingSignal, 
    SignalDirection,
    get_leading_indicators
)

logger = logging.getLogger("EARLY_SIGNAL_ENGINE")


@dataclass
class EarlySignal:
    """Erken sinyal sonucu"""
    symbol: str
    action: str                    # BUY, SELL, HOLD
    confidence: float              # 0-100
    entry_zone: Tuple[float, float]  # Min-Max giriş bölgesi
    stop_loss: float
    take_profit: float
    risk_reward: float
    leading_signal: LeadingSignal
    ml_prediction: Optional[Dict] = None
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': round(self.confidence, 1),
            'entry_zone': self.entry_zone,
            'stop_loss': round(self.stop_loss, 2),
            'take_profit': round(self.take_profit, 2),
            'risk_reward': round(self.risk_reward, 2),
            'reasoning': self.reasoning,
            'leading_indicators': self.leading_signal.to_dict(),
            'ml_prediction': self.ml_prediction,
            'timestamp': self.timestamp.isoformat()
        }


class FeatureCollector:
    """
    ML Feature Collection & Training Data
    
    Leading indicator'ları feature vector'e çevirir.
    Training için veri toplar.
    """
    
    FEATURE_VERSION = "v1.0"
    DATA_DIR = Path("src/v10/training_data")
    
    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._collection_buffer: List[Dict] = []
        self._buffer_size = 100
        
    def extract_features(self, signal: LeadingSignal, current_price: float) -> np.ndarray:
        """
        LeadingSignal'dan ML için feature vector çıkar.
        
        Returns:
            20 boyutlu feature vector
        """
        features = []
        
        # 1. Birleşik skor özellikleri (3)
        features.append(signal.strength / 100)  # Normalize
        features.append(signal.confidence / 100)
        features.append(1 if signal.direction.value.startswith("STRONG") else 0)
        
        # 2. Her indicator'ın değeri (5)
        indicator_values = {i.name: i.value for i in signal.indicators}
        for name in ['whale', 'orderbook', 'oi_divergence', 'funding', 'volume']:
            features.append(indicator_values.get(name, 0) / 100)
        
        # 3. Her indicator'ın confidence'ı (5)
        indicator_conf = {i.name: i.confidence for i in signal.indicators}
        for name in ['whale', 'orderbook', 'oi_divergence', 'funding', 'volume']:
            features.append(indicator_conf.get(name, 0) / 100)
        
        # 4. Indicator detayları (7)
        for ind in signal.indicators:
            if ind.name == 'whale':
                net_flow = ind.details.get('net_flow', 0)
                features.append(np.tanh(net_flow / 1000000))  # Normalize
            elif ind.name == 'orderbook':
                features.append(ind.details.get('imbalance_pct', 0) / 100)
            elif ind.name == 'oi_divergence':
                features.append(ind.details.get('oi_change_pct', 0) / 20)
                features.append(ind.details.get('price_change_pct', 0) / 10)
            elif ind.name == 'funding':
                features.append(ind.details.get('current_funding', 0) * 100)
            elif ind.name == 'volume':
                features.append(min(2, ind.details.get('volume_ratio', 1)) / 2)
                features.append(ind.details.get('buy_ratio', 0.5))
        
        # Pad to exactly 20 features
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def collect_training_sample(
        self, 
        signal: LeadingSignal, 
        current_price: float,
        future_price: Optional[float] = None
    ):
        """
        Training verisi topla.
        
        future_price verilirse (4 saat sonraki fiyat), label olarak kullanılır.
        """
        features = self.extract_features(signal, current_price)
        
        sample = {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal.symbol,
            'features': features.tolist(),
            'current_price': current_price,
            'future_price': future_price,
            'direction': signal.direction.value,
            'version': self.FEATURE_VERSION
        }
        
        self._collection_buffer.append(sample)
        
        # Buffer dolunca disk'e yaz
        if len(self._collection_buffer) >= self._buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Buffer'ı disk'e yaz"""
        if not self._collection_buffer:
            return
        
        filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.DATA_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(self._collection_buffer, f)
        
        logger.info(f"💾 Saved {len(self._collection_buffer)} training samples to {filename}")
        self._collection_buffer = []
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tüm training verisini yükle.
        
        Returns:
            X (features), y (labels)
        """
        X = []
        y = []
        
        for filepath in self.DATA_DIR.glob("training_*.json"):
            with open(filepath, 'r') as f:
                samples = json.load(f)
            
            for sample in samples:
                if sample.get('future_price') and sample.get('current_price'):
                    X.append(sample['features'])
                    # Label: Fiyat değişimi yönü
                    price_change = (sample['future_price'] - sample['current_price']) / sample['current_price']
                    if price_change > 0.01:  # >1% up
                        y.append(1)  # BUY
                    elif price_change < -0.01:  # >1% down
                        y.append(-1)  # SELL
                    else:
                        y.append(0)  # HOLD
        
        return np.array(X), np.array(y)


class EarlySignalEngine:
    """
    Early Signal Engine - ML Enhanced
    
    Ana motor. Leading indicator'ları alır, ML ile güçlendirir, sinyal üretir.
    """
    
    def __init__(self):
        self.leading_indicators: Optional[LeadingIndicators] = None
        self.feature_collector = FeatureCollector()
        self.ml_model = None  # Sonra yüklenecek
        self._last_signals: Dict[str, EarlySignal] = {}
        
        logger.info("🚀 Early Signal Engine initialized")
    
    async def initialize(self):
        """Async initialization"""
        self.leading_indicators = await get_leading_indicators()
    
    async def analyze(self, symbol: str = "BTCUSDT") -> EarlySignal:
        """
        Sembol için erken sinyal analizi yap.
        """
        if not self.leading_indicators:
            await self.initialize()
        
        logger.info(f"🔍 Early Signal Analysis: {symbol}")
        
        # 1. Leading indicators hesapla
        leading_signal = await self.leading_indicators.calculate_all(symbol)
        
        # 2. Güncel fiyatı al
        current_price = await self._get_current_price(symbol)
        
        # 3. Feature vector çıkar
        features = self.feature_collector.extract_features(leading_signal, current_price)
        
        # 4. ML prediction (eğer model varsa)
        ml_prediction = None
        if self.ml_model:
            ml_prediction = self._predict_with_ml(features)
        
        # 5. Sinyal üret
        signal = self._generate_signal(symbol, leading_signal, current_price, ml_prediction)
        
        # 6. Training için veri topla
        self.feature_collector.collect_training_sample(leading_signal, current_price)
        
        # Cache'e kaydet
        self._last_signals[symbol] = signal
        
        return signal
    
    async def _get_current_price(self, symbol: str) -> float:
        """Güncel fiyatı al"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return float(data['price'])
        except:
            pass
        
        return 0
    
    def _predict_with_ml(self, features: np.ndarray) -> Dict:
        """ML modeli ile tahmin (ileride implement edilecek)"""
        # TODO: LSTM/RL model entegrasyonu
        return {
            'prediction': 'NEUTRAL',
            'confidence': 0,
            'note': 'ML model not trained yet'
        }
    
    def _generate_signal(
        self, 
        symbol: str, 
        leading: LeadingSignal,
        current_price: float,
        ml_pred: Optional[Dict]
    ) -> EarlySignal:
        """
        Leading signals'dan trading sinyali üret.
        """
        # Yön belirleme
        if leading.direction in [SignalDirection.STRONG_BULLISH, SignalDirection.BULLISH]:
            action = "BUY"
        elif leading.direction in [SignalDirection.STRONG_BEARISH, SignalDirection.BEARISH]:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Confidence hesapla
        confidence = leading.confidence
        if leading.direction.value.startswith("STRONG"):
            confidence = min(100, confidence + 15)
        
        # Minimum confidence threshold
        if confidence < 50:
            action = "HOLD"
        
        # Entry zone hesapla (fiyatın ±0.5%)
        entry_min = current_price * 0.995
        entry_max = current_price * 1.005
        
        # SL/TP hesapla
        if action == "BUY":
            stop_loss = current_price * 0.98  # -2%
            take_profit = current_price * 1.04  # +4%
        elif action == "SELL":
            stop_loss = current_price * 1.02  # +2%
            take_profit = current_price * 0.96  # -4%
        else:
            stop_loss = current_price
            take_profit = current_price
        
        # Risk/Reward
        if action != "HOLD":
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0
        
        # Reasoning oluştur
        reasons = []
        for ind in leading.indicators:
            if abs(ind.value) > 15:
                emoji = "🟢" if ind.value > 0 else "🔴"
                reasons.append(f"{emoji} {ind.name}: {ind.value:+.0f}")
        
        reasoning = " | ".join(reasons) if reasons else "No strong signals"
        
        return EarlySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_zone=(round(entry_min, 2), round(entry_max, 2)),
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            leading_signal=leading,
            ml_prediction=ml_pred,
            reasoning=reasoning
        )
    
    def get_last_signal(self, symbol: str) -> Optional[EarlySignal]:
        """Son sinyali getir"""
        return self._last_signals.get(symbol)
    
    async def close(self):
        """Cleanup"""
        if self.leading_indicators:
            await self.leading_indicators.close()
        self.feature_collector._flush_buffer()


# Global instance
_engine: Optional[EarlySignalEngine] = None


async def get_early_signal_engine() -> EarlySignalEngine:
    """Get or create Early Signal Engine"""
    global _engine
    if _engine is None:
        _engine = EarlySignalEngine()
        await _engine.initialize()
    return _engine
