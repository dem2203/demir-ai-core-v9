# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - PREDICTOR ENGINE
================================
Piyasa verilerini analiz edip HAREKET ÖNCE tahmin üretir.

ÖZELLİKLER:
- Support/Resistance tespiti
- Pattern recognition (breakout, reversal)
- Multi-timeframe confluence
- Minimum $500 potansiyel filtresi
- Entry/TP/SL hesaplama

ÇIKTI: TradingSignal (entry, tp1, tp2, tp3, sl, confidence, reasoning)
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("PREDICTOR")


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class TradingSignal:
    """Tahmin edilen trading sinyali"""
    symbol: str
    signal_type: SignalType
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Entry/Exit Levels
    entry_low: float = 0        # Giriş aralığı alt
    entry_high: float = 0       # Giriş aralığı üst
    tp1: float = 0              # Take Profit 1
    tp2: float = 0              # Take Profit 2
    tp3: float = 0              # Take Profit 3 (aggressive)
    sl: float = 0               # Stop Loss
    
    # Metrics
    risk_reward: float = 0      # R/R oranı
    confidence: float = 0       # 0-100
    risk_level: RiskLevel = RiskLevel.MEDIUM
    potential_usd: float = 0    # Potansiyel hareket $
    
    # Reasoning
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Data quality
    data_sources_ok: int = 0
    data_sources_total: int = 7
    
    @property
    def is_valid(self) -> bool:
        """Sinyal geçerli mi? (min $500 potansiyel)"""
        return (
            self.signal_type != SignalType.WAIT and 
            self.confidence >= 60 and 
            self.potential_usd >= 500 and
            self.risk_reward >= 1.5
        )


class PredictorEngine:
    """
    DEMIR AI v10 - Tahmin Motoru
    
    Piyasa verilerini alıp hareket ÖNCE sinyal üretir.
    Entry, TP1/2/3, SL seviyeleri hesaplar.
    """
    
    # Minimum hareket filtreleri (USD)
    MIN_POTENTIAL = {
        'BTCUSDT': 500,
        'ETHUSDT': 50,
        'SOLUSDT': 3,
        'LTCUSDT': 5
    }
    
    def __init__(self):
        logger.info("🧠 Predictor Engine initialized")
    
    def generate_signal(self, snapshot) -> TradingSignal:
        """
        Market snapshot'ından trading sinyali üret.
        
        Args:
            snapshot: MarketSnapshot from DataHub
        
        Returns:
            TradingSignal with entry/TP/SL
        """
        signal = TradingSignal(
            symbol=snapshot.symbol,
            data_sources_ok=7 - len(snapshot.errors) if hasattr(snapshot, 'errors') else 7
        )
        
        if not snapshot.is_valid:
            signal.signal_type = SignalType.WAIT
            signal.warnings.append("Veri yetersiz veya hatalı")
            return signal
        
        # Analiz skorları topla
        bullish_score = 0
        bearish_score = 0
        
        # 1. TREND ANALİZİ (ağırlık: 2x)
        trend_result = self._analyze_trend(snapshot)
        if trend_result['direction'] == 'BULLISH':
            bullish_score += 2 * trend_result['strength']
            signal.reasons.append(f"📈 Trend: YUKARI ({trend_result['reason']})")
        elif trend_result['direction'] == 'BEARISH':
            bearish_score += 2 * trend_result['strength']
            signal.reasons.append(f"📉 Trend: AŞAĞI ({trend_result['reason']})")
        
        # 2. RSI ANALİZİ (ağırlık: 1.5x)
        rsi_result = self._analyze_rsi(snapshot)
        if rsi_result['signal'] == 'BUY':
            bullish_score += 1.5 * rsi_result['strength']
            signal.reasons.append(f"🔥 RSI: {snapshot.rsi_1h:.0f} ({rsi_result['reason']})")
        elif rsi_result['signal'] == 'SELL':
            bearish_score += 1.5 * rsi_result['strength']
            signal.reasons.append(f"⚠️ RSI: {snapshot.rsi_1h:.0f} ({rsi_result['reason']})")
        
        # 3. ORDER BOOK ANALİZİ (ağırlık: 2x)
        ob_result = self._analyze_orderbook(snapshot)
        if ob_result['signal'] == 'BUY':
            bullish_score += 2 * ob_result['strength']
            signal.reasons.append(f"📗 Order Book: {snapshot.bid_ask_ratio:.2f}x BID ({ob_result['reason']})")
        elif ob_result['signal'] == 'SELL':
            bearish_score += 2 * ob_result['strength']
            signal.reasons.append(f"📕 Order Book: {1/snapshot.bid_ask_ratio:.2f}x ASK ({ob_result['reason']})")
        
        # 4. FUNDING RATE ANALİZİ (ağırlık: 1.5x)
        funding_result = self._analyze_funding(snapshot)
        if funding_result['signal'] == 'BUY':
            bullish_score += 1.5 * funding_result['strength']
            signal.reasons.append(f"💰 Funding: {snapshot.funding_rate:.4f}% ({funding_result['reason']})")
        elif funding_result['signal'] == 'SELL':
            bearish_score += 1.5 * funding_result['strength']
            signal.reasons.append(f"⚠️ Funding: {snapshot.funding_rate:.4f}% ({funding_result['reason']})")
        
        # 5. WHALE AKTİVİTESİ (ağırlık: 2x)
        whale_result = self._analyze_whales(snapshot)
        if whale_result['signal'] == 'BUY':
            bullish_score += 2 * whale_result['strength']
            signal.reasons.append(f"🐋 Whale: Net +{snapshot.whale_net_flow} ({whale_result['reason']})")
        elif whale_result['signal'] == 'SELL':
            bearish_score += 2 * whale_result['strength']
            signal.reasons.append(f"🐋 Whale: Net {snapshot.whale_net_flow} ({whale_result['reason']})")
        
        # 6. SUPPORT/RESISTANCE
        sr_result = self._analyze_support_resistance(snapshot)
        if sr_result['near_support']:
            bullish_score += 1
            signal.reasons.append(f"🟢 Güçlü destek yakın: ${sr_result['support']:,.0f}")
        if sr_result['near_resistance']:
            bearish_score += 1
            signal.warnings.append(f"🔴 Direnç yakın: ${sr_result['resistance']:,.0f}")
        
        # 7. LONG/SHORT RATIO (contrarian)
        ls_result = self._analyze_long_short(snapshot)
        if ls_result['signal'] == 'BUY':
            bullish_score += 1 * ls_result['strength']
            signal.reasons.append(f"⚖️ L/S Ratio: {snapshot.long_ratio/snapshot.short_ratio:.2f} ({ls_result['reason']})")
        elif ls_result['signal'] == 'SELL':
            bearish_score += 1 * ls_result['strength']
            signal.reasons.append(f"⚖️ L/S Ratio: {snapshot.long_ratio/snapshot.short_ratio:.2f} ({ls_result['reason']})")
        
        # KARAR VER
        total_score = bullish_score + bearish_score
        
        if bullish_score > bearish_score * 1.5 and bullish_score >= 4:
            signal.signal_type = SignalType.LONG
            signal.confidence = min(95, 50 + (bullish_score - bearish_score) * 5)
        elif bearish_score > bullish_score * 1.5 and bearish_score >= 4:
            signal.signal_type = SignalType.SHORT
            signal.confidence = min(95, 50 + (bearish_score - bullish_score) * 5)
        else:
            signal.signal_type = SignalType.WAIT
            signal.confidence = 40
            signal.warnings.append("Net yön yok - bekle")
            return signal
        
        # ENTRY/TP/SL HESAPLA
        self._calculate_levels(signal, snapshot, sr_result)
        
        # Potansiyel hareket hesapla
        if signal.signal_type == SignalType.LONG:
            signal.potential_usd = signal.tp2 - signal.entry_high
        else:
            signal.potential_usd = signal.entry_low - signal.tp2
        
        # Minimum potansiyel filtresi
        min_potential = self.MIN_POTENTIAL.get(snapshot.symbol, 500)
        if signal.potential_usd < min_potential:
            signal.signal_type = SignalType.WAIT
            signal.warnings.append(f"Potansiyel çok düşük: ${signal.potential_usd:.0f} < ${min_potential}")
        
        # Risk seviyesi
        if signal.risk_reward >= 3:
            signal.risk_level = RiskLevel.LOW
        elif signal.risk_reward >= 2:
            signal.risk_level = RiskLevel.MEDIUM
        else:
            signal.risk_level = RiskLevel.HIGH
        
        return signal
    
    # =========================================
    # ANALİZ FONKSİYONLARI
    # =========================================
    
    def _analyze_trend(self, snapshot) -> Dict:
        """EMA bazlı trend analizi"""
        if snapshot.ema_20 == 0 or snapshot.ema_50 == 0:
            return {'direction': 'NEUTRAL', 'strength': 0, 'reason': 'Veri yok'}
        
        price = snapshot.price
        
        if price > snapshot.ema_20 > snapshot.ema_50:
            strength = min(1.0, (price - snapshot.ema_50) / snapshot.ema_50 * 50)
            return {'direction': 'BULLISH', 'strength': strength, 'reason': 'EMA20 > EMA50, fiyat üstünde'}
        elif price < snapshot.ema_20 < snapshot.ema_50:
            strength = min(1.0, (snapshot.ema_50 - price) / price * 50)
            return {'direction': 'BEARISH', 'strength': strength, 'reason': 'EMA20 < EMA50, fiyat altında'}
        else:
            return {'direction': 'NEUTRAL', 'strength': 0.3, 'reason': 'Kararsız'}
    
    def _analyze_rsi(self, snapshot) -> Dict:
        """RSI bazlı aşırı alım/satım analizi"""
        rsi = snapshot.rsi_1h
        
        if rsi < 30:
            strength = (30 - rsi) / 30
            return {'signal': 'BUY', 'strength': strength, 'reason': 'Aşırı satım'}
        elif rsi < 40:
            return {'signal': 'BUY', 'strength': 0.5, 'reason': 'Düşük RSI'}
        elif rsi > 70:
            strength = (rsi - 70) / 30
            return {'signal': 'SELL', 'strength': strength, 'reason': 'Aşırı alım'}
        elif rsi > 60:
            return {'signal': 'SELL', 'strength': 0.5, 'reason': 'Yüksek RSI'}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Normal'}
    
    def _analyze_orderbook(self, snapshot) -> Dict:
        """Order book imbalance analizi"""
        ratio = snapshot.bid_ask_ratio
        
        if ratio > 2.0:
            return {'signal': 'BUY', 'strength': 1.0, 'reason': 'Güçlü alıcı duvarı'}
        elif ratio > 1.5:
            return {'signal': 'BUY', 'strength': 0.7, 'reason': 'Alıcı baskısı'}
        elif ratio < 0.5:
            return {'signal': 'SELL', 'strength': 1.0, 'reason': 'Güçlü satıcı duvarı'}
        elif ratio < 0.67:
            return {'signal': 'SELL', 'strength': 0.7, 'reason': 'Satıcı baskısı'}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Dengeli'}
    
    def _analyze_funding(self, snapshot) -> Dict:
        """Funding rate analizi (contrarian)"""
        rate = snapshot.funding_rate
        
        if rate < -0.02:
            return {'signal': 'BUY', 'strength': 1.0, 'reason': 'Negatif funding - short squeeze potansiyeli'}
        elif rate < 0:
            return {'signal': 'BUY', 'strength': 0.5, 'reason': 'Hafif negatif funding'}
        elif rate > 0.05:
            return {'signal': 'SELL', 'strength': 1.0, 'reason': 'Yüksek funding - long squeeze potansiyeli'}
        elif rate > 0.02:
            return {'signal': 'SELL', 'strength': 0.5, 'reason': 'Yüksek funding'}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Normal funding'}
    
    def _analyze_whales(self, snapshot) -> Dict:
        """Whale aktivitesi analizi"""
        net = snapshot.whale_net_flow
        
        if net >= 3:
            return {'signal': 'BUY', 'strength': 1.0, 'reason': 'Güçlü whale alımı'}
        elif net >= 1:
            return {'signal': 'BUY', 'strength': 0.6, 'reason': 'Whale alımı'}
        elif net <= -3:
            return {'signal': 'SELL', 'strength': 1.0, 'reason': 'Güçlü whale satımı'}
        elif net <= -1:
            return {'signal': 'SELL', 'strength': 0.6, 'reason': 'Whale satımı'}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Whale aktivitesi düşük'}
    
    def _analyze_support_resistance(self, snapshot) -> Dict:
        """Order book'tan S/R seviyeleri"""
        current = snapshot.price
        
        # Strong bids = support
        support = min(snapshot.strong_bids) if snapshot.strong_bids else current * 0.97
        # Strong asks = resistance
        resistance = max(snapshot.strong_asks) if snapshot.strong_asks else current * 1.03
        
        # Yakınlık kontrolü
        near_support = (current - support) / current < 0.01  # %1'den yakın
        near_resistance = (resistance - current) / current < 0.01
        
        return {
            'support': support,
            'resistance': resistance,
            'near_support': near_support,
            'near_resistance': near_resistance
        }
    
    def _analyze_long_short(self, snapshot) -> Dict:
        """Long/Short ratio analizi (contrarian)"""
        ratio = snapshot.long_ratio / snapshot.short_ratio if snapshot.short_ratio > 0 else 1.0
        
        if ratio > 2.0:
            return {'signal': 'SELL', 'strength': 0.8, 'reason': 'Aşırı long kalabalık - düzeltme riski'}
        elif ratio > 1.5:
            return {'signal': 'SELL', 'strength': 0.5, 'reason': 'Long kalabalık'}
        elif ratio < 0.5:
            return {'signal': 'BUY', 'strength': 0.8, 'reason': 'Aşırı short kalabalık - squeeze riski'}
        elif ratio < 0.67:
            return {'signal': 'BUY', 'strength': 0.5, 'reason': 'Short kalabalık'}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Dengeli'}
    
    # =========================================
    # SEVİYE HESAPLAMA
    # =========================================
    
    def _calculate_levels(self, signal: TradingSignal, snapshot, sr_result: Dict):
        """Entry, TP, SL seviyelerini hesapla"""
        price = snapshot.price
        
        # ATR tahmini (son 24h range'in %30'u)
        atr = (snapshot.high_24h - snapshot.low_24h) * 0.3
        
        if signal.signal_type == SignalType.LONG:
            # LONG seviyeleri
            signal.entry_low = price * 0.998  # %0.2 altı
            signal.entry_high = price * 1.002  # %0.2 üstü
            
            # TP seviyeleri
            if sr_result['resistance']:
                signal.tp1 = sr_result['resistance']
            else:
                signal.tp1 = price * 1.015  # %1.5
            signal.tp2 = signal.tp1 * 1.015  # TP1'in %1.5 üstü
            signal.tp3 = signal.tp1 * 1.035  # Agresif: %3.5
            
            # SL
            if sr_result['support']:
                signal.sl = sr_result['support'] * 0.995  # Destek altı
            else:
                signal.sl = price * 0.985  # %1.5 altı
            
        else:  # SHORT
            signal.entry_low = price * 0.998
            signal.entry_high = price * 1.002
            
            # TP seviyeleri (aşağı)
            if sr_result['support']:
                signal.tp1 = sr_result['support']
            else:
                signal.tp1 = price * 0.985
            signal.tp2 = signal.tp1 * 0.985
            signal.tp3 = signal.tp1 * 0.965
            
            # SL
            if sr_result['resistance']:
                signal.sl = sr_result['resistance'] * 1.005
            else:
                signal.sl = price * 1.015
        
        # Risk/Reward hesapla
        if signal.signal_type == SignalType.LONG:
            reward = signal.tp2 - signal.entry_high
            risk = signal.entry_high - signal.sl
        else:
            reward = signal.entry_low - signal.tp2
            risk = signal.sl - signal.entry_low
        
        signal.risk_reward = reward / risk if risk > 0 else 0


# Singleton
_predictor: Optional[PredictorEngine] = None

def get_predictor() -> PredictorEngine:
    global _predictor
    if _predictor is None:
        _predictor = PredictorEngine()
    return _predictor
