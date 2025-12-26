# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - ENHANCED PREDICTOR ENGINE
=========================================
Geliştirilmiş tahmin motoru - tüm yeni modülleri kullanır:
- 7 Gelişmiş Teknik Gösterge
- 5 Timeframe Confluence
- LSTM Fiyat Tahmini
- Performans Takibi

ÇIKTI: TradingSignal with enhanced confidence and reasoning
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.v10.advanced_indicators import get_advanced_indicators, IndicatorSignal
from src.v10.multi_timeframe import get_mtf_analyzer
from src.v10.lstm_predictor import get_lstm_predictor

logger = logging.getLogger("ENHANCED_PREDICTOR")


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
    """Enhanced trading signal with more data"""
    symbol: str
    signal_type: SignalType = SignalType.WAIT
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Entry/Exit Levels
    entry_low: float = 0
    entry_high: float = 0
    tp1: float = 0
    tp2: float = 0
    tp3: float = 0
    sl: float = 0
    
    # Metrics
    risk_reward: float = 0
    confidence: float = 0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    potential_usd: float = 0
    
    # Enhanced reasoning
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # NEW: Advanced analysis results
    advanced_signals: List[str] = field(default_factory=list)
    mtf_confluence: str = ""
    lstm_prediction: str = ""
    
    # Data quality
    data_sources_ok: int = 0
    data_sources_total: int = 10  # increased from 7
    
    @property
    def is_valid(self) -> bool:
        return (
            self.signal_type != SignalType.WAIT and 
            self.confidence >= 60 and 
            self.potential_usd >= 500 and
            self.risk_reward >= 1.5
        )


class EnhancedPredictor:
    """
    Geliştirilmiş Tahmin Motoru
    
    Tüm yeni modülleri kullanarak daha güçlü tahmin yapar:
    1. Temel analiz (7 faktör)
    2. Gelişmiş göstergeler (7 ek gösterge)
    3. Multi-timeframe confluence (5 TF)
    4. LSTM fiyat tahmini
    """
    
    MIN_POTENTIAL = {
        'BTCUSDT': 500,
        'ETHUSDT': 50,
        'SOLUSDT': 3,
        'LTCUSDT': 5
    }
    
    def __init__(self):
        self.advanced_indicators = get_advanced_indicators()
        self.mtf_analyzer = get_mtf_analyzer()
        self.lstm_predictor = get_lstm_predictor()
        
        logger.info("🧠 Enhanced Predictor initialized with all modules")
    
    async def generate_signal_async(self, snapshot, klines: List = None) -> TradingSignal:
        """
        Async signal generation with all enhanced analysis.
        """
        signal = TradingSignal(
            symbol=snapshot.symbol,
            data_sources_ok=7 - len(snapshot.errors) if hasattr(snapshot, 'errors') else 7
        )
        
        if not snapshot.is_valid:
            signal.signal_type = SignalType.WAIT
            signal.warnings.append("Veri yetersiz veya hatalı")
            return signal
        
        # ========================================
        # PHASE 1: TEMEL ANALİZ (Mevcut 7 faktör)
        # ========================================
        bullish_score = 0
        bearish_score = 0
        
        # 1. Trend
        trend_result = self._analyze_trend(snapshot)
        if trend_result['direction'] == 'BULLISH':
            bullish_score += 2 * trend_result['strength']
            signal.reasons.append(f"📈 Trend: YUKARI ({trend_result['reason']})")
        elif trend_result['direction'] == 'BEARISH':
            bearish_score += 2 * trend_result['strength']
            signal.reasons.append(f"📉 Trend: AŞAĞI ({trend_result['reason']})")
        
        # 2. RSI
        rsi_result = self._analyze_rsi(snapshot)
        if rsi_result['signal'] == 'BUY':
            bullish_score += 1.5 * rsi_result['strength']
            signal.reasons.append(f"🔥 RSI: {snapshot.rsi_1h:.0f} ({rsi_result['reason']})")
        elif rsi_result['signal'] == 'SELL':
            bearish_score += 1.5 * rsi_result['strength']
            signal.reasons.append(f"⚠️ RSI: {snapshot.rsi_1h:.0f} ({rsi_result['reason']})")
        
        # 3. Order Book
        ob_result = self._analyze_orderbook(snapshot)
        if ob_result['signal'] == 'BUY' and snapshot.bid_ask_ratio >= 0:
            bullish_score += 2 * ob_result['strength']
            signal.reasons.append(f"📗 Order Book: {snapshot.bid_ask_ratio:.2f}x BID")
        elif ob_result['signal'] == 'SELL' and snapshot.bid_ask_ratio >= 0:
            bearish_score += 2 * ob_result['strength']
            signal.reasons.append(f"📕 Order Book: Satıcı baskısı")
        
        # 4. Funding
        funding_result = self._analyze_funding(snapshot)
        if funding_result['signal'] == 'BUY':
            bullish_score += 1.5 * funding_result['strength']
            signal.reasons.append(f"💰 Funding: {snapshot.funding_rate:.4f}% ({funding_result['reason']})")
        elif funding_result['signal'] == 'SELL':
            bearish_score += 1.5 * funding_result['strength']
        
        # 5. Whale
        whale_result = self._analyze_whales(snapshot)
        if whale_result['signal'] == 'BUY':
            bullish_score += 2 * whale_result['strength']
            signal.reasons.append(f"🐋 Whale: Net +{snapshot.whale_net_flow:.0f}")
        elif whale_result['signal'] == 'SELL':
            bearish_score += 2 * whale_result['strength']
        
        # ========================================
        # PHASE 2: GELİŞMİŞ GÖSTERGELER (7 yeni)
        # ========================================
        if klines and len(klines) >= 50:
            try:
                adv_result = self.advanced_indicators.analyze_all(klines, snapshot.price)
                
                for adv_signal in adv_result.get('signals', []):
                    if adv_signal.signal == "BUY" and adv_signal.strength > 0.5:
                        bullish_score += adv_signal.strength
                        signal.advanced_signals.append(f"✅ {adv_signal.name}: {adv_signal.description}")
                    elif adv_signal.signal == "SELL" and adv_signal.strength > 0.5:
                        bearish_score += adv_signal.strength
                        signal.advanced_signals.append(f"❌ {adv_signal.name}: {adv_signal.description}")
                
                signal.data_sources_ok += 1
                
            except Exception as e:
                logger.warning(f"Advanced indicators error: {e}")
        
        # ========================================
        # PHASE 3: MULTI-TIMEFRAME CONFLUENCE
        # ========================================
        try:
            mtf_result = await self.mtf_analyzer.analyze(snapshot.symbol)
            
            if mtf_result.is_valid:
                if mtf_result.confluence_direction == "BULLISH":
                    bullish_score += mtf_result.confluence_score * 0.5
                    signal.mtf_confluence = f"🎯 MTF: {mtf_result.confluence_score}/5 TF BULLISH"
                elif mtf_result.confluence_direction == "BEARISH":
                    bearish_score += mtf_result.confluence_score * 0.5
                    signal.mtf_confluence = f"🎯 MTF: {mtf_result.confluence_score}/5 TF BEARISH"
                else:
                    signal.mtf_confluence = "MTF: Karışık sinyaller"
                
                signal.data_sources_ok += 1
                
        except Exception as e:
            logger.warning(f"MTF analysis error: {e}")
        
        # ========================================
        # PHASE 4: LSTM FİYAT TAHMİNİ
        # ========================================
        try:
            prediction = await self.lstm_predictor.predict(snapshot.symbol)
            
            if prediction:
                if prediction.direction == "UP":
                    bullish_score += prediction.confidence / 50  # Max +2
                    signal.lstm_prediction = f"🧠 LSTM: {prediction.predicted_change_pct:+.2f}% (4h, güven: {prediction.confidence:.0f}%)"
                elif prediction.direction == "DOWN":
                    bearish_score += prediction.confidence / 50
                    signal.lstm_prediction = f"🧠 LSTM: {prediction.predicted_change_pct:+.2f}% (4h, güven: {prediction.confidence:.0f}%)"
                else:
                    signal.lstm_prediction = "LSTM: Flat tahmin"
                
                signal.data_sources_ok += 1
                
        except Exception as e:
            logger.warning(f"LSTM prediction error: {e}")
        
        # ========================================
        # KARAR VER
        # ========================================
        total_score = bullish_score + bearish_score
        
        # Daha yüksek eşik - confluence gerekli
        if bullish_score > bearish_score * 1.5 and bullish_score >= 6:
            signal.signal_type = SignalType.LONG
            signal.confidence = min(95, 50 + (bullish_score - bearish_score) * 4)
        elif bearish_score > bullish_score * 1.5 and bearish_score >= 6:
            signal.signal_type = SignalType.SHORT
            signal.confidence = min(95, 50 + (bearish_score - bullish_score) * 4)
        else:
            signal.signal_type = SignalType.WAIT
            signal.confidence = 40
            signal.warnings.append("Net yön yok veya skor yetersiz")
            return signal
        
        # Entry/TP/SL hesapla
        self._calculate_levels(signal, snapshot)
        
        # Potansiyel
        if signal.signal_type == SignalType.LONG:
            signal.potential_usd = signal.tp2 - signal.entry_high
        else:
            signal.potential_usd = signal.entry_low - signal.tp2
        
        # Minimum filtre
        min_potential = self.MIN_POTENTIAL.get(snapshot.symbol, 500)
        if signal.potential_usd < min_potential:
            signal.signal_type = SignalType.WAIT
            signal.warnings.append(f"Potansiyel düşük: ${signal.potential_usd:.0f} < ${min_potential}")
        
        # Risk seviyesi
        if signal.risk_reward >= 3:
            signal.risk_level = RiskLevel.LOW
        elif signal.risk_reward >= 2:
            signal.risk_level = RiskLevel.MEDIUM
        else:
            signal.risk_level = RiskLevel.HIGH
        
        return signal
    
    def generate_signal(self, snapshot) -> TradingSignal:
        """Sync wrapper for async generation"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.generate_signal_async(snapshot))
                    return future.result()
            else:
                return asyncio.run(self.generate_signal_async(snapshot))
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return TradingSignal(symbol=snapshot.symbol)
    
    # =========================================
    # TEMEL ANALİZ FONKSİYONLARI
    # =========================================
    
    def _analyze_trend(self, snapshot) -> Dict:
        if snapshot.ema_20 == 0 or snapshot.ema_50 == 0:
            return {'direction': 'NEUTRAL', 'strength': 0, 'reason': 'Veri yok'}
        
        price = snapshot.price
        
        if price > snapshot.ema_20 > snapshot.ema_50:
            strength = min(1.0, (price - snapshot.ema_50) / snapshot.ema_50 * 50)
            return {'direction': 'BULLISH', 'strength': strength, 'reason': 'EMA20 > EMA50'}
        elif price < snapshot.ema_20 < snapshot.ema_50:
            strength = min(1.0, (snapshot.ema_50 - price) / price * 50)
            return {'direction': 'BEARISH', 'strength': strength, 'reason': 'EMA20 < EMA50'}
        else:
            return {'direction': 'NEUTRAL', 'strength': 0.3, 'reason': 'Kararsız'}
    
    def _analyze_rsi(self, snapshot) -> Dict:
        rsi = snapshot.rsi_1h
        if rsi < 0:  # Veri yok
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Veri yok'}
        
        if rsi < 30:
            return {'signal': 'BUY', 'strength': (30 - rsi) / 30, 'reason': 'Aşırı satım'}
        elif rsi < 40:
            return {'signal': 'BUY', 'strength': 0.5, 'reason': 'Düşük RSI'}
        elif rsi > 70:
            return {'signal': 'SELL', 'strength': (rsi - 70) / 30, 'reason': 'Aşırı alım'}
        elif rsi > 60:
            return {'signal': 'SELL', 'strength': 0.5, 'reason': 'Yüksek RSI'}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Normal'}
    
    def _analyze_orderbook(self, snapshot) -> Dict:
        ratio = snapshot.bid_ask_ratio
        if ratio < 0:  # Veri yok
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Veri yok'}
        
        if ratio > 2.0:
            return {'signal': 'BUY', 'strength': 1.0, 'reason': 'Güçlü alıcı'}
        elif ratio > 1.5:
            return {'signal': 'BUY', 'strength': 0.7, 'reason': 'Alıcı baskısı'}
        elif ratio < 0.5:
            return {'signal': 'SELL', 'strength': 1.0, 'reason': 'Güçlü satıcı'}
        elif ratio < 0.67:
            return {'signal': 'SELL', 'strength': 0.7, 'reason': 'Satıcı baskısı'}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Dengeli'}
    
    def _analyze_funding(self, snapshot) -> Dict:
        rate = snapshot.funding_rate
        if rate <= -900:  # Veri yok (-999)
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Veri yok'}
        
        if rate < -0.02:
            return {'signal': 'BUY', 'strength': 1.0, 'reason': 'Short squeeze potansiyeli'}
        elif rate < 0:
            return {'signal': 'BUY', 'strength': 0.5, 'reason': 'Negatif funding'}
        elif rate > 0.05:
            return {'signal': 'SELL', 'strength': 1.0, 'reason': 'Long squeeze riski'}
        elif rate > 0.02:
            return {'signal': 'SELL', 'strength': 0.5, 'reason': 'Yüksek funding'}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Normal'}
    
    def _analyze_whales(self, snapshot) -> Dict:
        net = snapshot.whale_net_flow
        if net <= -900:  # Veri yok (-999)
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Veri yok'}
        
        if net >= 3:
            return {'signal': 'BUY', 'strength': 1.0, 'reason': 'Güçlü whale alımı'}
        elif net >= 1:
            return {'signal': 'BUY', 'strength': 0.6, 'reason': 'Whale alımı'}
        elif net <= -3:
            return {'signal': 'SELL', 'strength': 1.0, 'reason': 'Güçlü whale satımı'}
        elif net <= -1:
            return {'signal': 'SELL', 'strength': 0.6, 'reason': 'Whale satımı'}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'Aktivite düşük'}
    
    def _calculate_levels(self, signal: TradingSignal, snapshot):
        """Calculate entry, TP, SL levels"""
        price = snapshot.price
        
        # Use new S/R from snapshot if available
        support = snapshot.support if snapshot.support > 0 else price * 0.97
        resistance = snapshot.resistance if snapshot.resistance > 0 else price * 1.03
        
        if signal.signal_type == SignalType.LONG:
            signal.entry_low = price * 0.998
            signal.entry_high = price * 1.002
            signal.tp1 = resistance
            signal.tp2 = resistance * 1.015
            signal.tp3 = resistance * 1.035
            signal.sl = support * 0.995
        else:
            signal.entry_low = price * 0.998
            signal.entry_high = price * 1.002
            signal.tp1 = support
            signal.tp2 = support * 0.985
            signal.tp3 = support * 0.965
            signal.sl = resistance * 1.005
        
        # R/R
        if signal.signal_type == SignalType.LONG:
            reward = signal.tp2 - signal.entry_high
            risk = signal.entry_high - signal.sl
        else:
            reward = signal.entry_low - signal.tp2
            risk = signal.sl - signal.entry_low
        
        signal.risk_reward = reward / risk if risk > 0 else 0


# Singleton
_predictor: Optional[EnhancedPredictor] = None

def get_enhanced_predictor() -> EnhancedPredictor:
    global _predictor
    if _predictor is None:
        _predictor = EnhancedPredictor()
    return _predictor
