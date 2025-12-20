# -*- coding: utf-8 -*-
"""
DEMIR AI - UNIFIED BRAIN
=========================
Tek merkezi yapay zeka karar sistemi.

Bu dosya eskiden 3 ayrı sistemde dağılmış olan tüm "beyin" fonksiyonlarını birleştirir:
- SignalOrchestrator (1392 satır) → collect_signals()
- LivingAIBrain (734 satır) → think()
- AIReasoningEngine (990 satır) → generate_reasoning()

PRENSIP: Basitlik > Karmaşıklık
- Bir sinyal sistemi
- Gerçek tahmin (LSTM + Pattern)
- Anlaşılır çıktı

Author: DEMIR AI Core Team
Date: 2024-12
"""
import logging
import numpy as np
import pandas as pd
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import os
import json

logger = logging.getLogger("UNIFIED_BRAIN")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Signal:
    """Trading sinyali."""
    symbol: str
    direction: str  # 'LONG', 'SHORT', 'HOLD'
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    sources: List[str]  # Hangi modüller bu sinyali destekliyor
    reasoning: str  # Türkçe açıklama
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketContext:
    """Piyasa durumu özeti."""
    price: float
    change_1h: float
    change_24h: float
    volume_ratio: float  # Current vs average
    volatility: float
    trend: str  # 'BULL', 'BEAR', 'RANGE'
    regime: str  # 'TRENDING', 'RANGING', 'VOLATILE'
    fear_greed: int
    funding_rate: float
    long_short_ratio: float
    btc_dominance: float


# =============================================================================
# UNIFIED BRAIN
# =============================================================================

class UnifiedBrain:
    """
    DEMIR AI - Birleşik Yapay Zeka Beyni
    
    Tüm karar verme bu sınıftan geçer.
    
    Modüller:
    1. DataCollector - Tüm verileri toplar
    2. ContextBuilder - Piyasa bağlamı oluşturur
    3. Predictor - LSTM ve pattern tahminleri
    4. SignalGenerator - Sinyalleri sentezler
    5. RiskAnalyzer - Risk değerlendirmesi
    """
    
    # Minimum gereksinimler
    MIN_CONFIDENCE = 65  # %65 altı sinyal yok
    MIN_RISK_REWARD = 1.5  # R:R < 1.5 sinyal yok
    SIGNAL_COOLDOWN = 3600  # 1 saat (aynı coin için)
    
    def __init__(self):
        self.last_signals: Dict[str, datetime] = {}  # symbol -> last signal time
        self.performance_stats = {
            'total_signals': 0,
            'successful': 0,
            'accuracy': 0.0
        }
        
        # LSTM models (lazy load)
        self.lstm_models = {}
        self.scalers = {}
        
        logger.info("🧠 Unified Brain initialized")
    
    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    
    async def analyze(self, symbol: str = 'BTCUSDT') -> Optional[Signal]:
        """
        ANA ANALİZ FONKSİYONU
        Tüm modülleri çalıştırır ve sinyal üretir.
        
        Returns:
            Signal or None (eğer sinyal yoksa)
        """
        # 1. Cooldown kontrolü
        if not self._can_signal(symbol):
            logger.debug(f"⏳ {symbol} cooldown'da")
            return None
        
        # 2. Veri topla
        context = await self._collect_data(symbol)
        if not context:
            logger.warning(f"❌ {symbol} veri toplanamadı")
            return None
        
        # 3. Tahminleri al
        predictions = await self._get_predictions(symbol, context)
        
        # 4. Sinyal sentezle
        signal = self._synthesize_signal(symbol, context, predictions)
        
        # 5. Kalite kontrolü
        if signal and signal.confidence >= self.MIN_CONFIDENCE:
            self.last_signals[symbol] = datetime.now()
            self.performance_stats['total_signals'] += 1
            logger.info(f"🎯 SIGNAL: {symbol} {signal.direction} %{signal.confidence:.0f}")
            return signal
        
        return None
    
    # =========================================================================
    # DATA COLLECTION
    # =========================================================================
    
    async def _collect_data(self, symbol: str) -> Optional[MarketContext]:
        """Tüm piyasa verilerini topla."""
        try:
            # 1. Binance fiyat verileri
            price_data = await self._get_price_data(symbol)
            if not price_data:
                return None
            
            # 2. Derivatives verileri
            derivatives = await self._get_derivatives_data(symbol)
            
            # 3. Sentiment verileri
            sentiment = await self._get_sentiment_data()
            
            # 4. Volatilite hesapla
            volatility = self._calculate_volatility(price_data)
            
            # 5. Trend tespit et
            trend = self._detect_trend(price_data)
            
            # 6. Regime tespit et
            regime = self._detect_regime(volatility, trend)
            
            return MarketContext(
                price=price_data['current_price'],
                change_1h=price_data.get('change_1h', 0),
                change_24h=price_data.get('change_24h', 0),
                volume_ratio=price_data.get('volume_ratio', 1.0),
                volatility=volatility,
                trend=trend,
                regime=regime,
                fear_greed=sentiment.get('fear_greed_index', 50),
                funding_rate=derivatives.get('funding_rate', 0),
                long_short_ratio=derivatives.get('long_short_ratio', 1.0),
                btc_dominance=sentiment.get('btc_dominance', 50)
            )
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return None
    
    async def _get_price_data(self, symbol: str) -> Optional[Dict]:
        """Binance'dan fiyat verileri al."""
        try:
            # 24hr ticker
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={'symbol': symbol},
                timeout=10
            )
            if resp.status_code != 200:
                return None
            
            ticker = resp.json()
            current_price = float(ticker['lastPrice'])
            
            # Saatlik değişim için kline
            kline_resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1h', 'limit': 25},
                timeout=10
            )
            
            if kline_resp.status_code == 200:
                klines = kline_resp.json()
                closes = [float(k[4]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                change_1h = ((current_price / closes[-2]) - 1) * 100 if len(closes) >= 2 else 0
                avg_volume = sum(volumes[:-1]) / (len(volumes) - 1) if len(volumes) > 1 else 1
                volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
            else:
                change_1h = 0
                volume_ratio = 1.0
            
            return {
                'current_price': current_price,
                'change_1h': change_1h,
                'change_24h': float(ticker.get('priceChangePercent', 0)),
                'volume_ratio': volume_ratio,
                'high_24h': float(ticker.get('highPrice', current_price)),
                'low_24h': float(ticker.get('lowPrice', current_price)),
                'closes': closes if 'closes' in dir() else [current_price]
            }
            
        except Exception as e:
            logger.debug(f"Price data fetch failed: {e}")
            return None
    
    async def _get_derivatives_data(self, symbol: str) -> Dict:
        """Futures verileri al (funding, OI, L/S ratio)."""
        result = {'funding_rate': 0, 'long_short_ratio': 1.0, 'open_interest': 0}
        
        try:
            # Funding rate
            fr_resp = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            if fr_resp.status_code == 200:
                data = fr_resp.json()
                if data:
                    result['funding_rate'] = float(data[0].get('fundingRate', 0)) * 100  # %
            
            # Long/Short ratio
            ls_resp = requests.get(
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                params={'symbol': symbol, 'period': '1h', 'limit': 1},
                timeout=5
            )
            if ls_resp.status_code == 200:
                data = ls_resp.json()
                if data:
                    result['long_short_ratio'] = float(data[0].get('longShortRatio', 1.0))
            
            # Open Interest
            oi_resp = requests.get(
                "https://fapi.binance.com/fapi/v1/openInterest",
                params={'symbol': symbol},
                timeout=5
            )
            if oi_resp.status_code == 200:
                result['open_interest'] = float(oi_resp.json().get('openInterest', 0))
                
        except Exception as e:
            logger.debug(f"Derivatives fetch failed: {e}")
        
        return result
    
    async def _get_sentiment_data(self) -> Dict:
        """Sentiment verileri al."""
        result = {'fear_greed_index': 50, 'btc_dominance': 50}
        
        try:
            # Fear & Greed Index
            fg_resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=5
            )
            if fg_resp.status_code == 200:
                data = fg_resp.json().get('data', [])
                if data:
                    result['fear_greed_index'] = int(data[0].get('value', 50))
            
            # BTC Dominance from CoinGecko
            cg_resp = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=10
            )
            if cg_resp.status_code == 200:
                data = cg_resp.json().get('data', {})
                result['btc_dominance'] = data.get('market_cap_percentage', {}).get('btc', 50)
                
        except Exception as e:
            logger.debug(f"Sentiment fetch failed: {e}")
        
        return result
    
    # =========================================================================
    # CALCULATIONS
    # =========================================================================
    
    def _calculate_volatility(self, price_data: Dict) -> float:
        """Volatilite hesapla (% olarak)."""
        closes = price_data.get('closes', [])
        if len(closes) < 10:
            return 2.0  # Default
        
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * np.sqrt(24) * 100  # Annualized daily
        return round(volatility, 2)
    
    def _detect_trend(self, price_data: Dict) -> str:
        """Trend tespit et: BULL, BEAR, RANGE."""
        closes = price_data.get('closes', [])
        if len(closes) < 20:
            return 'RANGE'
        
        # EMA crossover
        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        
        if ema9 > ema21 * 1.005:  # 0.5% above
            return 'BULL'
        elif ema9 < ema21 * 0.995:
            return 'BEAR'
        else:
            return 'RANGE'
    
    def _detect_regime(self, volatility: float, trend: str) -> str:
        """Piyasa rejimi tespit et."""
        if volatility > 5:
            return 'VOLATILE'
        elif trend == 'RANGE':
            return 'RANGING'
        else:
            return 'TRENDING'
    
    def _ema(self, data: List[float], period: int) -> float:
        """EMA hesapla."""
        if len(data) < period:
            return data[-1] if data else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period  # Initial SMA
        
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    async def _get_predictions(self, symbol: str, context: MarketContext) -> Dict:
        """Tüm tahminleri al."""
        predictions = {
            'lstm': {'direction': 'NEUTRAL', 'confidence': 0},
            'pattern': {'pattern': None, 'bias': 'NEUTRAL'},
            'momentum': {'direction': 'NEUTRAL', 'strength': 0}
        }
        
        # 1. LSTM Tahmin (eğer model varsa)
        try:
            from src.brain.models.lstm_trend import LSTMTrendPredictor
            
            if symbol not in self.lstm_models:
                self.lstm_models[symbol] = LSTMTrendPredictor(symbol=symbol)
            
            model = self.lstm_models[symbol]
            if model.trained:
                # Veri hazırla
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={'symbol': symbol, 'interval': '1h', 'limit': 30},
                    timeout=10
                )
                if resp.status_code == 200:
                    klines = resp.json()
                    df = pd.DataFrame({
                        'close': [float(k[4]) for k in klines],
                        'volume': [float(k[5]) for k in klines],
                        'high': [float(k[2]) for k in klines],
                        'low': [float(k[3]) for k in klines]
                    })
                    
                    pred = model.predict(df)
                    predictions['lstm'] = {
                        'direction': pred.get('direction', 'NEUTRAL'),
                        'confidence': pred.get('confidence', 0)
                    }
        except Exception as e:
            logger.debug(f"LSTM prediction failed: {e}")
        
        # 2. Momentum Analizi (basit ama etkili)
        if context.change_1h > 1.5:
            predictions['momentum'] = {'direction': 'LONG', 'strength': min(abs(context.change_1h) * 20, 80)}
        elif context.change_1h < -1.5:
            predictions['momentum'] = {'direction': 'SHORT', 'strength': min(abs(context.change_1h) * 20, 80)}
        
        # 3. Pattern Recognition (basit)
        predictions['pattern'] = self._detect_patterns(context)
        
        return predictions
    
    def _detect_patterns(self, context: MarketContext) -> Dict:
        """Basit pattern tespiti."""
        result = {'pattern': None, 'bias': 'NEUTRAL'}
        
        # Fear & Greed extreme
        if context.fear_greed <= 20:
            result = {'pattern': 'EXTREME_FEAR', 'bias': 'LONG'}  # Contrarian
        elif context.fear_greed >= 80:
            result = {'pattern': 'EXTREME_GREED', 'bias': 'SHORT'}
        
        # Funding rate extreme
        if context.funding_rate > 0.1:
            result = {'pattern': 'HIGH_FUNDING', 'bias': 'SHORT'}
        elif context.funding_rate < -0.1:
            result = {'pattern': 'NEGATIVE_FUNDING', 'bias': 'LONG'}
        
        # Long/Short imbalance
        if context.long_short_ratio > 1.8:
            result = {'pattern': 'LONG_HEAVY', 'bias': 'SHORT'}  # Contrarian
        elif context.long_short_ratio < 0.6:
            result = {'pattern': 'SHORT_HEAVY', 'bias': 'LONG'}
        
        return result
    
    # =========================================================================
    # SIGNAL SYNTHESIS
    # =========================================================================
    
    def _synthesize_signal(
        self, 
        symbol: str, 
        context: MarketContext, 
        predictions: Dict
    ) -> Optional[Signal]:
        """Tüm bilgilerden sinyal sentezle."""
        
        # Vote toplama
        votes = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        sources = []
        
        # 1. LSTM vote
        lstm = predictions['lstm']
        if lstm['direction'] == 'UP' and lstm['confidence'] > 50:
            votes['LONG'] += lstm['confidence'] * 0.3
            sources.append(f"LSTM({lstm['confidence']:.0f}%)")
        elif lstm['direction'] == 'DOWN' and lstm['confidence'] > 50:
            votes['SHORT'] += lstm['confidence'] * 0.3
            sources.append(f"LSTM({lstm['confidence']:.0f}%)")
        
        # 2. Momentum vote
        mom = predictions['momentum']
        if mom['direction'] == 'LONG':
            votes['LONG'] += mom['strength'] * 0.25
            sources.append(f"Momentum+{context.change_1h:.1f}%")
        elif mom['direction'] == 'SHORT':
            votes['SHORT'] += mom['strength'] * 0.25
            sources.append(f"Momentum{context.change_1h:.1f}%")
        
        # 3. Pattern vote
        pattern = predictions['pattern']
        if pattern['bias'] == 'LONG':
            votes['LONG'] += 20
            sources.append(pattern['pattern'])
        elif pattern['bias'] == 'SHORT':
            votes['SHORT'] += 20
            sources.append(pattern['pattern'])
        
        # 4. Trend alignment bonus
        if context.trend == 'BULL':
            votes['LONG'] += 15
        elif context.trend == 'BEAR':
            votes['SHORT'] += 15
        
        # Kazananı bul
        total = sum(votes.values())
        if total == 0:
            return None
        
        direction = max(votes, key=votes.get)
        confidence = (votes[direction] / total) * 100
        
        if direction == 'HOLD' or confidence < self.MIN_CONFIDENCE:
            return None
        
        # TP/SL hesapla
        price = context.price
        volatility_factor = max(1.5, min(4.0, context.volatility))  # 1.5-4%
        
        if direction == 'LONG':
            stop_loss = price * (1 - volatility_factor / 100)
            take_profit = price * (1 + volatility_factor * 2 / 100)
        else:
            stop_loss = price * (1 + volatility_factor / 100)
            take_profit = price * (1 - volatility_factor * 2 / 100)
        
        risk = abs(price - stop_loss)
        reward = abs(take_profit - price)
        risk_reward = reward / risk if risk > 0 else 0
        
        if risk_reward < self.MIN_RISK_REWARD:
            return None
        
        # Reasoning oluştur
        reasoning = self._generate_reasoning(direction, context, predictions, sources)
        
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=price,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            risk_reward=round(risk_reward, 2),
            sources=sources,
            reasoning=reasoning
        )
    
    def _generate_reasoning(
        self, 
        direction: str, 
        context: MarketContext,
        predictions: Dict,
        sources: List[str]
    ) -> str:
        """Türkçe açıklama oluştur."""
        
        dir_tr = 'ALIŞ' if direction == 'LONG' else 'SATIŞ'
        trend_tr = {'BULL': 'yükseliş', 'BEAR': 'düşüş', 'RANGE': 'yatay'}[context.trend]
        
        parts = [
            f"📊 {dir_tr} sinyali oluşturuldu.",
            f"Piyasa {trend_tr} trendinde.",
            f"Korku/Açgözlülük: {context.fear_greed}",
            f"Funding: {context.funding_rate:.3f}%",
        ]
        
        if sources:
            parts.append(f"Kaynaklar: {', '.join(sources)}")
        
        return " | ".join(parts)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _can_signal(self, symbol: str) -> bool:
        """Cooldown kontrolü."""
        if symbol not in self.last_signals:
            return True
        
        elapsed = (datetime.now() - self.last_signals[symbol]).total_seconds()
        return elapsed >= self.SIGNAL_COOLDOWN
    
    def format_for_telegram(self, signal: Signal) -> str:
        """Telegram formatında sinyal."""
        emoji = '🟢' if signal.direction == 'LONG' else '🔴'
        dir_tr = 'AL' if signal.direction == 'LONG' else 'SAT'
        
        stars = '⭐' * min(5, int(signal.confidence / 20))
        
        return f"""
🧠 DEMIR AI SİNYAL
━━━━━━━━━━━━━━━━━━
{emoji} {signal.symbol}: {dir_tr}
📊 Güven: %{signal.confidence:.0f} {stars}
💰 Giriş: ${signal.entry_price:,.2f}
🎯 Kar Al: ${signal.take_profit:,.2f}
🛑 Zarar Kes: ${signal.stop_loss:,.2f}
📈 R:R = {signal.risk_reward:.1f}
━━━━━━━━━━━━━━━━━━
{signal.reasoning}
━━━━━━━━━━━━━━━━━━
⏰ {signal.timestamp.strftime('%d.%m.%Y %H:%M')}
""".strip()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_brain: Optional[UnifiedBrain] = None

def get_unified_brain() -> UnifiedBrain:
    """Get or create unified brain instance."""
    global _brain
    if _brain is None:
        _brain = UnifiedBrain()
    return _brain


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        brain = get_unified_brain()
        signal = await brain.analyze('BTCUSDT')
        
        if signal:
            print(brain.format_for_telegram(signal))
        else:
            print("No signal generated (below confidence threshold or cooldown)")
    
    asyncio.run(test())
