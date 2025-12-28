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
    LeadingIndicators, 
    LeadingSignal, 
    SignalDirection,
    get_leading_indicators
)

# NEW MODULES FOR PRECISION
from src.brain.liquidation_hunter import get_liquidation_hunter
from src.brain.pattern_engine import get_pattern_engine
from src.brain.pivot_points import get_pivot_points
from src.brain.volatility_predictor import VolatilityPredictor
from src.brain.news_scraper import CryptoNewsScraper
from src.brain.regime_classifier import RegimeClassifier

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
        self.liquidation_hunter = None
        self.pattern_engine = None
        self.pivot_analyzer = None
        self.volatility_predictor = VolatilityPredictor()
        
        # SENSORY ORGANS (Eyes & Ears)
        self.news_scraper = CryptoNewsScraper()
        self.regime_classifier = RegimeClassifier()
        
        self.feature_collector = FeatureCollector()
        self.ml_model = None  # Sonra yüklenecek
        self._last_signals: Dict[str, EarlySignal] = {}
        
        logger.info("🚀 Early Signal Engine initialized")
    
    async def initialize(self):
        """Async initialization"""
    async def initialize(self):
        """Async initialization"""
        self.leading_indicators = await get_leading_indicators()
        self.liquidation_hunter = get_liquidation_hunter()
        self.pattern_engine = get_pattern_engine()
        self.pivot_analyzer = get_pivot_points()
    
    async def analyze(self, symbol: str = "BTCUSDT") -> EarlySignal:
        """
        Sembol için erken sinyal analizi yap.
        """
        if not self.leading_indicators:
            await self.initialize()
        
        logger.info(f"🔍 Early Signal Analysis: {symbol}")
        
        # PARALLEL EXECUTION OF ALL BRAIN MODULES
        # 1. Leading indicators
        # 2. Liquidation Levels (Magnet Zones)
        # 3. Chart Patterns
        # 4. Pivot Points (Support/Resistance)
        # 5. Volatility (Squeeze/Breakout)
        # 6. News Sentiment (Global Mood)
        # 7. Market Regime (Trending/Ranging) via internal helper
        
        tasks = [
            self.leading_indicators.calculate_all(symbol),
            self.liquidation_hunter.analyze(symbol),
            self.pattern_engine.analyze(symbol),
            self.pivot_analyzer.analyze(symbol),
            asyncio.to_thread(self.volatility_predictor.predict_volatility, symbol),
            asyncio.to_thread(self.news_scraper.get_market_sentiment),
            self._analyze_regime(symbol) # Helper for RegimeClassifier
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        leading_signal = results[0] if not isinstance(results[0], Exception) else None
        liquidation_data = results[1] if not isinstance(results[1], Exception) else {}
        pattern_data = results[2] if not isinstance(results[2], Exception) else {}
        pivot_data = results[3] if not isinstance(results[3], Exception) else {}
        volatility_data = results[4] if not isinstance(results[4], Exception) else {}
        news_data = results[5] if not isinstance(results[5], Exception) else {}
        regime_data = results[6] if not isinstance(results[6], Exception) else {"regime": "UNKNOWN"}
        
        if not leading_signal:
             return None

        # 2. Güncel fiyatı al
        current_price = await self._get_current_price(symbol)
        if current_price == 0 and pivot_data.get('current_price'):
            current_price = pivot_data['current_price']
        
        # 3. Feature vector çıkar
        features = self.feature_collector.extract_features(leading_signal, current_price)
        
        # 4. ML prediction (eğer model varsa)
        ml_prediction = None
        if self.ml_model:
            ml_prediction = self._predict_with_ml(features)
        
        # 5. Sinyal üret - ENRICHED WITH ALL DATA
        signal = self._generate_signal(
            symbol=symbol, 
            leading=leading_signal, 
            current_price=current_price, 
            ml_pred=ml_prediction,
            liq_data=liquidation_data,
            pattern_data=pattern_data,
            pivot_data=pivot_data,
            vol_data=volatility_data,
            news_data=news_data,
            regime_data=regime_data
        )
        
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
    
    async def _analyze_regime(self, symbol: str) -> Dict:
        """Fetch data and classify regime - NO MOCK DATA"""
        try:
            import aiohttp
            import pandas as pd
            import numpy as np
            
            async with aiohttp.ClientSession() as session:
                 # Fetch 100 candles (1h) for regime analysis
                 url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=100"
                 async with session.get(url) as resp:
                     if resp.status == 200:
                         data = await resp.json()
                         df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
                         df['close'] = df['close'].astype(float)
                         df['high'] = df['high'].astype(float)
                         df['low'] = df['low'].astype(float)
                         df['volume'] = df['volume'].astype(float)
                         
                         # === MANUAL ADX CALCULATION ===
                         # True Range
                         df['prev_close'] = df['close'].shift(1)
                         df['tr1'] = df['high'] - df['low']
                         df['tr2'] = abs(df['high'] - df['prev_close'])
                         df['tr3'] = abs(df['low'] - df['prev_close'])
                         df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
                         
                         # +DM / -DM
                         df['up_move'] = df['high'] - df['high'].shift(1)
                         df['down_move'] = df['low'].shift(1) - df['low']
                         df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
                         df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
                         
                         # Smoothed TR, +DM, -DM (14 period)
                         period = 14
                         df['atr'] = df['tr'].rolling(window=period).mean()
                         df['+di'] = 100 * (df['+dm'].rolling(window=period).mean() / df['atr'])
                         df['-di'] = 100 * (df['-dm'].rolling(window=period).mean() / df['atr'])
                         
                         # DX and ADX
                         df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
                         df['adx'] = df['dx'].rolling(window=period).mean()
                         
                         # === MANUAL BOLLINGER BANDS WIDTH ===
                         mid = df['close'].rolling(window=20).mean()
                         std = df['close'].rolling(window=20).std()
                         upper = mid + (std * 2)
                         lower = mid - (std * 2)
                         df['bb_width'] = (upper - lower) / mid
                         
                         # === MANUAL VWAP ===
                         df['tp'] = (df['high'] + df['low'] + df['close']) / 3
                         df['vwap'] = (df['tp'] * df['volume']).cumsum() / df['volume'].cumsum()
                         
                         # Classify
                         regime = self.regime_classifier.identify_regime(df)
                         risk_adj = self.regime_classifier.get_risk_adjustment(regime)
                         
                         return {"regime": regime, "risk_adjustment": risk_adj}
        except Exception as e:
            logger.warning(f"Regime analysis failed: {e}")
            
        return {"regime": "UNKNOWN", "risk_adjustment": {}}

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
        ml_pred: Optional[Dict],
        liq_data: Dict,
        pattern_data: Dict,
        pivot_data: Dict,
        vol_data: Dict,
        news_data: Dict,
        regime_data: Dict
    ) -> EarlySignal:
        """
        Leading signals + All Modules'dan trading sinyali üret.
        """
        # Yön belirleme
        action = "HOLD"
        
        # 1. Core Direction
        if leading.direction in [SignalDirection.STRONG_BULLISH, SignalDirection.BULLISH]:
            action = "BUY"
        elif leading.direction in [SignalDirection.STRONG_BEARISH, SignalDirection.BEARISH]:
            action = "SELL"
            
        # --- MARKET REGIME & SENTIMENT FILTER ---
        regime = regime_data.get('regime', 'UNKNOWN')
        sentiment = news_data.get('sentiment', 'NEUTRAL')
        
        # Filter: Don't trade against extreme Sentiment (Contra-trading logic can be risky for bots)
        # If Sentiment is BEARISH and Signal is BUY -> Reduce Confidence
        if sentiment == 'BEARISH' and action == "BUY":
            confidence_penalty = 20
        elif sentiment == 'BULLISH' and action == "SELL":
            confidence_penalty = 20
        else:
            confidence_penalty = 0

        # Regime Adjustment
        regime_adj = regime_data.get('risk_adjustment', {})
        target_confidence = regime_adj.get('confidence_threshold', 0.60) * 100
        
        # Confidence calculation
        confidence = leading.confidence 
        if leading.direction.value.startswith("STRONG"):
            confidence += 15
            
        confidence -= confidence_penalty
        
        # 2. Pattern Validation (Confirmation)
        pattern_score = 0
        pattern_reason = ""
        if pattern_data and 'patterns' in pattern_data:
            for pat in pattern_data['patterns']:
                if pat['confidence'] > 50:
                    icon = "📈" if pat['type'] == 'BULLISH' else "📉"
                    pattern_reason = f"{icon} {pat['name']}"
                    # Bonus confidence for patterns
                    confidence += 10

        # 3. Volatility Check (Breakout Validation)
        vol_state = vol_data.get('state', 'NORMAL')
        is_breakout = vol_state == 'SQUEEZE' or vol_data.get('volatility_ratio', 1) < 0.8
        
        # Confidence hesapla
        confidence = leading.confidence
        if leading.direction.value.startswith("STRONG"):
            confidence += 15
        
        if is_breakout:
            confidence += 10 # Bonus for timing
        
        # Minimum confidence threshold
        if confidence < 50:
            action = "HOLD"
        
        # --- PRECISION LEVELS (SL/TP) ---
        
        # Default limits
        stop_loss = current_price * 0.98
        take_profit = current_price * 1.04
        
        # Use Pivot Points for Precision
        reasons = []
        
        if pivot_data and 'daily_pivots' in pivot_data:
            pivots = pivot_data['daily_pivots']
            
            if action == "BUY":
                # TP = Next Resistance
                # SL = Nearest Support
                
                # Find R1, R2, S1
                r1 = next((p['price'] for p in pivots if '_R1' in p['name']), None)
                r2 = next((p['price'] for p in pivots if '_R2' in p['name']), None)
                s1 = next((p['price'] for p in pivots if '_S1' in p['name']), None)
                
                if r1 and r1 > current_price:
                    take_profit = r1
                    reasons.append(f"🎯 TP at Daily R1 (${r1:,.0f})")
                elif r2:
                    take_profit = r2
                    
                if s1 and s1 < current_price:
                    stop_loss = s1
                    reasons.append(f"🛡️ SL at Daily S1 (${s1:,.0f})")
                    
            elif action == "SELL":
                # TP = Next Support
                # SL = Nearest Resistance
                
                s1 = next((p['price'] for p in pivots if '_S1' in p['name']), None)
                s2 = next((p['price'] for p in pivots if '_S2' in p['name']), None)
                r1 = next((p['price'] for p in pivots if '_R1' in p['name']), None)
                
                if s1 and s1 < current_price:
                    take_profit = s1
                    reasons.append(f"🎯 TP at Daily S1 (${s1:,.0f})")
                elif s2:
                    take_profit = s2
                
                if r1 and r1 > current_price:
                    stop_loss = r1
                    reasons.append(f"🛡️ SL at Daily R1 (${r1:,.0f})")

        # --- SMC BASED SL/TP LABELS ---
        # Use Order Blocks and Liquidity for precision
        if pattern_data and 'order_blocks' in pattern_data:
            ob_data = pattern_data['order_blocks']
            
            if action == "BUY":
                # LONG: SL below nearest bullish OB
                bullish_obs = ob_data.get('bullish', [])
                if bullish_obs:
                    nearest_ob = min(bullish_obs, key=lambda x: abs(x.get('price', 0) - current_price))
                    ob_price = nearest_ob.get('price', 0)
                    if ob_price > 0 and ob_price < current_price:
                        stop_loss = ob_price * 0.995  # Just below OB
                        reasons.append(f"🧱 SL below OB (${ob_price:,.0f})")
                        
            elif action == "SELL":
                # SHORT: SL above nearest bearish OB
                bearish_obs = ob_data.get('bearish', [])
                if bearish_obs:
                    nearest_ob = min(bearish_obs, key=lambda x: abs(x.get('price', 0) - current_price))
                    ob_price = nearest_ob.get('price', 0)
                    if ob_price > 0 and ob_price > current_price:
                        stop_loss = ob_price * 1.005  # Just above OB
                        reasons.append(f"🧱 SL above OB (${ob_price:,.0f})")
        
        # Use Liquidation Magnet as TP
        if liq_data and liq_data.get('magnet_zone', 0) > 0:
            magnet = liq_data['magnet_zone']
            if action == "BUY" and magnet > current_price:
                take_profit = magnet
                reasons.append(f"🧲 TP at Liq Magnet (${magnet:,.0f})")
            elif action == "SELL" and magnet < current_price:
                take_profit = magnet
                reasons.append(f"🧲 TP at Liq Magnet (${magnet:,.0f})")

        # Risk/Reward
        if action != "HOLD":
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0
            
        # Add Module Reasons
        if is_breakout:
            reasons.append("🌋 Volatility Squeeze (Big Move Incoming)")
            
        if regime != "UNKNOWN":
            reasons.append(f"🧠 Regime: {regime}")
            
        if sentiment != 'NEUTRAL':
            icon = "🐂" if sentiment == 'BULLISH' else "🐻"
            reasons.append(f"{icon} Sentiment: {sentiment}")
            
        if pattern_reason:
            reasons.append(pattern_reason)

        if liq_data and 'heatmap_clusters' in liq_data:
             clusters = liq_data['heatmap_clusters']
             if clusters:
                 best_cluster = clusters[0] # Assume sorted via intensity
                 reasons.append(f"🧲 Magnet Level: ${best_cluster['price']:,.0f}")

        # Leading reasons
        for ind in leading.indicators:
            if abs(ind.value) > 15:
                emoji = "🟢" if ind.value > 0 else "🔴"
                reasons.append(f"{emoji} {ind.name}: {ind.value:+.0f}")
        
        reasoning = " | ".join(reasons) if reasons else "No strong signals"
        
        return EarlySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_zone=(round(current_price * 0.999, 2), round(current_price * 1.001, 2)),
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
