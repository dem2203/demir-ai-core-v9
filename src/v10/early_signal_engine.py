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

# AI MODELS - Real AI Decision Making
from src.v10.lstm_predictor import get_lstm_predictor, PricePrediction
from src.v10.ai_integration import get_ai_bridge

# HYBRID AI SYSTEM - Macro Context + LLM Brain + Momentum Detector
from src.brain.macro_context import get_macro_context, MacroContext
from src.brain.llm_brain import get_llm_brain
from src.brain.momentum_detector import get_momentum_context, MomentumContext
from src.brain.risk_manager import get_risk_manager
from src.brain.fractal_analyzer import get_fractal_analyzer
from src.brain.institutional_aggregator import InstitutionalAggregator, LiveDataSnapshot # PHASE 14

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
    llm_reasoning: str = ""        # Claude Haiku reasoning
    momentum_alerts: list = None   # Volume spike, etc.
    score_breakdown: Dict = None   # Tech, Macro, Onchain scores
    score_breakdown: Dict = None   # Tech, Macro, Onchain scores
    risk_profile: Dict = None      # 💰 Smart Risk Manager output
    institutional_data: Dict = None # 🏦 PHASE 14: Raw Institutional Data
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.momentum_alerts is None:
            self.momentum_alerts = []
        if self.score_breakdown is None:
            self.score_breakdown = {}
    
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
            'llm_reasoning': self.llm_reasoning,
            'momentum_alerts': self.momentum_alerts,
            'risk_profile': self.risk_profile,
            'risk_profile': self.risk_profile,
            'leading_indicators': self.leading_signal.to_dict(),
            'institutional_data': self.institutional_data,
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
        
        # Dashboard Cache State
        self.latest_signals: Dict[str, Dict] = {}
        self.latest_macro: Optional[Dict] = None
        
        # AI BRAIN - Real AI Decision Making
        self.lstm_predictor = get_lstm_predictor()
        self.ai_bridge = get_ai_bridge()  # For RL Agent access
        
        # HYBRID AI - LLM Brain (Claude Haiku)
        self.llm_brain = get_llm_brain()
        self.fractal_analyzer = get_fractal_analyzer() # NEW
        self.institutional_aggregator = InstitutionalAggregator() # PHASE 14 integration
        
        self.feature_collector = FeatureCollector()
        self.ml_model = None  # Legacy - replaced by lstm_predictor
        self._last_signals: Dict[str, EarlySignal] = {}
        
        logger.info("🚀 Early Signal Engine + AI BRAIN + ENSEMBLE VOTING initialized")
    
    async def initialize(self):
        """Async initialization"""
        self.leading_indicators = await get_leading_indicators()
        self.liquidation_hunter = get_liquidation_hunter()
        self.pattern_engine = get_pattern_engine()
        self.pivot_analyzer = get_pivot_points()
    
    async def analyze(self, symbol: str = "BTCUSDT") -> EarlySignal:
        """
        HYBRID AI ANALYSIS
        Sembol için erken sinyal analizi yap.
        Teknik + Makro + LLM kombinasyonu kullanır.
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
        # 7. Market Regime (Trending/Ranging)
        # 8. MACRO CONTEXT (BTC.D, Fear Index) - NEW!
        
        tasks = [
            self.leading_indicators.calculate_all(symbol),
            self.liquidation_hunter.analyze(symbol),
            self.pattern_engine.analyze(symbol),
            self.pivot_analyzer.analyze(symbol),
            asyncio.to_thread(self.volatility_predictor.predict_volatility, symbol),
            asyncio.to_thread(self.news_scraper.get_market_sentiment),
            self._analyze_regime(symbol),
            get_macro_context(),
            get_momentum_context(symbol)  # NEW: Momentum for breakout detection
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        leading_signal = results[0] if not isinstance(results[0], Exception) else None
        liquidation_data = results[1] if not isinstance(results[1], Exception) else {}
        pattern_data = results[2] if not isinstance(results[2], Exception) else {}
        pivot_data = results[3] if not isinstance(results[3], Exception) else {}
        volatility_data = results[4] if not isinstance(results[4], Exception) else {}
        news_data = results[5] if not isinstance(results[5], Exception) else {}
        regime_data = results[6] if not isinstance(results[6], Exception) else {"regime": "UNKNOWN"}
        macro_context = results[7] if not isinstance(results[7], Exception) else None
        momentum_context = results[8] if not isinstance(results[8], Exception) else None
        
        if not leading_signal:
             return None
             
        # Use snapshot klines for Fractal check
        fractal_match = None
        if self.leading_indicators and self.leading_indicators.latest_snapshot:
             try:
                 snapshot = self.leading_indicators.latest_snapshot
                 # Need at least 100 candles
                 if len(snapshot.klines) > 100:
                     closes = [float(k[4]) for k in snapshot.klines]
                     fractal_match = self.fractal_analyzer.find_fractal_match(closes, closes)
             except Exception as e:
                 logger.debug(f"Fractal check failed: {e}")

        # 2. Güncel fiyatı al
        current_price = await self._get_current_price(symbol)
        if current_price == 0 and pivot_data.get('current_price'):
            current_price = pivot_data['current_price']
        
        # 3. Feature vector çıkar
        features = self.feature_collector.extract_features(leading_signal, current_price)
        
        # 4. AI BRAIN - LSTM Prediction (REAL AI)
        lstm_prediction = None
        try:
            lstm_prediction = await self.lstm_predictor.predict(symbol)
            if lstm_prediction:
                logger.info(f"🧠 LSTM: {symbol} → {lstm_prediction.direction} ({lstm_prediction.predicted_change_pct:+.2f}%)")
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
        
        # 5. FRACTAL MEMORY (Pattern Matching) - Existing logic moved/modified
        # The existing fractal_match logic is already above, so we'll just ensure it's used.
        
        # 6. Institutional AGGREGATOR (PHASE 14 - Brain Activation)
        # ---------------------------------------------------
        # Fetch comprehensive data snapshot
        inst_snapshot = await self.institutional_aggregator.get_live_snapshot(symbol)
        sudden_triggers = await self.institutional_aggregator.check_sudden_triggers(symbol)

        # 5. Sinyal üret - HYBRID AI BRAIN DECISION
        signal = await self._generate_signal(
            symbol=symbol, 
            leading=leading_signal, 
            current_price=current_price, 
            ml_pred=lstm_prediction,
            liq_data=liquidation_data,
            pattern_data=pattern_data,
            pivot_data=pivot_data,
            vol_data=volatility_data,
            news_data=news_data,
            regime_data=regime_data,
            macro_context=macro_context,
            momentum_context=momentum_context,
            fractal_match=fractal_match, # NEW
            inst_snapshot=inst_snapshot, # NEW
            sudden_triggers=sudden_triggers # NEW
        )
        
        # 6. Training için veri topla
        self.feature_collector.collect_training_sample(leading_signal, current_price)
        
        # Cache'e kaydet
        self._last_signals[symbol] = signal
        
        # 7. Dashboard Update (NEW)
        self.latest_signals[symbol] = signal.to_dict()
        if macro_context:
            self.latest_macro = macro_context.to_dict()
        self._update_dashboard_json()
        
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

    async def _check_multi_timeframe_confluence(self, symbol: str) -> Dict:
        """
        Multi-Timeframe Confluence Check
        Analyzes 1h, 4h, 1d trends and checks agreement.
        
        Returns:
            confluence_score: -100 to +100 (higher = more agreement)
            trend_alignment: True/False
            dominant_trend: "BULLISH" / "BEARISH" / "MIXED"
        """
        try:
            import aiohttp
            
            trends = {}
            timeframes = ['1h', '4h', '1d']
            
            async with aiohttp.ClientSession() as session:
                for tf in timeframes:
                    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={tf}&limit=50"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            klines = await resp.json()
                            closes = [float(k[4]) for k in klines]
                            
                            # Simple trend determination: EMA9 vs EMA21
                            if len(closes) >= 21:
                                ema9 = sum(closes[-9:]) / 9
                                ema21 = sum(closes[-21:]) / 21
                                
                                if ema9 > ema21 * 1.002:  # 0.2% threshold
                                    trends[tf] = "BULLISH"
                                elif ema9 < ema21 * 0.998:
                                    trends[tf] = "BEARISH"
                                else:
                                    trends[tf] = "NEUTRAL"
            
            # Calculate confluence
            bullish_count = sum(1 for t in trends.values() if t == "BULLISH")
            bearish_count = sum(1 for t in trends.values() if t == "BEARISH")
            
            if bullish_count >= 2:
                confluence_score = 50 + (bullish_count * 15)
                dominant_trend = "BULLISH"
                trend_alignment = bullish_count == 3
            elif bearish_count >= 2:
                confluence_score = -(50 + (bearish_count * 15))
                dominant_trend = "BEARISH"
                trend_alignment = bearish_count == 3
            else:
                confluence_score = 0
                dominant_trend = "MIXED"
                trend_alignment = False
            
            logger.info(f"📊 MTF Confluence {symbol}: {trends} → {dominant_trend} ({confluence_score})")
            
            return {
                'confluence_score': confluence_score,
                'trend_alignment': trend_alignment,
                'dominant_trend': dominant_trend,
                'timeframe_trends': trends
            }
            
        except Exception as e:
            logger.warning(f"MTF Confluence check failed: {e}")
            return {'confluence_score': 0, 'trend_alignment': False, 'dominant_trend': 'UNKNOWN'}

    def _predict_with_ml(self, features: np.ndarray) -> Dict:
        """ML modeli ile tahmin (legacy - replaced by LSTM)"""
        return {
            'prediction': 'NEUTRAL',
            'confidence': 0,
            'note': 'Replaced by LSTM predictor'
        }
    
    async def _generate_signal(
        self, 
        symbol: str, 
        leading: LeadingSignal,
        current_price: float,
        ml_pred,  # PricePrediction or None
        liq_data: Dict,
        pattern_data: Dict,
        pivot_data: Dict,
        vol_data: Dict,
        news_data: Dict,
        regime_data: Dict,
        macro_ctx = None, # Renamed to fix sync issue
        momentum_context = None,  # NEW: MomentumContext for breakout detection
        fractal_match = None,      # NEW: Fractal memory
        inst_snapshot: LiveDataSnapshot = None, # NEW
        sudden_triggers = None # NEW
    ) -> EarlySignal:
        """
        HYBRID AI BRAIN DECISION
        """
        # === HYBRID AI DECISION SYSTEM ===
        
        # Score tracking: -100 to +100
        ai_score = 0
        reasons = []
        llm_reasoning = ""
        
        # --- 1. LSTM PREDICTION (Weight: 40%) ---
        lstm_weight = 40
        if ml_pred and hasattr(ml_pred, 'direction'):
            if ml_pred.direction == "UP":
                lstm_contribution = lstm_weight * (ml_pred.confidence / 100)
                ai_score += lstm_contribution
                reasons.append(f"🧠 LSTM: UP +{ml_pred.predicted_change_pct:.1f}%")
            elif ml_pred.direction == "DOWN":
                lstm_contribution = lstm_weight * (ml_pred.confidence / 100)
                ai_score -= lstm_contribution
                reasons.append(f"🧠 LSTM: DOWN {ml_pred.predicted_change_pct:.1f}%")
            else:
                reasons.append(f"🧠 LSTM: FLAT")
        else:
            # Fallback: Use leading indicators (reduced weight)
            if leading.direction in [SignalDirection.STRONG_BULLISH, SignalDirection.BULLISH]:
                ai_score += 20
            elif leading.direction in [SignalDirection.STRONG_BEARISH, SignalDirection.BEARISH]:
                ai_score -= 20
        
        # --- 2. ORDER BOOK PRESSURE (Weight: 25%) ---
        ob_weight = 25
        orderbook_score = leading.orderbook_score
        if orderbook_score > 50:  # Strong buy pressure
            ai_score += ob_weight * (orderbook_score / 100)
            reasons.append(f"📗 OB: BUY Pressure +{orderbook_score}")
        elif orderbook_score < -50:  # Strong sell pressure
            ai_score -= ob_weight * (abs(orderbook_score) / 100)
            reasons.append(f"📕 OB: SELL Pressure {orderbook_score}")
        
        # --- 3. WHALE ACTIVITY (Weight: 15%) ---
        whale_weight = 15
        whale_score = leading.whale_score
        if whale_score > 30:
            ai_score += whale_weight * (whale_score / 100)
            reasons.append(f"🐋 Whale: Accumulating +{whale_score}")
        elif whale_score < -30:
            ai_score -= whale_weight * (abs(whale_score) / 100)
            reasons.append(f"🐋 Whale: Distributing {whale_score}")
        
        # --- 4. RL AGENT PREDICTION (Weight: 15%) --- ENSEMBLE VOTING
        rl_weight = 15
        try:
            # Get RL Agent decision from AI Bridge
            if self.ai_bridge and self.ai_bridge.models_loaded:
                # Build simple state vector for RL
                rl_state = np.array([
                    leading.funding_score,
                    leading.orderbook_score,
                    leading.whale_score,
                    leading.oi_divergence_score,
                    ml_pred.predicted_change_pct if ml_pred else 0
                ])
                # Pad to expected size
                rl_state = np.pad(rl_state, (0, 32))[:37]
                
                action_idx, rl_conf = self.ai_bridge.rl_agent.predict(rl_state)
                
                if action_idx == 1:  # BUY
                    ai_score += rl_weight * (rl_conf / 100)
                    reasons.append(f"🤖 RL Agent: BUY ({rl_conf:.0f}%)")
                elif action_idx == 2:  # SELL
                    ai_score -= rl_weight * (rl_conf / 100)
                    reasons.append(f"🤖 RL Agent: SELL ({rl_conf:.0f}%)")
                else:
                    reasons.append(f"🤖 RL Agent: HOLD")
        except Exception as e:
            logger.debug(f"RL Agent prediction skipped: {e}")
        
        # --- 5. VOLATILITY STATE (Weight: 10%) ---
        vol_state = vol_data.get('state', 'NORMAL')
        if vol_state == 'SQUEEZE':
            reasons.append("🌋 Volatility Squeeze (Big Move Incoming)")
            # Increase potential for big move
        
        # --- 6. NEWS SENTIMENT IMPACT (Weight: 10%) ---
        sentiment = news_data.get('sentiment', 'NEUTRAL')
        sentiment_weight = 10
        if sentiment == 'BULLISH':
            ai_score += sentiment_weight
            reasons.append("📰 News: BULLISH Sentiment")
        elif sentiment == 'BEARISH':
            ai_score -= sentiment_weight
            reasons.append("📰 News: BEARISH Sentiment")
        
        # --- 7. REGIME CONTEXT ---
        regime = regime_data.get('regime', 'UNKNOWN')
        if regime == 'TRENDING_BULL' and ai_score > 0:
            ai_score *= 1.1  # Boost bullish signals in bull market
        elif regime == 'TRENDING_BEAR' and ai_score < 0:
            ai_score *= 1.1  # Boost bearish signals in bear market
        reasons.append(f"🧠 Regime: {regime}")
        
        # --- 8. MACRO CONTEXT (Weight: 25%) --- NEW!
        macro_weight = 25
        if macro_context:
            try:
                # Fear & Greed contrarian signals
                fear_index = macro_context.fear_greed_index
                if fear_index < 25:  # Extreme Fear
                    ai_score += macro_weight * 0.5  # Contrarian BUY
                    reasons.append(f"📊 Fear Index: {fear_index} (Extreme Fear → Contrarian BUY)")
                elif fear_index > 75:  # Extreme Greed
                    ai_score -= macro_weight * 0.5  # Contrarian SELL
                    reasons.append(f"📊 Fear Index: {fear_index} (Extreme Greed → Contrarian SELL)")
                else:
                    reasons.append(f"📊 Fear Index: {fear_index} ({macro_context.fear_greed_label})")
                
                # BTC Dominance trend
                btc_d = macro_context.btc_dominance
                btc_d_change = macro_context.btc_dominance_change_24h
                if btc_d_change < -1:  # BTC.D falling = money flowing to alts
                    if symbol != "BTCUSDT":
                        ai_score += macro_weight * 0.3
                        reasons.append(f"📈 BTC.D düşüyor ({btc_d:.1f}%, {btc_d_change:+.1f}%) → Altcoin fırsatı")
                elif btc_d_change > 1:  # BTC.D rising = risk-off
                    if symbol == "BTCUSDT":
                        ai_score += macro_weight * 0.2
                    else:
                        ai_score -= macro_weight * 0.4  # Stronger penalty for alts
                        reasons.append(f"📉 BTC.D yükseliyor → Altcoin Baskısı")
                        
                # --- MACRO SHIELD (NEW) ---
                # USDT Dominance check
                usdt_d = macro_context.usdt_dominance
                usdt_d_change = macro_context.usdt_dominance_change_24h
                
                if usdt_d_change > 1.0: # Dolar güçleniyor (Panic selling)
                    ai_score -= 30  # MASSIVE PENALTY FOR ANY BUY SIGNAL
                    reasons.append(f"🛡️ MACRO SHIELD: USDT.D PUMP ({usdt_d_change:+.1f}%) → Risk Off!")
                    
            except Exception as e:
                logger.debug(f"Macro scoring error: {e}")
        
        # --- 9. MOMENTUM / BREAKOUT DETECTION (Weight: 15%) --- NEW!
        momentum_weight = 15
        if momentum_context:
            try:
                # Volume Spike = Strong signal
                if momentum_context.volume_spike:
                    if momentum_context.momentum_direction in ["STRONG_BULL", "BULL"]:
                        ai_score += momentum_weight * 0.4
                        reasons.append(f"🔥 Volume Spike + Bullish Momentum ({momentum_context.volume_ratio:.1f}x)")
                    elif momentum_context.momentum_direction in ["STRONG_BEAR", "BEAR"]:
                        ai_score -= momentum_weight * 0.4
                        reasons.append(f"🔥 Volume Spike + Bearish Momentum ({momentum_context.volume_ratio:.1f}x)")
                
                if momentum_context.liq_magnet and momentum_context.liq_magnet != "NONE":
                    dist = momentum_context.liq_distance_pct
                    if dist < 2.0: # Very close (2%)
                        # Trade TOWARDS the magnet
                        if momentum_context.liq_magnet == "SHORT_LIQ": # Shorts will be squeezed (Price UP)
                             ai_score += 25 
                             reasons.append(f"🧲 LIQUIDATION HUNTER: $50M+ Short Cluster nearby ({dist:.1f}%) → Magnet UP")
                        elif momentum_context.liq_magnet == "LONG_LIQ": # Longs will be squeezed (Price DOWN)
                             ai_score -= 25
                             reasons.append(f"🧲 LIQUIDATION HUNTER: $50M+ Long Cluster nearby ({dist:.1f}%) → Magnet DOWN")

                
                # Strong momentum
                if abs(momentum_context.momentum_5m) > 0.5:
                    if momentum_context.momentum_5m > 0:
                        reasons.append(f"📈 5m Momentum: {momentum_context.momentum_5m:+.2f}%")
                    else:
                        reasons.append(f"📉 5m Momentum: {momentum_context.momentum_5m:+.2f}%")
            except Exception as e:
                logger.debug(f"Momentum scoring error: {e}")
                
        # --- 10. FRACTAL MEMORY (Weight: 10%) --- NEW!
        if fractal_match:
            try:
                # Sim > 0.85
                fractal_weight = 10
                if fractal_match.future_change_pct > 1.0: # History says UP
                    ai_score += fractal_weight
                    reasons.append(f"🧩 FRACTAL MEMORY: History repeating ({fractal_match.similarity*100:.0f}% match) → +{fractal_match.future_change_pct:.1f}%")
                elif fractal_match.future_change_pct < -1.0: # History says DOWN
                    ai_score -= fractal_weight
                    reasons.append(f"🧩 FRACTAL MEMORY: History repeating ({fractal_match.similarity*100:.0f}% match) → {fractal_match.future_change_pct:.1f}%")
            except Exception as e:
                logger.debug(f"Fractal scoring error: {e}")
        
        # --- 11. LLM BRAIN ANALYSIS (Weight: 15%) --- NEW!
        llm_weight = 15
        try:
            if self.llm_brain and self.llm_brain.is_enabled:
                # Build technical data for LLM
                technical_data = {
                    'lstm_direction': ml_pred.direction if ml_pred else 'N/A',
                    'lstm_change': ml_pred.predicted_change_pct if ml_pred else 0,
                    'lstm_confidence': ml_pred.confidence if ml_pred else 0,
                    'rsi': leading.indicators[0].value if leading.indicators else 50,
                    'orderbook_score': leading.orderbook_score,
                    'wyckoff_phase': pattern_data.get('wyckoff', {}).get('phase', 'N/A'),
                    'volatility_state': vol_state
                }
                
                macro_data = macro_ctx.to_dict() if macro_ctx else {}
                momentum_data = momentum_context.to_dict() if momentum_context else {}
                
                onchain_data = {
                    'whale_flow': leading.whale_score,
                    'funding_rate': leading.funding_score,
                    'exchange_flow': 'N/A'
                }
                
                sentiment_data = {
                    'news_sentiment': sentiment,
                    'social_sentiment': 'N/A'
                }
                
                # Call LLM Brain
                llm_analysis = await self.llm_brain.analyze(
                    symbol=symbol,
                    current_price=current_price,
                    technical_data=technical_data,
                    macro_data=macro_data,
                    onchain_data=onchain_data,
                    sentiment_data=sentiment_data,
                    momentum_data=momentum_data  # NEW: Momentum context
                )
                
                if llm_analysis:
                    # Add LLM contribution to score
                    if llm_analysis.direction == "BUY":
                        ai_score += llm_weight * (llm_analysis.confidence / 100)
                        reasons.append(f"🤖 Claude: BUY ({llm_analysis.confidence}%)")
                    elif llm_analysis.direction == "SELL":
                        ai_score -= llm_weight * (llm_analysis.confidence / 100)
                        reasons.append(f"🤖 Claude: SELL ({llm_analysis.confidence}%)")
                    else:
                        reasons.append(f"🤖 Claude: HOLD")
                    
                    # Store LLM reasoning for display
                    llm_reasoning = llm_analysis.reasoning
                    
                    logger.info(f"🧠 LLM Brain: {llm_analysis.direction} ({llm_analysis.confidence}%)")
        except Exception as e:
            logger.warning(f"LLM Brain error: {e}")
        
        # --- ADD INDIVIDUAL INDICATOR SCORES TO REASONING ---
        if leading.whale_score != 0:
            color = "🟢" if leading.whale_score > 0 else "🔴"
            reasons.append(f"{color} whale: {leading.whale_score:+.0f}")
        if leading.orderbook_score != 0:
            color = "🟢" if leading.orderbook_score > 0 else "🔴"
            reasons.append(f"{color} orderbook: {leading.orderbook_score:+.0f}")
        if leading.oi_divergence_score != 0:
            color = "🟢" if leading.oi_divergence_score > 0 else "🔴"
            reasons.append(f"{color} oi divergence: {leading.oi_divergence_score:+.0f}")
            
        # --- 12. INSTITUTIONAL DATA (PHASE 14) ---
        if inst_snapshot:
            # CME Gap
            if not inst_snapshot.cme_gap_filled:
                gap_dist = (inst_snapshot.cme_gap_price - current_price) / current_price * 100
                if abs(gap_dist) < 2.0:
                    ai_score += 15 if gap_dist > 0 else -15
                    reasons.append(f"🕳️ CME Gap Target: ${inst_snapshot.cme_gap_price:,.0f} ({gap_dist:+.1f}%)")
            
            # Exchange Netflow
            if inst_snapshot.exchange_netflow > 1000000: # Inflow > $1M
                ai_score -= 10
                reasons.append("🏦 Exchange Inflow: High Selling Risk")
            elif inst_snapshot.exchange_netflow < -1000000: # Outflow > $1M
                ai_score += 10
                reasons.append("🏦 Exchange Outflow: Accumulation")
                
            # MVRV / On-Chain extracted from MVRV Proxy logic if available or re-implemented here
            # (InstitutionalAggregator doesn't calculate MVRV Proxy by default, relies on separate call or sub-module)
            # but it has whale_net_flow
            
        # --- 13. SUDDEN TRIGGERS (PHASE 14) ---
        if sudden_triggers and sudden_triggers.should_alert:
            for trigger in sudden_triggers.triggers:
                weight = 15 if trigger.severity == "HIGH" else 8
                if trigger.direction == "BULLISH":
                    ai_score += weight
                    reasons.append(f"⚡ {trigger.name}: BULLISH ({trigger.value})")
                elif trigger.direction == "BEARISH":
                    ai_score -= weight
                    reasons.append(f"⚡ {trigger.name}: BEARISH ({trigger.value})")
                elif trigger.severity == "CRITICAL":
                    reasons.append(f"⚠️ CRTICAL ALERT: {trigger.name} ({trigger.value})")
        
        # === FINAL DECISION ===
        # Threshold: Need significant score to generate signal
        # LOWERED from 30 to 20 for more signal opportunities
        if ai_score > 20:
            action = "BUY"
            confidence = min(95, 50 + ai_score)  # 50-95% range
        elif ai_score < -20:
            action = "SELL"
            confidence = min(95, 50 + abs(ai_score))
        else:
            action = "HOLD"
            confidence = 30
        
        logger.info(f"🤖 AI Brain {symbol}: Score={ai_score:.0f} → {action} ({confidence:.0f}%)")
        
        # === DYNAMIC SL/TP (ATR-Based) ===
        # Volatility determines SL/TP distance
        volatility_ratio = vol_data.get('volatility_ratio', 1.0)
        
        # Base multipliers (will be adjusted by volatility)
        if volatility_ratio < 0.7:  # Low volatility / Squeeze
            sl_mult = 0.015  # Tighter SL (1.5%)
            tp_mult = 0.025  # Tighter TP (2.5%)
            vol_reason = "📉 Low Vol: Tight SL/TP"
        elif volatility_ratio > 1.3:  # High volatility / Expansion
            sl_mult = 0.03   # Wider SL (3%)
            tp_mult = 0.06   # Wider TP (6%)
            vol_reason = "📈 High Vol: Wide SL/TP"
        else:  # Normal volatility
            sl_mult = 0.02   # Standard SL (2%)
            tp_mult = 0.04   # Standard TP (4%)
            vol_reason = ""
        
        # Apply dynamic SL/TP
        if action == "BUY":
            stop_loss = current_price * (1 - sl_mult)
            take_profit = current_price * (1 + tp_mult)
        elif action == "SELL":
            stop_loss = current_price * (1 + sl_mult)
            take_profit = current_price * (1 - tp_mult)
        else:
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.04
        
        # Initialize reasons list
        reasons = []
        if vol_reason:
            reasons.append(vol_reason)
        
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
        if vol_state == 'SQUEEZE':
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
        
        # Prepare additional data for wrapper
        momentum_alerts = momentum_context.alerts if momentum_context else []
        
        # Calculate score breakdown (approximated for display)
        score_breakdown = {
            'technical': min(40, max(-40, leading.strength * 0.4)),
            'macro': ai_score * 0.25, # Simplified
            'onchain': leading.whale_score * 0.2 + leading.funding_score * 10,
            'llm': 15 if "Claude: BUY" in reasoning else -15 if "Claude: SELL" in reasoning else 0
        }

        # 💰 Calculate Risk Profile
        risk_manager = get_risk_manager()
        vol_ratio = vol_data.get('volatility_ratio', 1.0)
        risk_profile_obj = risk_manager.calculate_risk(confidence, vol_ratio)
        
        # Convert dataclass to dict for usage
        risk_profile = {
            'position_size_pct': risk_profile_obj.position_size_pct,
            'leverage': risk_profile_obj.leverage,
            'risk_usd': risk_profile_obj.risk_per_trade_usd,
            'reason': risk_profile_obj.reason
        }

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
            reasoning=reasoning,
            llm_reasoning=llm_reasoning,
            momentum_alerts=[a.to_dict() for a in momentum_alerts],
            score_breakdown=score_breakdown,
            risk_profile=risk_profile,
            institutional_data=inst_data # NEW
        )
    
    def get_last_signal(self, symbol: str) -> Optional[EarlySignal]:
        """Son sinyali getir"""
        return self._last_signals.get(symbol)
    
    async def close(self):
        """Cleanup"""
        if self.leading_indicators:
            await self.leading_indicators.close()
        self.feature_collector._flush_buffer()

    def _update_dashboard_json(self):
        """Dashboard verilerini JSON dosyasına yaz"""
        try:
            # Prepare data
            dashboard_data = {
                "updated_at": datetime.now().isoformat(),
                "macro_context": self.latest_macro,
                "signals": list(self.latest_signals.values())
            }
            
            # File path: data/dashboard_data.json
            # Use self.DATA_DIR if available, else derive from current file
            base_dir = Path(__file__).parent.parent.parent
            data_file = base_dir / "data" / "dashboard_data.json"
            
            # Ensure dir exists
            if not data_file.parent.exists():
                data_file.parent.mkdir(parents=True, exist_ok=True)
                
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, indent=4, default=str)
                
        except Exception as e:
            logger.error(f"Dashboard update error: {e}")

# Global instance
_engine: Optional[EarlySignalEngine] = None


async def get_early_signal_engine() -> EarlySignalEngine:
    """Get or create Early Signal Engine"""
    global _engine
    if _engine is None:
        _engine = EarlySignalEngine()
        await _engine.initialize()
    return _engine
