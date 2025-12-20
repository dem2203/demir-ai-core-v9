# -*- coding: utf-8 -*-
"""
DEMIR AI - Living AI Brain
Düşünen, öğrenen, gelişen canlı kripto trader yapay zekası.

PHASE 103-107: TRUE LIVING AI TRADER
- LSTM modelleri ile pattern recognition
- RL agent ile karar verme
- Self-evaluation ile öğrenme
- Adaptive strategy ile strateji değiştirme
- Continuous learning loop

Bu dosya TÜM ML bileşenlerini birleştirir ve GERÇEK karar verici olarak çalışır.
"""
import logging
import numpy as np
import pandas as pd
import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import requests

logger = logging.getLogger("LIVING_AI_BRAIN")


@dataclass
class AIDecision:
    """AI kararı."""
    action: str  # 'LONG', 'SHORT', 'HOLD'
    confidence: float  # 0-100
    reasoning: Dict  # Hangi modüller neden
    lstm_prediction: Dict  # LSTM model çıktısı
    rl_action: Optional[int] = None  # RL agent çıktısı (varsa)
    pattern_match: Optional[str] = None  # Tespit edilen pattern
    risk_score: float = 50.0  # Risk skoru
    timestamp: datetime = field(default_factory=datetime.now)


class LivingAIBrain:
    """
    Canlı Yapay Zeka Beyni
    
    İndikatör değil, gerçek ML modelleri ile karar verir.
    Geçmişten öğrenir, kendini değerlendirir, strateji değiştirir.
    """
    
    BRAIN_STATE_FILE = "living_brain_state.json"
    DECISION_HISTORY_FILE = "decision_history.json"
    
    # Learning parameters
    MIN_DECISIONS_FOR_LEARNING = 20
    LEARNING_WINDOW = 100  # Son N karar
    
    def __init__(self):
        self.lstm_models = {}
        self.scalers = {}
        self.signal_combiner = None
        self.rl_agent = None
        
        # State
        self.decision_history: List[Dict] = []
        self.performance_stats: Dict = {
            'total_decisions': 0,
            'correct_decisions': 0,
            'accuracy': 0,
            'avg_confidence': 0,
            'best_pattern': None,
            'worst_pattern': None
        }
        
        # Market regime
        self.current_regime: str = 'UNKNOWN'  # BULL, BEAR, RANGE
        self.regime_confidence: float = 0
        
        # Initialize
        self._load_state()
        self._load_models()
        
        logger.info("🧠 Living AI Brain initialized - GERÇEK YAPAY ZEKA AKTİF")
    
    def _load_state(self):
        """State yükle."""
        try:
            if os.path.exists(self.BRAIN_STATE_FILE):
                with open(self.BRAIN_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.performance_stats = data.get('performance', self.performance_stats)
                    self.current_regime = data.get('regime', 'UNKNOWN')
                    self.regime_confidence = data.get('regime_confidence', 0)
            
            if os.path.exists(self.DECISION_HISTORY_FILE):
                with open(self.DECISION_HISTORY_FILE, 'r') as f:
                    self.decision_history = json.load(f)[-self.LEARNING_WINDOW:]
                    
        except Exception as e:
            logger.warning(f"Brain state load failed: {e}")
    
    def _save_state(self):
        """State kaydet."""
        try:
            with open(self.BRAIN_STATE_FILE, 'w') as f:
                json.dump({
                    'performance': self.performance_stats,
                    'regime': self.current_regime,
                    'regime_confidence': self.regime_confidence,
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
            
            with open(self.DECISION_HISTORY_FILE, 'w') as f:
                json.dump(self.decision_history[-self.LEARNING_WINDOW:], f, indent=2)
                
        except Exception as e:
            logger.warning(f"Brain state save failed: {e}")
    
    def _load_models(self):
        """Eğitilmiş modelleri yükle."""
        # LSTM modelleri
        try:
            from src.brain.models.lstm_trend import LSTMTrendPredictor
            
            for symbol in ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT']:
                try:
                    self.lstm_models[symbol] = LSTMTrendPredictor(symbol=symbol)
                    logger.info(f"✅ LSTM model loaded: {symbol}")
                except Exception as e:
                    logger.warning(f"LSTM {symbol} load failed: {e}")
                    
        except ImportError:
            logger.warning("LSTM models not available")
        
        # Signal Combiner
        try:
            from src.brain.signal_combiner import SignalCombinerModel
            self.signal_combiner = SignalCombinerModel()
            logger.info("✅ Signal Combiner loaded")
        except Exception as e:
            logger.warning(f"Signal Combiner load failed: {e}")
        
        # RL Agent - FIXED: state_dim must match StateVectorBuilder (37)
        try:
            from src.brain.rl_agent.ppo_agent import RLAgent
            self.rl_agent = RLAgent()  # Uses storage_dir for loading
            
            # Try to load trained weights
            loaded = self.rl_agent.load("ppo_btcusdt_v1")
            if loaded:
                logger.info("✅ RL Agent loaded with trained weights")
            else:
                logger.info("⚠️ RL Agent initialized without trained weights")
                
        except Exception as e:
            logger.warning(f"RL Agent load failed: {e}")
            self.rl_agent = None
    
    async def think(self, symbol: str, market_data: Dict) -> AIDecision:
        """
        ANA DÜŞÜNME FONKSİYONU
        
        Tüm ML modellerini çalıştırır ve karar verir.
        
        Args:
            symbol: Trading pair (BTCUSDT)
            market_data: Piyasa verileri (price, volume, indicators, etc)
        
        Returns:
            AIDecision: Yapay zeka kararı
        """
        reasoning = {}
        votes = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        
        # 1. LSTM Pattern Recognition
        lstm_pred = await self._get_lstm_prediction(symbol, market_data)
        reasoning['lstm'] = lstm_pred
        
        if lstm_pred.get('direction') == 'UP':
            votes['LONG'] += lstm_pred.get('confidence', 50)
        elif lstm_pred.get('direction') == 'DOWN':
            votes['SHORT'] += lstm_pred.get('confidence', 50)
        else:
            votes['HOLD'] += 30
        
        # 2. Signal Combiner (35 modül konsensüsü)
        combiner_signal = await self._get_combiner_prediction(market_data)
        reasoning['combiner'] = combiner_signal
        
        if combiner_signal.get('action') == 'LONG':
            votes['LONG'] += combiner_signal.get('confidence', 50)
        elif combiner_signal.get('action') == 'SHORT':
            votes['SHORT'] += combiner_signal.get('confidence', 50)
        else:
            votes['HOLD'] += 20
        
        # 3. RL Agent (varsa)
        rl_action = None
        if self.rl_agent:
            rl_action = await self._get_rl_decision(symbol, market_data)
            reasoning['rl_agent'] = {'action': rl_action}
            
            if rl_action == 0:  # LONG
                votes['LONG'] += 40
            elif rl_action == 2:  # SHORT
                votes['SHORT'] += 40
            else:  # HOLD
                votes['HOLD'] += 20
        
        # 4. Market Regime Adaptation
        regime_multiplier = self._get_regime_multiplier()
        reasoning['regime'] = {
            'current': self.current_regime,
            'confidence': self.regime_confidence,
            'multiplier': regime_multiplier
        }
        
        # Apply regime to votes
        if self.current_regime == 'BULL':
            votes['LONG'] *= regime_multiplier
        elif self.current_regime == 'BEAR':
            votes['SHORT'] *= regime_multiplier
        
        # 5. Pattern Detection
        pattern = await self._detect_pattern(symbol, market_data)
        reasoning['pattern'] = pattern
        
        if pattern:
            if 'bullish' in pattern.lower():
                votes['LONG'] += 25
            elif 'bearish' in pattern.lower():
                votes['SHORT'] += 25
        
        # 6. Risk Assessment
        risk_score = await self._calculate_risk(symbol, market_data)
        reasoning['risk'] = {'score': risk_score}
        
        # Final Decision
        total_votes = sum(votes.values())
        if total_votes == 0:
            total_votes = 1
        
        # Normalize and find winner
        normalized = {k: (v / total_votes) * 100 for k, v in votes.items()}
        action = max(votes, key=votes.get)
        confidence = normalized[action]
        
        # Self-evaluation adjustment
        historical_accuracy = await self._get_action_accuracy(action)
        if historical_accuracy is not None:
            confidence = confidence * 0.7 + historical_accuracy * 0.3
        
        decision = AIDecision(
            action=action,
            confidence=min(confidence, 95),  # Cap at 95%
            reasoning=reasoning,
            lstm_prediction=lstm_pred,
            rl_action=rl_action,
            pattern_match=pattern,
            risk_score=risk_score
        )
        
        # Record decision for learning
        await self._record_decision(symbol, decision, market_data)
        
        logger.info(f"🧠 AI DECISION: {action} {confidence:.0f}% | LSTM: {lstm_pred.get('direction')} | RL: {rl_action} | Pattern: {pattern}")
        
        return decision
    
    async def _get_lstm_prediction(self, symbol: str, market_data: Dict) -> Dict:
        """LSTM model ile tahmin."""
        try:
            if symbol not in self.lstm_models:
                return {'direction': 'NEUTRAL', 'confidence': 30, 'error': 'No model'}
            
            model = self.lstm_models[symbol]
            
            # Get recent price data
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1h', 'limit': 30},
                timeout=10
            )
            
            if resp.status_code != 200:
                return {'direction': 'NEUTRAL', 'confidence': 30, 'error': 'API failed'}
            
            klines = resp.json()
            df = pd.DataFrame({
                'close': [float(k[4]) for k in klines],
                'volume': [float(k[5]) for k in klines],
                'high': [float(k[2]) for k in klines],
                'low': [float(k[3]) for k in klines]
            })
            
            prediction = model.predict(df)
            
            return {
                'direction': prediction.get('direction', 'NEUTRAL'),
                'confidence': prediction.get('confidence', 50),
                'probabilities': prediction.get('probabilities', {})
            }
            
        except Exception as e:
            logger.debug(f"LSTM prediction failed: {e}")
            return {'direction': 'NEUTRAL', 'confidence': 30, 'error': str(e)}
    
    async def _get_combiner_prediction(self, market_data: Dict) -> Dict:
        """Signal Combiner ile tahmin."""
        try:
            if not self.signal_combiner:
                return {'action': 'HOLD', 'confidence': 30}
            
            signal = self.signal_combiner.predict(market_data)
            
            return {
                'action': signal.action if hasattr(signal, 'action') else 'HOLD',
                'confidence': signal.confidence if hasattr(signal, 'confidence') else 50,
                'top_bullish': signal.top_bullish if hasattr(signal, 'top_bullish') else [],
                'top_bearish': signal.top_bearish if hasattr(signal, 'top_bearish') else []
            }
            
        except Exception as e:
            logger.debug(f"Combiner prediction failed: {e}")
            return {'action': 'HOLD', 'confidence': 30, 'error': str(e)}
    
    async def _get_rl_decision(self, symbol: str, market_data: Dict) -> Optional[int]:
        """RL Agent ile karar."""
        try:
            if not self.rl_agent:
                return None
            
            # Prepare state for RL agent
            state = self._prepare_rl_state(market_data)
            
            if state is not None:
                action, _ = self.rl_agent.select_action(state)
                return action  # 0=LONG, 1=HOLD, 2=SHORT
                
        except Exception as e:
            logger.debug(f"RL decision failed: {e}")
        
        return None
    
    def _prepare_rl_state(self, market_data: Dict) -> Optional[np.ndarray]:
        """RL agent için state hazırla - FIXED: 37 features matching StateVectorBuilder."""
        try:
            # 37 features expected by RL agent (matches StateVectorBuilder.STATE_DIM)
            features = [
                # LSTM predictions (3)
                market_data.get('lstm_up_prob', 0.33),
                market_data.get('lstm_neutral_prob', 0.34),
                market_data.get('lstm_down_prob', 0.33),
                
                # Fractal analysis (3)
                market_data.get('fractal_trend', 0),
                market_data.get('fractal_strength', 0.5),
                market_data.get('fractal_distance', 0),
                
                # Order book (5)
                market_data.get('bid_volume_ratio', 0.5),
                market_data.get('ask_volume_ratio', 0.5),
                market_data.get('orderbook_imbalance', 0),
                market_data.get('bid_wall_distance', 0),
                market_data.get('ask_wall_distance', 0),
                
                # Correlation (4)
                market_data.get('eth_btc_ratio', 0.05),
                market_data.get('btc_dominance', 50) / 100,
                market_data.get('gold_btc_ratio', 0.02),
                market_data.get('sp500_btc_ratio', 0.5),
                
                # Funding and derivatives (3)
                market_data.get('funding_rate', 0) * 100,  # Scale to reasonable range
                market_data.get('long_short_ratio', 1),
                market_data.get('open_interest_change', 0),
                
                # Volatility (5)
                market_data.get('atr', 0) / 1000,  # Normalize
                market_data.get('bb_position', 0.5),
                market_data.get('volatility_pct', 2) / 10,
                market_data.get('price_change_1h', 0) / 10,
                market_data.get('price_change_24h', 0) / 10,
                
                # Anomaly detection (3)
                market_data.get('volume_anomaly', 0),
                market_data.get('price_anomaly', 0),
                market_data.get('flow_anomaly', 0),
                
                # Macro (5)
                market_data.get('macro_score', 0) / 100,
                market_data.get('fear_greed_index', 50) / 100,
                market_data.get('vix', 20) / 100,
                market_data.get('dxy', 100) / 150,
                market_data.get('interest_rate', 5) / 10,
                
                # Position info (3)
                1.0 if market_data.get('in_position', False) else 0.0,
                market_data.get('position_pnl', 0) / 10,
                market_data.get('position_duration', 0) / 24,  # Hours normalized
                
                # Performance (3)
                market_data.get('rsi', 50) / 100,
                market_data.get('macd', 0) / 1000,
                market_data.get('stoch_k', 50) / 100
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.debug(f"RL state prep failed: {e}")
            return None
    
    def _get_regime_multiplier(self) -> float:
        """Market regime için çarpan."""
        base = 1.0
        
        if self.current_regime in ['BULL', 'BEAR']:
            # High confidence = stronger multiplier
            base = 1.0 + (self.regime_confidence / 100) * 0.3  # Max 1.3x
        
        return base
    
    async def _detect_pattern(self, symbol: str, market_data: Dict) -> Optional[str]:
        """Pattern recognition."""
        try:
            from src.brain.candle_patterns import CandlePatternRecognizer
            
            recognizer = CandlePatternRecognizer()
            
            # Get recent candles
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1h', 'limit': 10},
                timeout=5
            )
            
            if resp.status_code == 200:
                klines = resp.json()
                df = pd.DataFrame({
                    'open': [float(k[1]) for k in klines],
                    'high': [float(k[2]) for k in klines],
                    'low': [float(k[3]) for k in klines],
                    'close': [float(k[4]) for k in klines],
                    'volume': [float(k[5]) for k in klines]
                })
                
                result = recognizer.analyze(df)
                if result.get('patterns'):
                    return result['patterns'][0]['name']
                    
        except Exception as e:
            logger.debug(f"Pattern detection failed: {e}")
        
        return None
    
    async def _calculate_risk(self, symbol: str, market_data: Dict) -> float:
        """Risk skoru hesapla."""
        risk = 50  # Base risk
        
        # Volatility risk
        atr = market_data.get('atr', 0)
        if atr > 2000:  # High ATR for BTC
            risk += 20
        
        # Position risk
        funding = market_data.get('funding_rate', 0)
        if abs(funding) > 0.05:
            risk += 15
        
        # Crowd risk
        ls_ratio = market_data.get('long_short_ratio', 1)
        if ls_ratio > 1.5 or ls_ratio < 0.67:
            risk += 10
        
        return min(risk, 100)
    
    async def _get_action_accuracy(self, action: str) -> Optional[float]:
        """Geçmiş performansa göre bu action'ın doğruluk oranı."""
        if len(self.decision_history) < self.MIN_DECISIONS_FOR_LEARNING:
            return None
        
        action_decisions = [d for d in self.decision_history if d.get('action') == action]
        
        if len(action_decisions) < 5:
            return None
        
        correct = sum(1 for d in action_decisions if d.get('was_correct', False))
        return (correct / len(action_decisions)) * 100
    
    async def _record_decision(self, symbol: str, decision: AIDecision, market_data: Dict):
        """Kararı kaydet (öğrenme için)."""
        entry_price = market_data.get('current_price', 0)
        
        self.decision_history.append({
            'timestamp': decision.timestamp.isoformat(),
            'symbol': symbol,
            'action': decision.action,
            'confidence': decision.confidence,
            'entry_price': entry_price,
            'lstm_direction': decision.lstm_prediction.get('direction'),
            'rl_action': decision.rl_action,
            'pattern': decision.pattern_match,
            'risk_score': decision.risk_score,
            'regime': self.current_regime,
            'was_correct': None,  # Will be updated later
            'profit_pct': None
        })
        
        self.performance_stats['total_decisions'] += 1
        
        self._save_state()
    
    async def evaluate_past_decision(self, decision_index: int, exit_price: float):
        """
        Geçmiş kararı değerlendir - ÖĞRENMENİN TEMELİ
        
        Bu fonksiyon her sinyal sonuçlandığında çağrılır ve
        AI'ın kendini değerlendirmesini sağlar.
        """
        if decision_index < 0 or decision_index >= len(self.decision_history):
            return
        
        decision = self.decision_history[decision_index]
        entry_price = decision.get('entry_price', 0)
        
        if entry_price == 0:
            return
        
        # Calculate profit
        profit_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Determine if decision was correct
        action = decision.get('action')
        was_correct = False
        
        if action == 'LONG' and profit_pct > 0.5:
            was_correct = True
        elif action == 'SHORT' and profit_pct < -0.5:
            was_correct = True  # Short was correct if price went down
        elif action == 'HOLD' and abs(profit_pct) < 1:
            was_correct = True  # Hold was correct if price didn't move much
        
        # Update decision
        self.decision_history[decision_index]['was_correct'] = was_correct
        self.decision_history[decision_index]['profit_pct'] = profit_pct
        
        # Update stats
        if was_correct:
            self.performance_stats['correct_decisions'] += 1
        
        total = self.performance_stats['total_decisions']
        correct = self.performance_stats['correct_decisions']
        
        if total > 0:
            self.performance_stats['accuracy'] = (correct / total) * 100
        
        self._save_state()
        
        logger.info(f"📊 SELF-EVAL: {action} → {'✅ CORRECT' if was_correct else '❌ WRONG'} | {profit_pct:+.2f}% | Overall: {self.performance_stats['accuracy']:.1f}%")
    
    async def adapt_strategy(self):
        """
        STRATEJİ ADAPTASYONU
        
        Geçmiş performansa göre strateji parametrelerini değiştirir.
        """
        if len(self.decision_history) < self.MIN_DECISIONS_FOR_LEARNING:
            return
        
        # Analyze what works
        recent = self.decision_history[-50:]
        
        # Find best performing patterns
        pattern_performance = {}
        for d in recent:
            pattern = d.get('pattern')
            if pattern:
                if pattern not in pattern_performance:
                    pattern_performance[pattern] = {'correct': 0, 'total': 0}
                pattern_performance[pattern]['total'] += 1
                if d.get('was_correct'):
                    pattern_performance[pattern]['correct'] += 1
        
        # Find best/worst patterns
        best_pattern = None
        worst_pattern = None
        best_rate = 0
        worst_rate = 100
        
        for pattern, stats in pattern_performance.items():
            if stats['total'] >= 3:
                rate = (stats['correct'] / stats['total']) * 100
                if rate > best_rate:
                    best_rate = rate
                    best_pattern = pattern
                if rate < worst_rate:
                    worst_rate = rate
                    worst_pattern = pattern
        
        self.performance_stats['best_pattern'] = best_pattern
        self.performance_stats['worst_pattern'] = worst_pattern
        
        self._save_state()
        
        logger.info(f"🔄 STRATEGY ADAPTED: Best: {best_pattern} ({best_rate:.0f}%) | Worst: {worst_pattern} ({worst_rate:.0f}%)")
    
    async def update_regime(self, symbol: str = 'BTCUSDT'):
        """Market rejimini güncelle."""
        try:
            from src.brain.regime_adapter import get_adapter
            adapter = get_adapter()
            regime, confidence = adapter.detect_regime(symbol)
            
            self.current_regime = regime
            self.regime_confidence = confidence
            
            self._save_state()
            
        except Exception as e:
            logger.debug(f"Regime update failed: {e}")
    
    def format_decision_for_telegram(self, decision: AIDecision, symbol: str, current_price: float = 0) -> str:
        """Telegram formatında karar - ANLAŞILIR TÜRKÇE."""
        
        # Türkçe çeviriler
        action_tr = {
            'LONG': 'AL',
            'SHORT': 'SAT',
            'HOLD': 'BEKLE'
        }
        
        action_emoji = {
            'LONG': '🟢',
            'SHORT': '🔴',
            'HOLD': '🟡'
        }
        
        regime_tr = {
            'BULL': 'Yükseliş',
            'BEAR': 'Düşüş',
            'RANGE': 'Yatay',
            'VOLATILE': 'Dalgalı',
            'UNKNOWN': 'Belirsiz'
        }
        
        lstm_tr = {
            'UP': 'Yukarı',
            'DOWN': 'Aşağı',
            'NEUTRAL': 'Nötr'
        }
        
        # Pattern Türkçe çevirisi
        pattern_tr = {
            'bullish_engulfing': 'Yutan Boğa',
            'bearish_engulfing': 'Yutan Ayı',
            'hammer': 'Çekiç',
            'shooting_star': 'Kayan Yıldız',
            'doji': 'Doji',
            'morning_star': 'Sabah Yıldızı',
            'evening_star': 'Akşam Yıldızı'
        }
        
        emoji = action_emoji.get(decision.action, '⚪')
        action_text = action_tr.get(decision.action, decision.action)
        
        # Risk seviyesi
        if decision.risk_score < 40:
            risk_text = "Düşük"
        elif decision.risk_score < 60:
            risk_text = "Orta"
        else:
            risk_text = "Yüksek"
        
        # TP/SL hesapla
        if current_price > 0:
            if decision.action == 'LONG':
                tp_price = current_price * 1.035  # +3.5%
                sl_price = current_price * 0.985  # -1.5%
                tp_pct = "+3.5%"
                sl_pct = "-1.5%"
            elif decision.action == 'SHORT':
                tp_price = current_price * 0.965  # -3.5%
                sl_price = current_price * 1.015  # +1.5%
                tp_pct = "+3.5%"
                sl_pct = "-1.5%"
            else:
                tp_price = 0
                sl_price = 0
                tp_pct = ""
                sl_pct = ""
        else:
            tp_price = 0
            sl_price = 0
            tp_pct = ""
            sl_pct = ""
        
        # Confidence stars
        stars = '⭐' * min(5, int(decision.confidence / 20))
        
        # LSTM direction
        lstm_dir = decision.lstm_prediction.get('direction', 'NEUTRAL')
        lstm_text = lstm_tr.get(lstm_dir, lstm_dir)
        lstm_conf = decision.lstm_prediction.get('confidence', 0)
        
        # RL action
        if decision.rl_action == 0:
            rl_text = "AL"
        elif decision.rl_action == 2:
            rl_text = "SAT"
        elif decision.rl_action == 1:
            rl_text = "BEKLE"
        else:
            rl_text = "Yok"
        
        # Pattern
        pattern = decision.pattern_match
        if pattern:
            pattern_text = pattern_tr.get(pattern.lower(), pattern.replace('_', ' ').title())
        else:
            pattern_text = "Yok"
        
        # Regime
        regime_text = regime_tr.get(self.current_regime, self.current_regime)
        
        # TP/SL bölümü
        tp_sl_section = ""
        if tp_price > 0 and sl_price > 0:
            tp_sl_section = f"""
🎯 Kar Al: ${tp_price:,.0f} ({tp_pct})
🛑 Zarar Kes: ${sl_price:,.0f} ({sl_pct})
━━━━━━━━━━━━━━━━━━━━━━"""
        
        msg = f"""
🧠 YAPAY ZEKA KARARI
━━━━━━━━━━━━━━━━━━━━━━
{emoji} {symbol}: {action_text}
📊 Güven: %{decision.confidence:.0f} {stars}
⚠️ Risk Seviyesi: %{decision.risk_score:.0f} ({risk_text})
━━━━━━━━━━━━━━━━━━━━━━
Beyin Analizi:
• Fiyat Tahmini: {lstm_text} (%{lstm_conf:.0f})
• Bot Kararı: {rl_text}
• Mum Formasyonu: {pattern_text}
• Piyasa Trendi: {regime_text}
━━━━━━━━━━━━━━━━━━━━━━{tp_sl_section}
📈 Geçmiş Doğruluk: %{self.performance_stats.get('accuracy', 0):.1f}
📊 Toplam Karar: {self.performance_stats.get('total_decisions', 0)}
━━━━━━━━━━━━━━━━━━━━━━
Bu bir canlı yapay zeka kararıdır
⏰ {decision.timestamp.strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_brain = None

def get_brain() -> LivingAIBrain:
    """Get or create brain instance."""
    global _brain
    if _brain is None:
        _brain = LivingAIBrain()
    return _brain
