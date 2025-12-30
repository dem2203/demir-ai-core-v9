# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - THINKING BRAIN
==============================
Gerçek düşünen AI sistemi. Kural tabanlı değil, öğrenen.

MİMARİ:
1. RL Agent → Öğrenilmiş davranış kararı
2. Claude AI → Mantıksal analiz
3. Rule Engine → İndikatör sinyalleri
4. Memory → Geçmiş benzer durumlar
5. Regime → Piyasa durumuna adaptasyon

KARAR FORMÜLÜ:
final_decision = weighted_fusion(
    rl_decision * 0.35 +
    claude_decision * 0.30 +
    rule_decision * 0.35
) * memory_factor * regime_factor * confidence_calibration
"""
import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("THINKING_BRAIN")


@dataclass
class ThinkingResult:
    """Düşünme sonucu"""
    symbol: str
    direction: str  # LONG, SHORT, BEKLE
    confidence: float  # 0-100
    
    # Ana karar kaynakları
    rl_decision: str
    rl_confidence: float
    claude_decision: str
    claude_confidence: float
    rule_decision: str
    rule_confidence: float
    
    # Modifiers
    memory_factor: float  # 0.5-1.5
    regime_factor: float  # 0.5-1.5
    calibration_factor: float  # 0.5-1.5
    
    # Trade levels (RL veya rules'dan)
    entry_price: float = 0
    stop_loss: float = 0
    take_profit_1: float = 0
    take_profit_2: float = 0
    
    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)
    
    # Meta
    regime: str = "UNKNOWN"
    similar_past_trades: int = 0
    past_win_rate: float = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_actionable(self) -> bool:
        """Sinyal alınabilir mi?"""
        return self.direction in ["LONG", "SHORT"] and self.confidence >= 65


class ThinkingBrain:
    """
    DÜŞÜNEN BEYİN
    
    3 kaynaktan gelen önerileri birleştirir, hafızayı kontrol eder,
    rejime adapte olur ve kalibre edilmiş güvenle karar verir.
    """
    
    # Kaynak ağırlıkları (adaptif - öğrenerek değişir)
    DEFAULT_WEIGHTS = {
        'rl': 0.35,
        'claude': 0.30,
        'rules': 0.35
    }
    
    # Regime bazlı strateji parametreleri
    REGIME_STRATEGIES = {
        'TRENDING_UP': {
            'min_confidence': 60,
            'tp_multiplier': 1.5,
            'position_size': 1.0,
            'prefer_long': True,
            'sl_tight': False
        },
        'TRENDING_DOWN': {
            'min_confidence': 65,
            'tp_multiplier': 1.2,
            'position_size': 0.7,
            'prefer_short': True,
            'sl_tight': True
        },
        'RANGING': {
            'min_confidence': 70,
            'tp_multiplier': 0.8,
            'position_size': 0.5,
            'mean_reversion': True,
            'sl_tight': True
        },
        'HIGH_VOLATILITY': {
            'min_confidence': 75,
            'tp_multiplier': 2.0,
            'position_size': 0.3,
            'wide_stops': True,
            'sl_tight': False
        },
        'UNKNOWN': {
            'min_confidence': 70,
            'tp_multiplier': 1.0,
            'position_size': 0.5,
            'sl_tight': True
        }
    }
    
    def __init__(self):
        self._weights = self.DEFAULT_WEIGHTS.copy()
        self._decision_history: List[Dict] = []
        self._performance_by_source: Dict[str, Dict] = {
            'rl': {'wins': 0, 'total': 0},
            'claude': {'wins': 0, 'total': 0},
            'rules': {'wins': 0, 'total': 0}
        }
        
        # Lazy load modüller
        self._rl_agent = None
        self._adaptive_intel = None
        self._feedback_db = None
        self._llm_brain = None
    
    def _get_rl_agent(self):
        """RL Agent lazy load"""
        if self._rl_agent is None:
            try:
                from src.brain.rl_agent.ppo_agent import RLAgent
                self._rl_agent = RLAgent()
                self._rl_agent.load()  # Pre-trained model yükle
                logger.info("🤖 RL Agent loaded")
            except Exception as e:
                logger.warning(f"RL Agent load failed: {e}")
        return self._rl_agent
    
    def _get_adaptive_intel(self):
        """Adaptive Intelligence lazy load"""
        if self._adaptive_intel is None:
            try:
                from src.brain.adaptive_intel import AdaptiveIntelligence
                self._adaptive_intel = AdaptiveIntelligence()
                logger.info("🧠 Adaptive Intelligence loaded")
            except Exception as e:
                logger.warning(f"Adaptive Intel load failed: {e}")
        return self._adaptive_intel
    
    def _get_feedback_db(self):
        """Feedback DB lazy load"""
        if self._feedback_db is None:
            try:
                from src.brain.feedback_db import get_feedback_db
                self._feedback_db = get_feedback_db()
                logger.info("📊 Feedback DB loaded")
            except Exception as e:
                logger.warning(f"Feedback DB load failed: {e}")
        return self._feedback_db
    
    async def think(
        self,
        symbol: str,
        market_state: np.ndarray,
        rule_analysis: Dict,
        claude_analysis: Optional[Dict] = None
    ) -> ThinkingResult:
        """
        ANA DÜŞÜNME FONKSİYONU
        
        Args:
            symbol: Trading pair (BTCUSDT)
            market_state: 37-boyutlu state vektörü (RL için)
            rule_analysis: Full AI Collector sonucu
            claude_analysis: LLM Brain sonucu (opsiyonel)
            
        Returns:
            ThinkingResult
        """
        logger.info(f"🧠 Thinking about {symbol}...")
        
        reasoning = []
        risk_warnings = []
        
        # =================================================================
        # 1. REGIME DETECTION
        # =================================================================
        regime = "UNKNOWN"
        adaptive = self._get_adaptive_intel()
        if adaptive:
            try:
                regime = adaptive.detect_regime(None, 
                    atr_pct=rule_analysis.get('atr_pct', 2),
                    volume_ratio=rule_analysis.get('volume_ratio', 1)
                )
                reasoning.append(f"📊 Piyasa Rejimi: {regime}")
            except Exception as e:
                logger.debug(f"Regime detection error: {e}")
        
        strategy = self.REGIME_STRATEGIES.get(regime, self.REGIME_STRATEGIES['UNKNOWN'])
        
        # =================================================================
        # 2. RL AGENT DECISION
        # =================================================================
        rl_decision = "BEKLE"
        rl_confidence = 0.0
        
        rl_agent = self._get_rl_agent()
        if rl_agent and rl_agent.model is not None:
            try:
                action, confidence = rl_agent.predict(market_state)
                rl_confidence = confidence
                
                if action == 1:
                    rl_decision = "LONG"
                elif action == 2:
                    rl_decision = "SHORT"
                else:
                    rl_decision = "BEKLE"
                
                reasoning.append(f"🤖 RL Agent: {rl_decision} ({rl_confidence:.0f}%)")
            except Exception as e:
                logger.debug(f"RL predict error: {e}")
                rl_confidence = 0
        
        # =================================================================
        # 3. CLAUDE AI DECISION
        # =================================================================
        claude_decision = "BEKLE"
        claude_confidence = 0.0
        
        if claude_analysis:
            claude_direction = claude_analysis.get('direction', 'BEKLE')
            if claude_direction in ['BUY', 'LONG']:
                claude_decision = "LONG"
            elif claude_direction in ['SELL', 'SHORT']:
                claude_decision = "SHORT"
            else:
                claude_decision = "BEKLE"
            
            claude_confidence = claude_analysis.get('confidence', 0)
            reasoning.append(f"🧠 Claude AI: {claude_decision} ({claude_confidence:.0f}%)")
        
        # =================================================================
        # 4. RULE ENGINE DECISION
        # =================================================================
        rule_decision = "BEKLE"
        rule_confidence = 0.0
        
        bullish = rule_analysis.get('total_bullish_signals', 0)
        bearish = rule_analysis.get('total_bearish_signals', 0)
        
        if bullish >= 5 and bullish > bearish + 2:
            rule_decision = "LONG"
            rule_confidence = min(90, 50 + bullish * 5)
        elif bearish >= 5 and bearish > bullish + 2:
            rule_decision = "SHORT"
            rule_confidence = min(90, 50 + bearish * 5)
        else:
            rule_decision = "BEKLE"
            rule_confidence = 50
        
        reasoning.append(f"📈 Rules: {rule_decision} (Bull:{bullish}, Bear:{bearish})")
        
        # =================================================================
        # 5. MEMORY CHECK (Geçmiş Benzer Durumlar)
        # =================================================================
        memory_factor = 1.0
        similar_past = 0
        past_win_rate = 0.5
        
        feedback_db = self._get_feedback_db()
        if feedback_db:
            try:
                # Son 100 trade'i kontrol et
                recent = feedback_db.get_recent_trades(100)
                
                # Aynı rejimde yapılan tradeleri bul
                regime_trades = [t for t in recent if t.get('regime') == regime]
                if regime_trades:
                    similar_past = len(regime_trades)
                    wins = sum(1 for t in regime_trades if t.get('actual_pnl', 0) > 0)
                    past_win_rate = wins / len(regime_trades) if regime_trades else 0.5
                    
                    if past_win_rate < 0.4:
                        memory_factor = 0.7
                        risk_warnings.append(f"⚠️ Bu rejimde win rate düşük: %{past_win_rate*100:.0f}")
                    elif past_win_rate > 0.7:
                        memory_factor = 1.2
                        reasoning.append(f"✅ Bu rejimde başarılıyız: %{past_win_rate*100:.0f}")
                    
            except Exception as e:
                logger.debug(f"Memory check error: {e}")
        
        # =================================================================
        # 6. CONFIDENCE CALIBRATION
        # =================================================================
        calibration_factor = 1.0
        
        if adaptive:
            try:
                # Geçmiş performansa göre güven kalibrasyonu
                calibration = adaptive.get_calibrated_confidence(100, regime)
                calibration_factor = calibration / 100  # 0.5 - 1.5 arası
            except:
                pass
        
        # =================================================================
        # 7. WEIGHTED FUSION (Ana Karar)
        # =================================================================
        
        # Her kaynağın yön skoru
        def direction_to_score(d: str) -> float:
            if d == "LONG":
                return 1.0
            elif d == "SHORT":
                return -1.0
            return 0.0
        
        rl_score = direction_to_score(rl_decision) * (rl_confidence / 100)
        claude_score = direction_to_score(claude_decision) * (claude_confidence / 100)
        rule_score = direction_to_score(rule_decision) * (rule_confidence / 100)
        
        # Ağırlıklı toplam
        weighted_score = (
            rl_score * self._weights['rl'] +
            claude_score * self._weights['claude'] +
            rule_score * self._weights['rules']
        )
        
        # Modifiers uygula
        final_score = weighted_score * memory_factor * calibration_factor
        
        # Regime'e göre strateji adaptasyonu
        regime_factor = 1.0
        if strategy.get('prefer_long') and final_score > 0:
            regime_factor = 1.15
        elif strategy.get('prefer_short') and final_score < 0:
            regime_factor = 1.15
        elif strategy.get('mean_reversion'):
            # Range'de aşırı yönleri sev
            if abs(final_score) > 0.5:
                regime_factor = 1.1
        
        final_score *= regime_factor
        
        # =================================================================
        # 8. FINAL DECISION
        # =================================================================
        
        if final_score > 0.25:
            direction = "LONG"
        elif final_score < -0.25:
            direction = "SHORT"
        else:
            direction = "BEKLE"
        
        # Confidence hesapla
        base_confidence = abs(final_score) * 100
        
        # Kaynak uyumu bonus
        sources_agree = sum([
            1 if rl_decision == direction else 0,
            1 if claude_decision == direction else 0,
            1 if rule_decision == direction else 0
        ])
        
        if sources_agree >= 3:
            base_confidence += 15
            reasoning.append("✨ TÜM KAYNAKLAR HEMFİKİR!")
        elif sources_agree >= 2:
            base_confidence += 5
        
        final_confidence = min(95, max(0, base_confidence))
        
        # =================================================================
        # 9. REGIME THRESHOLD CHECK
        # =================================================================
        min_confidence = strategy['min_confidence']
        if final_confidence < min_confidence and direction != "BEKLE":
            risk_warnings.append(f"⚠️ {regime} rejiminde min güven: %{min_confidence}")
            if final_confidence < min_confidence - 10:
                direction = "BEKLE"
                reasoning.append(f"❌ Güven yeterli değil ({final_confidence:.0f}% < {min_confidence}%)")
        
        # =================================================================
        # 10. BUILD RESULT
        # =================================================================
        
        # Entry/TP/SL (rule_analysis'tan veya hesaplanmış)
        entry_price = rule_analysis.get('current_price', 0)
        
        result = ThinkingResult(
            symbol=symbol,
            direction=direction,
            confidence=final_confidence,
            
            rl_decision=rl_decision,
            rl_confidence=rl_confidence,
            claude_decision=claude_decision,
            claude_confidence=claude_confidence,
            rule_decision=rule_decision,
            rule_confidence=rule_confidence,
            
            memory_factor=memory_factor,
            regime_factor=regime_factor,
            calibration_factor=calibration_factor,
            
            entry_price=entry_price,
            
            reasoning=reasoning,
            risk_warnings=risk_warnings,
            
            regime=regime,
            similar_past_trades=similar_past,
            past_win_rate=past_win_rate
        )
        
        # Karar kaydet (hafıza için)
        self._record_decision(result)
        
        logger.info(f"🧠 Decision: {direction} ({final_confidence:.0f}%) - {regime}")
        
        return result
    
    def _record_decision(self, result: ThinkingResult):
        """Kararı hafızaya kaydet"""
        self._decision_history.append({
            'symbol': result.symbol,
            'direction': result.direction,
            'confidence': result.confidence,
            'regime': result.regime,
            'timestamp': result.timestamp.isoformat(),
            'rl': result.rl_decision,
            'claude': result.claude_decision,
            'rules': result.rule_decision
        })
        
        # Son 500 karar tut
        self._decision_history = self._decision_history[-500:]
    
    def learn_from_outcome(self, decision_timestamp: str, pnl: float, was_correct: bool):
        """
        Trade sonucundan öğren.
        Hangi kaynak doğru bildi ise ağırlığını artır.
        """
        # İlgili kararı bul
        decision = None
        for d in self._decision_history:
            if d['timestamp'] == decision_timestamp:
                decision = d
                break
        
        if not decision:
            return
        
        final_direction = decision['direction']
        
        # Her kaynağın doğruluğunu kontrol et
        for source in ['rl', 'claude', 'rules']:
            source_decision = decision.get(source, 'BEKLE')
            was_source_correct = (source_decision == final_direction) and was_correct
            
            self._performance_by_source[source]['total'] += 1
            if was_source_correct:
                self._performance_by_source[source]['wins'] += 1
        
        # Ağırlıkları güncelle (her 10 trade'de bir)
        total_trades = self._performance_by_source['rl']['total']
        if total_trades > 0 and total_trades % 10 == 0:
            self._update_weights()
    
    def _update_weights(self):
        """Kaynak ağırlıklarını performansa göre güncelle"""
        win_rates = {}
        for source, stats in self._performance_by_source.items():
            if stats['total'] > 5:
                win_rates[source] = stats['wins'] / stats['total']
            else:
                win_rates[source] = 0.5
        
        # Win rate'e göre ağırlık hesapla
        total_wr = sum(win_rates.values())
        if total_wr > 0:
            for source in self._weights:
                new_weight = win_rates[source] / total_wr
                # Smooth update (ani değişim olmasın)
                self._weights[source] = 0.7 * self._weights[source] + 0.3 * new_weight
        
        logger.info(f"📊 Weights updated: {self._weights}")
    
    def get_state_vector(self, rule_analysis: Dict) -> np.ndarray:
        """
        RL Agent için 37-boyutlu state vektörü oluştur.
        """
        state = np.zeros(37, dtype=np.float32)
        
        try:
            # 0-4: Price features
            state[0] = rule_analysis.get('rsi', 50) / 100
            state[1] = rule_analysis.get('macd_histogram', 0) / 100
            state[2] = rule_analysis.get('adx_value', 25) / 100
            state[3] = (rule_analysis.get('bb_position', 'MIDDLE') == 'LOWER') * 1.0
            state[4] = (rule_analysis.get('bb_position', 'MIDDLE') == 'UPPER') * 1.0
            
            # 5-9: Momentum
            state[5] = (rule_analysis.get('stoch_signal', 'NEUTRAL') == 'OVERSOLD') * 1.0
            state[6] = (rule_analysis.get('stoch_signal', 'NEUTRAL') == 'OVERBOUGHT') * 1.0
            state[7] = (rule_analysis.get('macd_signal', 'NEUTRAL') == 'BULLISH') * 1.0
            state[8] = (rule_analysis.get('macd_signal', 'NEUTRAL') == 'BEARISH') * 1.0
            state[9] = (rule_analysis.get('adx_trend', 'WEAK') == 'STRONG') * 1.0
            
            # 10-14: LSTM
            state[10] = (rule_analysis.get('lstm_direction', 'NEUTRAL') == 'UP') * 1.0
            state[11] = (rule_analysis.get('lstm_direction', 'NEUTRAL') == 'DOWN') * 1.0
            state[12] = rule_analysis.get('lstm_confidence', 0) / 100
            state[13] = rule_analysis.get('lstm_change_pct', 0) / 10
            
            # 15-19: Elliott Wave
            state[14] = (rule_analysis.get('elliott_direction', 'NEUTRAL') == 'UP') * 1.0
            state[15] = (rule_analysis.get('elliott_direction', 'NEUTRAL') == 'DOWN') * 1.0
            state[16] = rule_analysis.get('elliott_confidence', 0)
            
            # 17-19: Harmonic
            state[17] = (rule_analysis.get('harmonic_bullish', False)) * 1.0
            state[18] = rule_analysis.get('harmonic_confidence', 0)
            
            # 20-24: MTF
            state[19] = (rule_analysis.get('mtf_confluence', 'MIXED') == 'BULLISH') * 1.0
            state[20] = (rule_analysis.get('mtf_confluence', 'MIXED') == 'BEARISH') * 1.0
            state[21] = (rule_analysis.get('trend_1h', 'NEUTRAL') == 'UP') * 1.0
            state[22] = (rule_analysis.get('trend_4h', 'NEUTRAL') == 'UP') * 1.0
            state[23] = (rule_analysis.get('trend_1d', 'NEUTRAL') == 'UP') * 1.0
            
            # 25-29: On-Chain
            state[24] = rule_analysis.get('mvrv', 1) - 1  # -0.5 to 2
            state[25] = rule_analysis.get('nupl', 0)
            state[26] = rule_analysis.get('exchange_netflow', 0) / 10000
            
            # 30-34: Macro
            state[27] = (rule_analysis.get('dxy_trend', 'NEUTRAL') == 'DOWN') * 1.0
            state[28] = (rule_analysis.get('vix_level', 'NORMAL') == 'HIGH') * 1.0
            state[29] = rule_analysis.get('news_score', 50) / 100
            
            # 35-36: Signals
            state[30] = rule_analysis.get('total_bullish_signals', 0) / 10
            state[31] = rule_analysis.get('total_bearish_signals', 0) / 10
            
            # Padding
            state[32:37] = 0
            
        except Exception as e:
            logger.debug(f"State vector error: {e}")
        
        return state


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_thinking_brain: Optional[ThinkingBrain] = None

def get_thinking_brain() -> ThinkingBrain:
    """Get or create thinking brain instance."""
    global _thinking_brain
    if _thinking_brain is None:
        _thinking_brain = ThinkingBrain()
    return _thinking_brain


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        brain = get_thinking_brain()
        
        # Mock data
        rule_analysis = {
            'rsi': 35,
            'macd_signal': 'BULLISH',
            'total_bullish_signals': 7,
            'total_bearish_signals': 2,
            'lstm_direction': 'UP',
            'lstm_confidence': 65,
            'mtf_confluence': 'BULLISH',
            'current_price': 94000
        }
        
        state = brain.get_state_vector(rule_analysis)
        
        result = await brain.think(
            symbol="BTCUSDT",
            market_state=state,
            rule_analysis=rule_analysis,
            claude_analysis={'direction': 'BUY', 'confidence': 70}
        )
        
        print(f"\n🧠 THINKING BRAIN RESULT")
        print(f"Direction: {result.direction}")
        print(f"Confidence: {result.confidence:.0f}%")
        print(f"Regime: {result.regime}")
        print(f"\nReasoning:")
        for r in result.reasoning:
            print(f"  {r}")
        print(f"\nRisk Warnings:")
        for w in result.risk_warnings:
            print(f"  {w}")
    
    asyncio.run(test())
