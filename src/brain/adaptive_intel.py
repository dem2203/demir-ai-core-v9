"""
DEMIR AI - ADAPTIVE INTELLIGENCE
Kendi kendine öğrenen ve adapte olan sistem

Regime detection, self-optimization, performance tracking
"""

import logging
import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logger = logging.getLogger("ADAPTIVE_INTEL")

class MarketRegime:
    """Piyasa rejimi tanımları"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    UNKNOWN = "UNKNOWN"

class AdaptiveIntelligence:
    """
    ADAPTİF ZEKA SİSTEMİ
    
    1. Market Regime Detection
    2. Strategy Performance Tracking
    3. Dynamic Parameter Adjustment
    4. Confidence Calibration
    5. Memory System (Geçmiş kararlardan öğrenme)
    """
    
    MEMORY_FILE = "src/brain/models/storage/ai_memory.json"
    MAX_MEMORY = 1000  # Son 1000 karar
    
    def __init__(self):
        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.regime_history = deque(maxlen=100)
        self.performance_stats = {
            'total_signals': 0,
            'correct_signals': 0,
            'regime_accuracy': {},
            'strategy_performance': {}
        }
        self.current_regime = MarketRegime.UNKNOWN
        self.confidence_calibration = 1.0  # Güven düzeltme faktörü
        
        # Memory'yi yükle
        self._load_memory()
    
    def _load_memory(self):
        """Geçmiş kararları diskten yükle."""
        try:
            if os.path.exists(self.MEMORY_FILE):
                with open(self.MEMORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.memory = deque(data.get('memory', []), maxlen=self.MAX_MEMORY)
                    self.performance_stats = data.get('performance_stats', self.performance_stats)
                    self.confidence_calibration = data.get('confidence_calibration', 1.0)
                    logger.info(f"📚 Loaded {len(self.memory)} memories from disk")
        except Exception as e:
            logger.warning(f"Could not load memory: {e}")
    
    def _save_memory(self):
        """Memory'yi diske kaydet."""
        try:
            os.makedirs(os.path.dirname(self.MEMORY_FILE), exist_ok=True)
            with open(self.MEMORY_FILE, 'w') as f:
                json.dump({
                    'memory': list(self.memory),
                    'performance_stats': self.performance_stats,
                    'confidence_calibration': self.confidence_calibration,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save memory: {e}")
    
    def detect_regime(self, df, atr_pct: float = None, volume_ratio: float = None) -> Dict:
        """
        Piyasa rejimini tespit et.
        
        Parametreler:
        - ADX > 25: Trending
        - ATR/Price > 3%: High Volatility
        - Volume spike: Breakout potential
        """
        if len(df) < 20:
            return {'regime': MarketRegime.UNKNOWN, 'confidence': 0}
        
        # Temel metrikler hesapla
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Trend yönü
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
        current_price = close[-1]
        
        price_vs_sma20 = (current_price - sma_20) / sma_20
        sma20_vs_sma50 = (sma_20 - sma_50) / sma_50
        
        # Volatilite
        if atr_pct is None:
            price_range = (high[-20:].max() - low[-20:].min()) / current_price
            atr_pct = price_range * 100 / 20
        
        # Volume spike
        if volume_ratio is None:
            avg_volume = np.mean(volume[-20:])
            recent_volume = np.mean(volume[-3:])
            volume_ratio = recent_volume / max(avg_volume, 1)
        
        # Trend gücü (basit momentum)
        momentum = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0
        
        # Rejim belirleme
        confidence = 0.5
        
        if atr_pct > 4:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(0.9, 0.5 + atr_pct * 0.1)
        elif atr_pct < 1.5:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 0.7
        elif volume_ratio > 2 and abs(momentum) > 0.02:
            regime = MarketRegime.BREAKOUT
            confidence = min(0.85, 0.5 + volume_ratio * 0.1)
        elif momentum > 0.03 and price_vs_sma20 > 0.01:
            regime = MarketRegime.TRENDING_UP
            confidence = min(0.85, 0.5 + momentum * 5)
        elif momentum < -0.03 and price_vs_sma20 < -0.01:
            regime = MarketRegime.TRENDING_DOWN
            confidence = min(0.85, 0.5 + abs(momentum) * 5)
        else:
            regime = MarketRegime.RANGING
            confidence = 0.6
        
        # Strategy önerileri
        if regime == MarketRegime.TRENDING_UP:
            strategy = "TREND_FOLLOW_LONG"
            risk_multiplier = 1.2
        elif regime == MarketRegime.TRENDING_DOWN:
            strategy = "TREND_FOLLOW_SHORT_OR_CASH"
            risk_multiplier = 0.8
        elif regime == MarketRegime.HIGH_VOLATILITY:
            strategy = "REDUCE_POSITION_SIZE"
            risk_multiplier = 0.5
        elif regime == MarketRegime.LOW_VOLATILITY:
            strategy = "RANGE_TRADE"
            risk_multiplier = 1.0
        elif regime == MarketRegime.BREAKOUT:
            strategy = "BREAKOUT_FOLLOW"
            risk_multiplier = 1.5
        else:
            strategy = "WAIT"
            risk_multiplier = 0.7
        
        # Regime değişikliği logla
        if regime != self.current_regime:
            logger.info(f"🔄 Regime Change: {self.current_regime} → {regime}")
            self.regime_history.append({
                'from': self.current_regime,
                'to': regime,
                'timestamp': datetime.now().isoformat()
            })
            self.current_regime = regime
        
        return {
            'regime': regime,
            'confidence': round(confidence, 2),
            'strategy': strategy,
            'risk_multiplier': risk_multiplier,
            'metrics': {
                'atr_pct': round(atr_pct, 2),
                'volume_ratio': round(volume_ratio, 2),
                'momentum': round(momentum * 100, 2),
                'price_vs_sma20': round(price_vs_sma20 * 100, 2)
            }
        }
    
    def record_decision(self, 
                        symbol: str,
                        signal: str, 
                        confidence: float,
                        price: float,
                        regime: str,
                        reasoning: Dict = None):
        """
        Kararı hafızaya kaydet.
        Daha sonra sonucu karşılaştırmak için.
        """
        decision = {
            'id': len(self.memory),
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'price': price,
            'regime': regime,
            'reasoning': reasoning or {},
            'outcome': None,  # Sonra doldurulacak
            'outcome_price': None,
            'profit_pct': None
        }
        
        self.memory.append(decision)
        self.performance_stats['total_signals'] += 1
        
        # Her 10 karardan sonra kaydet
        if len(self.memory) % 10 == 0:
            self._save_memory()
        
        return decision['id']
    
    def record_outcome(self, decision_id: int, outcome_price: float, was_correct: bool):
        """
        Kararın sonucunu kaydet ve öğren.
        """
        for decision in self.memory:
            if decision['id'] == decision_id:
                decision['outcome'] = 'CORRECT' if was_correct else 'INCORRECT'
                decision['outcome_price'] = outcome_price
                decision['profit_pct'] = ((outcome_price - decision['price']) / decision['price']) * 100
                
                if was_correct:
                    self.performance_stats['correct_signals'] += 1
                
                # Regime bazlı doğruluk güncelle
                regime = decision['regime']
                if regime not in self.performance_stats['regime_accuracy']:
                    self.performance_stats['regime_accuracy'][regime] = {'correct': 0, 'total': 0}
                
                self.performance_stats['regime_accuracy'][regime]['total'] += 1
                if was_correct:
                    self.performance_stats['regime_accuracy'][regime]['correct'] += 1
                
                self._save_memory()
                return True
        
        return False
    
    def get_calibrated_confidence(self, raw_confidence: float, regime: str = None) -> float:
        """
        Geçmiş performansa göre güven skorunu kalibre et.
        
        Eğer geçmişte çok yanlış yapıldıysa güveni düşür.
        """
        base_calibration = self.confidence_calibration
        
        # Regime bazlı kalibrasyon
        if regime and regime in self.performance_stats['regime_accuracy']:
            regime_stats = self.performance_stats['regime_accuracy'][regime]
            if regime_stats['total'] >= 10:
                regime_accuracy = regime_stats['correct'] / regime_stats['total']
                # Doğruluk %60'ın altındaysa güveni düşür
                if regime_accuracy < 0.6:
                    base_calibration *= 0.8
                elif regime_accuracy > 0.7:
                    base_calibration *= 1.1
        
        calibrated = raw_confidence * min(base_calibration, 1.2)  # Max %20 artış
        return round(min(1.0, max(0.1, calibrated)), 2)
    
    def update_calibration(self):
        """
        Genel güven kalibrasyonunu güncelle.
        """
        if self.performance_stats['total_signals'] < 20:
            return  # Yeterli veri yok
        
        accuracy = self.performance_stats['correct_signals'] / self.performance_stats['total_signals']
        
        # Hedef: %60 doğruluk
        if accuracy < 0.5:
            self.confidence_calibration = 0.7
        elif accuracy < 0.6:
            self.confidence_calibration = 0.85
        elif accuracy > 0.7:
            self.confidence_calibration = 1.1
        else:
            self.confidence_calibration = 1.0
        
        logger.info(f"📊 Calibration updated: {self.confidence_calibration:.2f} (Accuracy: {accuracy:.1%})")
        self._save_memory()
    
    def get_optimal_parameters(self, regime: str) -> Dict:
        """
        Rejime göre optimal parametreler öner.
        """
        params = {
            MarketRegime.TRENDING_UP: {
                'position_size_multiplier': 1.2,
                'stop_loss_atr_mult': 2.5,
                'take_profit_atr_mult': 4.0,
                'min_confidence': 0.5,
                'prefer_signals': ['BUY']
            },
            MarketRegime.TRENDING_DOWN: {
                'position_size_multiplier': 0.8,
                'stop_loss_atr_mult': 2.0,
                'take_profit_atr_mult': 3.0,
                'min_confidence': 0.6,
                'prefer_signals': ['SELL', 'HOLD']
            },
            MarketRegime.HIGH_VOLATILITY: {
                'position_size_multiplier': 0.5,
                'stop_loss_atr_mult': 3.0,
                'take_profit_atr_mult': 5.0,
                'min_confidence': 0.7,
                'prefer_signals': ['HOLD']
            },
            MarketRegime.LOW_VOLATILITY: {
                'position_size_multiplier': 1.0,
                'stop_loss_atr_mult': 1.5,
                'take_profit_atr_mult': 2.0,
                'min_confidence': 0.5,
                'prefer_signals': ['BUY', 'SELL']
            },
            MarketRegime.BREAKOUT: {
                'position_size_multiplier': 1.3,
                'stop_loss_atr_mult': 2.0,
                'take_profit_atr_mult': 5.0,
                'min_confidence': 0.6,
                'prefer_signals': ['BUY', 'SELL']
            },
            MarketRegime.RANGING: {
                'position_size_multiplier': 0.7,
                'stop_loss_atr_mult': 1.5,
                'take_profit_atr_mult': 2.0,
                'min_confidence': 0.6,
                'prefer_signals': ['HOLD']
            }
        }
        
        return params.get(regime, params[MarketRegime.RANGING])
    
    def get_performance_summary(self) -> Dict:
        """
        Performans özeti döndür.
        """
        total = self.performance_stats['total_signals']
        correct = self.performance_stats['correct_signals']
        
        if total == 0:
            return {'accuracy': 0, 'total_signals': 0, 'message': 'No signals yet'}
        
        # Son 50 sinyalin performansı
        recent = list(self.memory)[-50:]
        recent_correct = sum(1 for d in recent if d.get('outcome') == 'CORRECT')
        recent_total = sum(1 for d in recent if d.get('outcome') is not None)
        
        return {
            'total_accuracy': correct / total if total > 0 else 0,
            'recent_accuracy': recent_correct / recent_total if recent_total > 0 else 0,
            'total_signals': total,
            'correct_signals': correct,
            'regime_performance': self.performance_stats['regime_accuracy'],
            'confidence_calibration': self.confidence_calibration,
            'memory_size': len(self.memory)
        }
    
    def should_take_signal(self, signal: str, confidence: float, regime: str) -> Tuple:
        """
        Sinyalin alınıp alınmaması gerektiğine karar ver.
        """
        params = self.get_optimal_parameters(regime)
        calibrated_confidence = self.get_calibrated_confidence(confidence, regime)
        
        # Minimum güven kontrolü
        if calibrated_confidence < params['min_confidence']:
            return False, f"Confidence too low: {calibrated_confidence:.2f} < {params['min_confidence']}"
        
        # Rejime uygun sinyal mi?
        if signal not in params['prefer_signals'] and 'HOLD' not in params['prefer_signals']:
            return False, f"Signal {signal} not preferred in {regime} regime"
        
        return True, f"Signal approved with calibrated confidence {calibrated_confidence:.2f}"


# Type hint için
from typing import Tuple
