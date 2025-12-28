# -*- coding: utf-8 -*-
"""
DEMIR AI - ONLINE LEARNER
=========================
Feedback Loop verilerini kullanarak model confidence'ını adapte eder.
Self-learning altyapısının core bileşeni.

Nasıl Çalışır:
1. FeedbackDB'den son N trade'i çek
2. Win rate ve regime performance hesapla
3. Base model tahminlerini adjust et
"""
import logging
from typing import Dict, Optional
from src.brain.feedback_db import get_feedback_db

logger = logging.getLogger("ONLINE_LEARNER")


class OnlineLearner:
    """
    Online Learning: Feedback'den öğrenerek prediction'ları düzeltir.
    """
    
    def __init__(self):
        self.feedback_db = get_feedback_db()
        self.regime_weights = {}  # Regime-specific performance cache
        self._min_trades = 10  # Min trades needed for adjustment
    
    def adjust_prediction(self, base_prediction: Dict, current_regime: str = "UNKNOWN") -> Dict:
        """
        Base model tahminini feedback'e göre düzelt.
        
        Args:
            base_prediction: {'action': 'BUY', 'confidence': 0.75, ...}
            current_regime: 'TRENDING_BULL', 'RANGING', etc.
        
        Returns:
            Adjusted prediction with modified confidence
        """
        try:
            # Son 100 trade'i al
            recent_trades = self.feedback_db.get_last_n(100)
            
            if len(recent_trades) < self._min_trades:
                logger.debug(f"Not enough trades ({len(recent_trades)}) for adjustment")
                return base_prediction
            
            # Genel win rate hesapla
            wins = sum(1 for t in recent_trades if t.get('actual_pnl', 0) > 0)
            overall_win_rate = wins / len(recent_trades)
            
            # Regime-specific performance
            regime_trades = [t for t in recent_trades 
                           if t.get('entry_features', {}).get('regime') == current_regime]
            
            if regime_trades and len(regime_trades) >= 5:
                regime_wins = sum(1 for t in regime_trades if t.get('actual_pnl', 0) > 0)
                regime_win_rate = regime_wins / len(regime_trades)
            else:
                regime_win_rate = overall_win_rate
            
            # Confidence adjustment factor
            # Good performance → boost confidence
            # Bad performance → reduce confidence
            if regime_win_rate > 0.7:
                adjustment = 1.15  # +15% confidence
                reason = "high_win_rate"
            elif regime_win_rate > 0.55:
                adjustment = 1.05  # +5% confidence
                reason = "good_performance"
            elif regime_win_rate < 0.35:
                adjustment = 0.75  # -25% confidence
                reason = "poor_performance"
            elif regime_win_rate < 0.45:
                adjustment = 0.85  # -15% confidence
                reason = "below_average"
            else:
                adjustment = 1.0  # No change
                reason = "neutral"
            
            # Apply adjustment
            adjusted = base_prediction.copy()
            original_conf = adjusted.get('confidence', 0.5)
            adjusted['confidence'] = min(0.95, max(0.1, original_conf * adjustment))
            adjusted['online_adjustment'] = {
                'factor': adjustment,
                'reason': reason,
                'regime_win_rate': regime_win_rate,
                'overall_win_rate': overall_win_rate,
                'sample_size': len(recent_trades)
            }
            
            logger.info(f"Online adjustment: {original_conf:.2f} → {adjusted['confidence']:.2f} ({reason})")
            return adjusted
            
        except Exception as e:
            logger.warning(f"Online learning adjustment failed: {e}")
            return base_prediction
    
    def get_regime_stats(self) -> Dict:
        """Get performance by regime for dashboard"""
        return self.feedback_db.get_regime_stats()
    
    def get_overall_stats(self) -> Dict:
        """Get overall performance stats"""
        return self.feedback_db.get_stats()


# Singleton
_online_learner: Optional[OnlineLearner] = None


def get_online_learner() -> OnlineLearner:
    """Get or create OnlineLearner singleton"""
    global _online_learner
    if _online_learner is None:
        _online_learner = OnlineLearner()
    return _online_learner
