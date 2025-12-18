# -*- coding: utf-8 -*-
"""
DEMIR AI - ML Training Enhancer
ML modellerinin sürekli eğitimi ve iyileştirilmesi.

PHASE 112: ML Training Enhancement
- Continuous learning from signals
- Weight optimization
- Model performance tracking
"""
import logging
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("ML_ENHANCER")


class MLTrainingEnhancer:
    """
    ML Eğitim İyileştirici
    
    Sinyal sonuçlarından sürekli öğrenir ve modelleri iyileştirir.
    """
    
    STATE_FILE = "ml_training_state.json"
    
    def __init__(self):
        self.signal_results: List[Dict] = []
        self.module_performance: Dict[str, Dict] = {}
        self.learned_weights: Dict[str, float] = {}
        
        self._load_state()
        logger.info("✅ ML Training Enhancer initialized")
    
    def _load_state(self):
        """State yükle."""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.signal_results = data.get('signal_results', [])[-500:]
                    self.module_performance = data.get('module_performance', {})
                    self.learned_weights = data.get('learned_weights', {})
        except Exception as e:
            logger.debug(f"State load failed: {e}")
    
    def _save_state(self):
        """State kaydet."""
        try:
            with open(self.STATE_FILE, 'w') as f:
                json.dump({
                    'signal_results': self.signal_results[-500:],
                    'module_performance': self.module_performance,
                    'learned_weights': self.learned_weights,
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"State save failed: {e}")
    
    def record_signal(self, signal_data: Dict):
        """
        Sinyal verilerini kaydet.
        
        Args:
            signal_data: {
                'timestamp': datetime,
                'symbol': str,
                'direction': str,
                'confidence': float,
                'modules': {module_name: {direction, confidence}},
                'entry_price': float
            }
        """
        self.signal_results.append({
            **signal_data,
            'timestamp': datetime.now().isoformat() if 'timestamp' not in signal_data else signal_data['timestamp'],
            'result': None,  # Will be updated later
            'exit_price': None,
            'pnl': None
        })
        self._save_state()
    
    def record_result(self, signal_id: int, exit_price: float, result: str):
        """
        Sinyal sonucunu kaydet ve modül performansını güncelle.
        
        Args:
            signal_id: Index of signal in signal_results
            exit_price: Exit price
            result: 'WIN' or 'LOSS'
        """
        if signal_id < 0 or signal_id >= len(self.signal_results):
            return
        
        signal = self.signal_results[signal_id]
        signal['exit_price'] = exit_price
        signal['result'] = result
        
        entry = signal.get('entry_price', 0)
        if entry > 0:
            if signal.get('direction') == 'LONG':
                signal['pnl'] = ((exit_price - entry) / entry) * 100
            else:
                signal['pnl'] = ((entry - exit_price) / entry) * 100
        
        # Modül performansını güncelle
        modules = signal.get('modules', {})
        for module_name, module_signal in modules.items():
            if module_name not in self.module_performance:
                self.module_performance[module_name] = {
                    'total': 0, 'correct': 0, 'accuracy': 0,
                    'long_correct': 0, 'long_total': 0,
                    'short_correct': 0, 'short_total': 0
                }
            
            perf = self.module_performance[module_name]
            perf['total'] += 1
            
            # Modül doğru mu tahmin etti?
            if module_signal.get('direction') == signal.get('direction'):
                is_correct = result == 'WIN'
            else:
                is_correct = result == 'LOSS'  # Opposite direction was right
            
            if is_correct:
                perf['correct'] += 1
            
            perf['accuracy'] = (perf['correct'] / perf['total']) * 100
            
            # Long/Short breakdown
            if signal.get('direction') == 'LONG':
                perf['long_total'] += 1
                if result == 'WIN':
                    perf['long_correct'] += 1
            else:
                perf['short_total'] += 1
                if result == 'WIN':
                    perf['short_correct'] += 1
        
        self._save_state()
        
        # Ağırlıkları güncelle
        self._update_weights()
    
    def _update_weights(self):
        """Modül ağırlıklarını performansa göre güncelle."""
        if len(self.signal_results) < 20:
            return
        
        total_accuracy = sum(p.get('accuracy', 50) for p in self.module_performance.values())
        
        if total_accuracy == 0:
            return
        
        for module_name, perf in self.module_performance.items():
            # Accuracy based weight
            accuracy = perf.get('accuracy', 50)
            
            # Normalize to 0-1 range relative to average
            avg_accuracy = total_accuracy / len(self.module_performance)
            relative_weight = accuracy / avg_accuracy if avg_accuracy > 0 else 1
            
            # Clip to reasonable range
            self.learned_weights[module_name] = max(0.3, min(2.0, relative_weight))
        
        self._save_state()
        
        logger.info(f"🔄 ML weights updated - {len(self.learned_weights)} modules")
    
    def get_weight_for_module(self, module_name: str) -> float:
        """Modül için öğrenilmiş ağırlık al."""
        return self.learned_weights.get(module_name, 1.0)
    
    def get_performance_report(self) -> Dict:
        """Performans raporu."""
        if not self.module_performance:
            return {'error': 'No data yet'}
        
        # Sort by accuracy
        sorted_modules = sorted(
            self.module_performance.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        
        top_5 = sorted_modules[:5]
        bottom_5 = sorted_modules[-5:]
        
        return {
            'total_signals': len(self.signal_results),
            'modules_tracked': len(self.module_performance),
            'top_5': [(m, p['accuracy']) for m, p in top_5],
            'bottom_5': [(m, p['accuracy']) for m, p in bottom_5],
            'overall_signals_win_rate': self._calculate_overall_win_rate()
        }
    
    def _calculate_overall_win_rate(self) -> float:
        """Genel kazanma oranı."""
        results = [s for s in self.signal_results if s.get('result')]
        if not results:
            return 0
        
        wins = len([s for s in results if s['result'] == 'WIN'])
        return (wins / len(results)) * 100


# Global instance
_enhancer = None

def get_enhancer() -> MLTrainingEnhancer:
    """Get or create enhancer instance."""
    global _enhancer
    if _enhancer is None:
        _enhancer = MLTrainingEnhancer()
    return _enhancer
