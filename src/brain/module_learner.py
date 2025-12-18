# -*- coding: utf-8 -*-
"""
DEMIR AI - Module Performance Learner
Geçmiş sinyal performansından öğrenerek modül ağırlıklarını otomatik ayarlar.

PHASE 95: True AI - Learning from Mistakes
- Her modülün başarı oranını takip et
- Başarılı modüllerin ağırlığını artır
- Başarısız modüllerin ağırlığını azalt
- Günlük öğrenme döngüsü
"""
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger("MODULE_LEARNER")


class ModulePerformanceLearner:
    """
    Modül Performans Öğrenicisi
    
    Signal Performance Tracker'dan veri alarak hangi modülün
    ne kadar başarılı olduğunu öğrenir ve ağırlıkları dinamik ayarlar.
    """
    
    LEARNER_FILE = "module_learner_data.json"
    WEIGHTS_FILE = "learned_weights.json"
    
    # Learning parameters
    LEARNING_RATE = 0.1  # Her döngüde max %10 değişim
    MIN_WEIGHT = 0.01    # Minimum %1 ağırlık
    MAX_WEIGHT = 0.15    # Maximum %15 ağırlık
    MIN_SIGNALS = 5      # Öğrenme için minimum sinyal sayısı
    
    def __init__(self):
        self.module_stats: Dict[str, Dict] = {}
        self.learned_weights: Dict[str, float] = {}
        self.signal_module_map: Dict[str, List[str]] = {}  # signal_id -> [modules that agreed]
        self._load_data()
        logger.info("✅ Module Performance Learner initialized")
    
    def _load_data(self):
        """Mevcut verileri yükle."""
        try:
            if os.path.exists(self.LEARNER_FILE):
                with open(self.LEARNER_FILE, 'r') as f:
                    data = json.load(f)
                    self.module_stats = data.get('module_stats', {})
                    self.signal_module_map = data.get('signal_module_map', {})
            
            if os.path.exists(self.WEIGHTS_FILE):
                with open(self.WEIGHTS_FILE, 'r') as f:
                    self.learned_weights = json.load(f)
                    
        except Exception as e:
            logger.warning(f"Learner data load failed: {e}")
    
    def _save_data(self):
        """Verileri kaydet."""
        try:
            with open(self.LEARNER_FILE, 'w') as f:
                json.dump({
                    'module_stats': self.module_stats,
                    'signal_module_map': self.signal_module_map,
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
            
            with open(self.WEIGHTS_FILE, 'w') as f:
                json.dump(self.learned_weights, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Learner data save failed: {e}")
    
    def record_signal_modules(self, signal_id: str, direction: str, module_votes: Dict[str, Dict]):
        """
        Sinyal gönderildiğinde, hangi modüllerin bu sinyale katıldığını kaydet.
        
        Args:
            signal_id: Unique signal ID
            direction: LONG or SHORT
            module_votes: {module_name: {direction, confidence}}
        """
        agreeing_modules = []
        
        for module_name, vote in module_votes.items():
            module_direction = vote.get('direction', '')
            
            # LONG sinyali için: BUY veya LONG veren modüller
            if direction == 'LONG' and module_direction in ['LONG', 'BUY']:
                agreeing_modules.append(module_name)
            # SHORT sinyali için: SHORT veya SELL veren modüller
            elif direction == 'SHORT' and module_direction in ['SHORT', 'SELL']:
                agreeing_modules.append(module_name)
        
        self.signal_module_map[signal_id] = agreeing_modules
        self._save_data()
        
        logger.info(f"📊 Recorded {len(agreeing_modules)} agreeing modules for signal {signal_id}")
    
    def record_signal_result(self, signal_id: str, is_success: bool, profit_pct: float):
        """
        Sinyal sonuçlandığında, katılan modüllerin skorlarını güncelle.
        
        Args:
            signal_id: Signal ID
            is_success: TP vuruldu mu?
            profit_pct: Gerçekleşen kar/zarar %
        """
        if signal_id not in self.signal_module_map:
            return
        
        agreeing_modules = self.signal_module_map[signal_id]
        
        for module_name in agreeing_modules:
            if module_name not in self.module_stats:
                self.module_stats[module_name] = {
                    'total_signals': 0,
                    'successful': 0,
                    'failed': 0,
                    'total_profit': 0,
                    'win_rate': 0,
                    'avg_profit': 0
                }
            
            stats = self.module_stats[module_name]
            stats['total_signals'] += 1
            stats['total_profit'] += profit_pct
            
            if is_success:
                stats['successful'] += 1
            else:
                stats['failed'] += 1
            
            # Metrics güncelle
            if stats['total_signals'] > 0:
                stats['win_rate'] = (stats['successful'] / stats['total_signals']) * 100
                stats['avg_profit'] = stats['total_profit'] / stats['total_signals']
        
        # Temizlik
        del self.signal_module_map[signal_id]
        self._save_data()
        
        logger.info(f"📈 Updated {len(agreeing_modules)} module stats for signal {signal_id}")
    
    def calculate_learned_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Öğrenilen performansa göre yeni ağırlıkları hesapla.
        
        Args:
            base_weights: Orijinal SignalOrchestrator ağırlıkları
            
        Returns:
            Öğrenilmiş ağırlıklar
        """
        new_weights = base_weights.copy()
        
        for module_name, stats in self.module_stats.items():
            if module_name not in base_weights:
                continue
            
            # Minimum sinyal sayısı kontrolü
            if stats['total_signals'] < self.MIN_SIGNALS:
                continue
            
            base = base_weights[module_name]
            win_rate = stats['win_rate']
            avg_profit = stats['avg_profit']
            
            # Performance score: Win rate + profit weighted
            # 50% win rate = neutral, >50% = positive, <50% = negative
            performance_score = (win_rate - 50) / 50  # -1 to +1 range
            
            # Profit bonus/penalty
            profit_adjustment = min(max(avg_profit / 5, -0.2), 0.2)  # Max ±20%
            
            # Combined adjustment
            total_adjustment = (performance_score * 0.7 + profit_adjustment * 0.3)
            
            # Apply with learning rate
            weight_change = base * total_adjustment * self.LEARNING_RATE
            new_weight = base + weight_change
            
            # Clamp to limits
            new_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))
            
            new_weights[module_name] = new_weight
            
            logger.debug(f"{module_name}: {base:.3f} → {new_weight:.3f} (WR: {win_rate:.1f}%)")
        
        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        self.learned_weights = new_weights
        self._save_data()
        
        return new_weights
    
    def get_module_report(self) -> str:
        """Modül performans raporu oluştur."""
        if not self.module_stats:
            return "📊 Henüz yeterli veri yok."
        
        # Sort by win rate
        sorted_modules = sorted(
            self.module_stats.items(),
            key=lambda x: x[1].get('win_rate', 0),
            reverse=True
        )
        
        report = "🧠 **MODÜL PERFORMANS RAPORU**\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        report += "**🏆 En Başarılı:**\n"
        for name, stats in sorted_modules[:5]:
            wr = stats.get('win_rate', 0)
            total = stats.get('total_signals', 0)
            emoji = "🟢" if wr >= 60 else "🟡" if wr >= 50 else "🔴"
            report += f"  {emoji} {name}: %{wr:.1f} ({total} sinyal)\n"
        
        report += "\n**📉 Gelişmeli:**\n"
        for name, stats in sorted_modules[-3:]:
            wr = stats.get('win_rate', 0)
            total = stats.get('total_signals', 0)
            if total >= self.MIN_SIGNALS:
                report += f"  🔴 {name}: %{wr:.1f} ({total} sinyal)\n"
        
        return report
    
    def get_learned_weights(self) -> Dict[str, float]:
        """Öğrenilmiş ağırlıkları getir."""
        return self.learned_weights.copy()


# Global instance
_learner = None

def get_learner() -> ModulePerformanceLearner:
    """Get or create learner instance."""
    global _learner
    if _learner is None:
        _learner = ModulePerformanceLearner()
    return _learner
