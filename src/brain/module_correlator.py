# -*- coding: utf-8 -*-
"""
DEMIR AI - Module Correlation Analyzer
Modüller arası ilişkiyi analiz eder ve sinerjik kombinasyonları tespit eder.

PHASE 97: True AI - Context Understanding
- Hangi modüller birlikte iyi çalışıyor?
- Hangi kombinasyonlar başarılı sinyaller üretiyor?
- Çelişen modülleri tespit et
- Konsensüs kalitesini değerlendir
"""
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import itertools

logger = logging.getLogger("MODULE_CORRELATOR")


class ModuleCorrelationAnalyzer:
    """
    Modül Korelasyon Analizörü
    
    Hangi modül kombinasyonlarının başarılı olduğunu öğrenir.
    Güçlü korelasyonlar = bonus, çelişkiler = penalty.
    """
    
    CORRELATION_FILE = "module_correlations.json"
    
    # Minimum pair samples for statistical significance
    MIN_PAIR_SAMPLES = 3
    
    def __init__(self):
        self.pair_stats: Dict[str, Dict] = {}  # "ModA+ModB" -> {wins, losses, total}
        self.conflict_pairs: Set[str] = set()   # Known conflicting pairs
        self.synergy_pairs: Set[str] = set()    # Known synergistic pairs
        self._load_data()
        logger.info("✅ Module Correlation Analyzer initialized")
    
    def _load_data(self):
        """Mevcut verileri yükle."""
        try:
            if os.path.exists(self.CORRELATION_FILE):
                with open(self.CORRELATION_FILE, 'r') as f:
                    data = json.load(f)
                    self.pair_stats = data.get('pair_stats', {})
                    self.conflict_pairs = set(data.get('conflict_pairs', []))
                    self.synergy_pairs = set(data.get('synergy_pairs', []))
        except Exception as e:
            logger.warning(f"Correlation data load failed: {e}")
    
    def _save_data(self):
        """Verileri kaydet."""
        try:
            with open(self.CORRELATION_FILE, 'w') as f:
                json.dump({
                    'pair_stats': self.pair_stats,
                    'conflict_pairs': list(self.conflict_pairs),
                    'synergy_pairs': list(self.synergy_pairs),
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Correlation data save failed: {e}")
    
    def _make_pair_key(self, mod1: str, mod2: str) -> str:
        """İki modül için sıralı anahtar oluştur."""
        return '+'.join(sorted([mod1, mod2]))
    
    def record_module_combination(self, agreeing_modules: List[str], is_success: bool):
        """
        Başarılı/başarısız sinyaldeki modül kombinasyonlarını kaydet.
        
        Args:
            agreeing_modules: Aynı yönde sinyal veren modüller
            is_success: Sinyal başarılı mı?
        """
        if len(agreeing_modules) < 2:
            return
        
        # Tüm çiftleri kaydet
        for mod1, mod2 in itertools.combinations(agreeing_modules, 2):
            pair_key = self._make_pair_key(mod1, mod2)
            
            if pair_key not in self.pair_stats:
                self.pair_stats[pair_key] = {
                    'wins': 0,
                    'losses': 0,
                    'total': 0,
                    'win_rate': 0
                }
            
            stats = self.pair_stats[pair_key]
            stats['total'] += 1
            
            if is_success:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            
            if stats['total'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['total']) * 100
        
        # Synergy/conflict tespiti
        self._update_relationships()
        self._save_data()
    
    def _update_relationships(self):
        """Synergy ve conflict ilişkilerini güncelle."""
        self.synergy_pairs.clear()
        self.conflict_pairs.clear()
        
        for pair_key, stats in self.pair_stats.items():
            if stats['total'] < self.MIN_PAIR_SAMPLES:
                continue
            
            win_rate = stats['win_rate']
            
            # Synergy: %65+ win rate together
            if win_rate >= 65:
                self.synergy_pairs.add(pair_key)
            
            # Conflict: %35- win rate together
            elif win_rate <= 35:
                self.conflict_pairs.add(pair_key)
    
    def calculate_consensus_quality(self, agreeing_modules: List[str]) -> Tuple[float, str]:
        """
        Mevcut konsensüsün kalitesini hesapla.
        
        Args:
            agreeing_modules: Aynı yönde oy veren modüller
            
        Returns:
            (quality_score 0-100, quality_label)
        """
        if len(agreeing_modules) < 2:
            return 50, "INSUFFICIENT"
        
        quality_score = 50  # Base
        synergy_count = 0
        conflict_count = 0
        
        for mod1, mod2 in itertools.combinations(agreeing_modules, 2):
            pair_key = self._make_pair_key(mod1, mod2)
            
            if pair_key in self.synergy_pairs:
                synergy_count += 1
                quality_score += 5  # Synergy bonus
            
            if pair_key in self.conflict_pairs:
                conflict_count += 1
                quality_score -= 10  # Conflict penalty
        
        # Historical performance of this combination
        total_pair_win_rate = 0
        pair_count = 0
        
        for mod1, mod2 in itertools.combinations(agreeing_modules, 2):
            pair_key = self._make_pair_key(mod1, mod2)
            if pair_key in self.pair_stats:
                stats = self.pair_stats[pair_key]
                if stats['total'] >= self.MIN_PAIR_SAMPLES:
                    total_pair_win_rate += stats['win_rate']
                    pair_count += 1
        
        if pair_count > 0:
            avg_pair_win_rate = total_pair_win_rate / pair_count
            # Add/subtract based on historical performance
            quality_score += (avg_pair_win_rate - 50) * 0.3
        
        # Module count bonus
        quality_score += min(len(agreeing_modules) * 2, 20)
        
        # Clamp
        quality_score = max(0, min(100, quality_score))
        
        # Label
        if quality_score >= 75:
            label = "EXCELLENT"
        elif quality_score >= 60:
            label = "GOOD"
        elif quality_score >= 45:
            label = "FAIR"
        else:
            label = "POOR"
        
        return quality_score, label
    
    def get_synergy_boost(self, modules: List[str]) -> float:
        """
        Sinerji bazlı güven boost'u hesapla.
        
        Returns:
            Multiplier (0.8 - 1.3)
        """
        if len(modules) < 2:
            return 1.0
        
        synergy_points = 0
        conflict_points = 0
        
        for mod1, mod2 in itertools.combinations(modules, 2):
            pair_key = self._make_pair_key(mod1, mod2)
            
            if pair_key in self.synergy_pairs:
                synergy_points += 1
            if pair_key in self.conflict_pairs:
                conflict_points += 1
        
        # Net effect
        net_effect = (synergy_points * 0.05) - (conflict_points * 0.1)
        
        # Clamp to 0.8 - 1.3 range
        return max(0.8, min(1.3, 1.0 + net_effect))
    
    def get_top_synergies(self, n: int = 5) -> List[Tuple[str, float]]:
        """En başarılı modül çiftlerini getir."""
        pairs = []
        
        for pair_key, stats in self.pair_stats.items():
            if stats['total'] >= self.MIN_PAIR_SAMPLES:
                pairs.append((pair_key, stats['win_rate']))
        
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:n]
    
    def get_top_conflicts(self, n: int = 3) -> List[Tuple[str, float]]:
        """En sorunlu modül çiftlerini getir."""
        pairs = []
        
        for pair_key, stats in self.pair_stats.items():
            if stats['total'] >= self.MIN_PAIR_SAMPLES:
                pairs.append((pair_key, stats['win_rate']))
        
        pairs.sort(key=lambda x: x[1])
        return pairs[:n]
    
    def get_correlation_report(self) -> str:
        """Korelasyon raporu oluştur."""
        report = "🔗 **MODÜL KORELASYON RAPORU**\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        top_synergies = self.get_top_synergies(5)
        if top_synergies:
            report += "**💚 En İyi Kombinasyonlar:**\n"
            for pair, wr in top_synergies:
                report += f"  🟢 {pair}: %{wr:.1f}\n"
        
        top_conflicts = self.get_top_conflicts(3)
        if top_conflicts:
            report += "\n**💔 Sorunlu Kombinasyonlar:**\n"
            for pair, wr in top_conflicts:
                report += f"  🔴 {pair}: %{wr:.1f}\n"
        
        report += f"\n📊 Toplam çift: {len(self.pair_stats)}"
        report += f"\n💚 Sinerji: {len(self.synergy_pairs)}"
        report += f"\n💔 Çelişki: {len(self.conflict_pairs)}"
        
        return report


# Global instance
_correlator = None

def get_correlator() -> ModuleCorrelationAnalyzer:
    """Get or create correlator instance."""
    global _correlator
    if _correlator is None:
        _correlator = ModuleCorrelationAnalyzer()
    return _correlator
