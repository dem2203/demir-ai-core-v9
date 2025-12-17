# -*- coding: utf-8 -*-
"""
DEMIR AI - Signal Validator
Sinyal sonuçlarını doğrular ve AI'ı geliştirir.

PHASE 51: Automatic Signal Validation
- Her sinyalin sonucunu izle
- Win/Loss oranını hesapla
- Model performansını analiz et
- Zayıf modülleri tespit et
- Ağırlıkları otomatik ayarla
"""
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("SIGNAL_VALIDATOR")


@dataclass
class SignalResult:
    """Sinyal sonucu"""
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    result: str  # WIN / LOSS
    profit_pct: float
    modules_involved: List[str]
    strongest_module: str
    weakest_module: str
    timestamp: datetime


class SignalValidator:
    """
    Sinyal Doğrulama ve Model Geliştirme Sistemi
    
    Özellikler:
    1. Sinyal sonuçlarını kaydet
    2. Her modülün başarı oranını hesapla
    3. En iyi/kötü performans gösteren modülleri bul
    4. Ağırlık önerisi yap
    5. Zayıf modülleri uyar
    """
    
    RESULTS_FILE = 'signal_validation_results.json'
    
    def __init__(self):
        self.results: List[SignalResult] = []
        self.module_stats: Dict[str, Dict] = {}
        self._load_results()
    
    def _load_results(self):
        """Sonuçları dosyadan yükle."""
        try:
            if os.path.exists(self.RESULTS_FILE):
                with open(self.RESULTS_FILE, 'r') as f:
                    data = json.load(f)
                    for r in data.get('results', []):
                        r['timestamp'] = datetime.fromisoformat(r['timestamp'])
                        self.results.append(SignalResult(**r))
                    self.module_stats = data.get('module_stats', {})
        except Exception as e:
            logger.warning(f"Could not load results: {e}")
    
    def _save_results(self):
        """Sonuçları kaydet."""
        try:
            data = {
                'results': [
                    {**asdict(r), 'timestamp': r.timestamp.isoformat()}
                    for r in self.results[-200:]  # Son 200 sinyal
                ],
                'module_stats': self.module_stats
            }
            with open(self.RESULTS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save results: {e}")
    
    def record_result(self, signal_id: str, symbol: str, direction: str,
                     entry_price: float, exit_price: float,
                     modules_involved: List[str]) -> SignalResult:
        """Sinyal sonucunu kaydet."""
        # Kar/zarar hesapla
        if direction == 'LONG':
            profit_pct = ((exit_price / entry_price) - 1) * 100
        else:
            profit_pct = ((entry_price / exit_price) - 1) * 100
        
        result = 'WIN' if profit_pct > 0 else 'LOSS'
        
        # En güçlü/zayıf modülü bul (bu sinyale katkıda bulunanlar)
        strongest = modules_involved[0] if modules_involved else 'Unknown'
        weakest = modules_involved[-1] if modules_involved else 'Unknown'
        
        signal_result = SignalResult(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            result=result,
            profit_pct=profit_pct,
            modules_involved=modules_involved,
            strongest_module=strongest,
            weakest_module=weakest,
            timestamp=datetime.now()
        )
        
        self.results.append(signal_result)
        
        # Modül istatistiklerini güncelle
        for module in modules_involved:
            if module not in self.module_stats:
                self.module_stats[module] = {'wins': 0, 'losses': 0, 'total_profit': 0}
            
            if result == 'WIN':
                self.module_stats[module]['wins'] += 1
            else:
                self.module_stats[module]['losses'] += 1
            
            self.module_stats[module]['total_profit'] += profit_pct
        
        self._save_results()
        logger.info(f"📊 Signal validated: {signal_id} = {result} ({profit_pct:+.2f}%)")
        
        return signal_result
    
    def get_module_performance(self) -> Dict:
        """Her modülün performansını hesapla."""
        performance = {}
        
        for module, stats in self.module_stats.items():
            total = stats['wins'] + stats['losses']
            if total == 0:
                continue
            
            win_rate = (stats['wins'] / total) * 100
            avg_profit = stats['total_profit'] / total
            
            performance[module] = {
                'wins': stats['wins'],
                'losses': stats['losses'],
                'total': total,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'rating': 'EXCELLENT' if win_rate >= 65 else 'GOOD' if win_rate >= 55 else 'AVERAGE' if win_rate >= 45 else 'POOR'
            }
        
        # En iyi ve en kötü modülleri bul
        if performance:
            best = max(performance, key=lambda x: performance[x]['win_rate'])
            worst = min(performance, key=lambda x: performance[x]['win_rate'])
            
            performance['_best_module'] = best
            performance['_worst_module'] = worst
        
        return performance
    
    def suggest_weight_adjustments(self) -> Dict:
        """Performansa dayalı ağırlık önerileri."""
        perf = self.get_module_performance()
        
        if not perf or '_best_module' not in perf:
            return {'message': 'Yeterli veri yok'}
        
        suggestions = {}
        
        for module, stats in perf.items():
            if module.startswith('_'):
                continue
            
            win_rate = stats['win_rate']
            current_weight = 0.10  # Default
            
            # Performansa göre ağırlık öner
            if win_rate >= 70:
                suggested = current_weight * 1.3  # %30 artış
                suggestion = 'INCREASE'
            elif win_rate >= 60:
                suggested = current_weight * 1.15  # %15 artış
                suggestion = 'SLIGHT_INCREASE'
            elif win_rate >= 50:
                suggested = current_weight  # Değişiklik yok
                suggestion = 'KEEP'
            elif win_rate >= 40:
                suggested = current_weight * 0.85  # %15 azalış
                suggestion = 'SLIGHT_DECREASE'
            else:
                suggested = current_weight * 0.7  # %30 azalış
                suggestion = 'DECREASE'
            
            suggestions[module] = {
                'current_win_rate': win_rate,
                'suggestion': suggestion,
                'current_weight': current_weight,
                'suggested_weight': min(0.20, max(0.03, suggested))
            }
        
        return suggestions
    
    def get_weekly_report(self) -> Dict:
        """Haftalık performans raporu."""
        week_ago = datetime.now() - timedelta(days=7)
        weekly = [r for r in self.results if r.timestamp >= week_ago]
        
        if not weekly:
            return {'message': 'Bu hafta sinyal yok'}
        
        wins = [r for r in weekly if r.result == 'WIN']
        total_profit = sum(r.profit_pct for r in weekly)
        
        return {
            'period': 'Son 7 Gün',
            'total_signals': len(weekly),
            'wins': len(wins),
            'losses': len(weekly) - len(wins),
            'win_rate': (len(wins) / len(weekly)) * 100,
            'total_profit_pct': total_profit,
            'avg_profit_pct': total_profit / len(weekly),
            'best_signal': max(weekly, key=lambda x: x.profit_pct).signal_id if weekly else None,
            'worst_signal': min(weekly, key=lambda x: x.profit_pct).signal_id if weekly else None
        }
    
    def identify_weak_spots(self) -> List[str]:
        """Zayıf noktaları tespit et."""
        weak_spots = []
        perf = self.get_module_performance()
        
        for module, stats in perf.items():
            if module.startswith('_'):
                continue
            
            if stats['win_rate'] < 45:
                weak_spots.append(f"⚠️ {module}: %{stats['win_rate']:.0f} win rate - iyileştirme gerekli")
            
            if stats['avg_profit'] < -1:
                weak_spots.append(f"⚠️ {module}: Ortalama {stats['avg_profit']:.1f}% kayıp")
        
        # Genel zayıflıklar
        report = self.get_weekly_report()
        if 'win_rate' in report and report['win_rate'] < 50:
            weak_spots.append(f"⚠️ Genel: Haftalık win rate %{report['win_rate']:.0f} - hedef %60+")
        
        return weak_spots
    
    def get_dashboard_summary(self) -> Dict:
        """Dashboard için özet."""
        perf = self.get_module_performance()
        report = self.get_weekly_report()
        weak = self.identify_weak_spots()
        
        return {
            'total_validated': len(self.results),
            'overall_win_rate': sum(1 for r in self.results if r.result == 'WIN') / len(self.results) * 100 if self.results else 0,
            'weekly_report': report,
            'module_performance': perf,
            'weak_spots': weak,
            'suggestions': self.suggest_weight_adjustments()
        }


# Convenience functions
def validate_signal(signal_id: str, symbol: str, direction: str,
                   entry: float, exit_price: float, modules: List[str]) -> Dict:
    """Sinyal doğrula."""
    validator = SignalValidator()
    result = validator.record_result(signal_id, symbol, direction, entry, exit_price, modules)
    return asdict(result)


def get_validation_summary() -> Dict:
    """Doğrulama özeti."""
    validator = SignalValidator()
    return validator.get_dashboard_summary()


def get_module_rankings() -> Dict:
    """Modül sıralaması."""
    validator = SignalValidator()
    return validator.get_module_performance()
