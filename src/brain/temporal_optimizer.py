# -*- coding: utf-8 -*-
"""
DEMIR AI - Temporal Module Optimizer
Zaman ve volatilite bazlı modül performansını optimize eder.

PHASE 98: True AI - Temporal Optimization
- Günün hangi saatinde hangi modül daha doğru?
- Yüksek/düşük volatilitede hangi modül iyice?
- Hafta içi/sonu performans farkları
- Session bazlı (Asia/Europe/US) optimizasyon
"""
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger("TEMPORAL_OPTIMIZER")


class TemporalModuleOptimizer:
    """
    Zamansal Modül Optimize Edici
    
    Hangi modülün hangi zaman diliminde daha başarılı olduğunu öğrenir.
    """
    
    TEMPORAL_FILE = "temporal_data.json"
    
    # Trading sessions (UTC)
    SESSIONS = {
        'ASIA': (0, 8),      # 00:00 - 08:00 UTC
        'EUROPE': (8, 16),   # 08:00 - 16:00 UTC
        'US': (16, 24),      # 16:00 - 24:00 UTC
    }
    
    # Volatility levels
    VOLATILITY_LEVELS = ['LOW', 'MEDIUM', 'HIGH']
    
    def __init__(self):
        # {module: {session: {wins, losses, total, win_rate}}}
        self.session_stats: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0
        }))
        
        # {module: {volatility_level: {wins, losses, total, win_rate}}}
        self.volatility_stats: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0
        }))
        
        # {module: {hour: {wins, losses, total, win_rate}}}
        self.hourly_stats: Dict[str, Dict[int, Dict]] = defaultdict(lambda: defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0
        }))
        
        self._load_data()
        logger.info("✅ Temporal Module Optimizer initialized")
    
    def _load_data(self):
        """Mevcut verileri yükle."""
        try:
            if os.path.exists(self.TEMPORAL_FILE):
                with open(self.TEMPORAL_FILE, 'r') as f:
                    data = json.load(f)
                    
                    # Session stats
                    for module, sessions in data.get('session_stats', {}).items():
                        for session, stats in sessions.items():
                            self.session_stats[module][session] = stats
                    
                    # Volatility stats
                    for module, levels in data.get('volatility_stats', {}).items():
                        for level, stats in levels.items():
                            self.volatility_stats[module][level] = stats
                    
                    # Hourly stats
                    for module, hours in data.get('hourly_stats', {}).items():
                        for hour, stats in hours.items():
                            self.hourly_stats[module][int(hour)] = stats
                            
        except Exception as e:
            logger.warning(f"Temporal data load failed: {e}")
    
    def _save_data(self):
        """Verileri kaydet."""
        try:
            with open(self.TEMPORAL_FILE, 'w') as f:
                json.dump({
                    'session_stats': dict(self.session_stats),
                    'volatility_stats': dict(self.volatility_stats),
                    'hourly_stats': {m: {str(h): v for h, v in hours.items()} 
                                    for m, hours in self.hourly_stats.items()},
                    'last_update': datetime.now().isoformat()
                }, f, indent=2, default=dict)
        except Exception as e:
            logger.warning(f"Temporal data save failed: {e}")
    
    def _get_current_session(self) -> str:
        """Mevcut trading session'ı belirle."""
        hour = datetime.utcnow().hour
        
        for session, (start, end) in self.SESSIONS.items():
            if start <= hour < end:
                return session
        
        return 'ASIA'  # Default
    
    def _get_volatility_level(self, volatility_pct: float) -> str:
        """Volatilite seviyesini belirle."""
        if volatility_pct < 2:
            return 'LOW'
        elif volatility_pct < 5:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def record_module_temporal(self, modules: List[str], is_success: bool, 
                               volatility_pct: float = 2.0, signal_time: Optional[datetime] = None):
        """
        Modül performansını zamansal olarak kaydet.
        
        Args:
            modules: Sinyalde yer alan modüller
            is_success: Başarılı mı?
            volatility_pct: Sinyal anındaki volatilite %
            signal_time: Sinyal zamanı (None = şimdi)
        """
        if signal_time is None:
            signal_time = datetime.utcnow()
        
        session = self._get_current_session()
        volatility_level = self._get_volatility_level(volatility_pct)
        hour = signal_time.hour
        
        for module in modules:
            # Session stats
            stats = self.session_stats[module][session]
            stats['total'] += 1
            if is_success:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            if stats['total'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['total']) * 100
            
            # Volatility stats
            vstats = self.volatility_stats[module][volatility_level]
            vstats['total'] += 1
            if is_success:
                vstats['wins'] += 1
            else:
                vstats['losses'] += 1
            if vstats['total'] > 0:
                vstats['win_rate'] = (vstats['wins'] / vstats['total']) * 100
            
            # Hourly stats
            hstats = self.hourly_stats[module][hour]
            hstats['total'] += 1
            if is_success:
                hstats['wins'] += 1
            else:
                hstats['losses'] += 1
            if hstats['total'] > 0:
                hstats['win_rate'] = (hstats['wins'] / hstats['total']) * 100
        
        self._save_data()
    
    def get_temporal_multiplier(self, module: str, volatility_pct: float = 2.0) -> float:
        """
        Mevcut zaman ve volatilite için modül multiplier'ı hesapla.
        
        Returns:
            Multiplier (0.7 - 1.4)
        """
        session = self._get_current_session()
        volatility_level = self._get_volatility_level(volatility_pct)
        hour = datetime.utcnow().hour
        
        multiplier = 1.0
        count = 0
        
        # Session-based adjustment
        if module in self.session_stats and session in self.session_stats[module]:
            stats = self.session_stats[module][session]
            if stats['total'] >= 3:  # Min samples
                wr = stats['win_rate']
                # 50% = neutral, >50% = boost, <50% = penalty
                session_adj = (wr - 50) / 100  # -0.5 to +0.5
                multiplier += session_adj * 0.3
                count += 1
        
        # Volatility-based adjustment
        if module in self.volatility_stats and volatility_level in self.volatility_stats[module]:
            stats = self.volatility_stats[module][volatility_level]
            if stats['total'] >= 3:
                wr = stats['win_rate']
                vol_adj = (wr - 50) / 100
                multiplier += vol_adj * 0.3
                count += 1
        
        # Hour-based adjustment
        if module in self.hourly_stats and hour in self.hourly_stats[module]:
            stats = self.hourly_stats[module][hour]
            if stats['total'] >= 2:
                wr = stats['win_rate']
                hour_adj = (wr - 50) / 100
                multiplier += hour_adj * 0.2
                count += 1
        
        # Clamp
        return max(0.7, min(1.4, multiplier))
    
    def apply_temporal_weights(self, base_weights: Dict[str, float], 
                                volatility_pct: float = 2.0) -> Dict[str, float]:
        """
        Zamansal optimizasyon ile ağırlıkları ayarla.
        
        Args:
            base_weights: Temel ağırlıklar
            volatility_pct: Mevcut volatilite
            
        Returns:
            Zamansal olarak optimize edilmiş ağırlıklar
        """
        new_weights = {}
        
        for module, weight in base_weights.items():
            multiplier = self.get_temporal_multiplier(module, volatility_pct)
            new_weights[module] = weight * multiplier
        
        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        return new_weights
    
    def get_best_modules_now(self, n: int = 5) -> List[Tuple[str, float]]:
        """Şu an için en iyi modülleri getir."""
        session = self._get_current_session()
        results = []
        
        for module, sessions in self.session_stats.items():
            if session in sessions:
                stats = sessions[session]
                if stats['total'] >= 3:
                    results.append((module, stats['win_rate']))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]
    
    def get_temporal_report(self) -> str:
        """Zamansal performans raporu."""
        session = self._get_current_session()
        hour = datetime.utcnow().hour
        
        report = "⏰ **ZAMANSAL PERFORMANS RAPORU**\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        report += f"📍 Session: **{session}** | Saat: {hour}:00 UTC\n\n"
        
        best = self.get_best_modules_now(5)
        if best:
            report += "**🏆 Bu Session'da En İyi:**\n"
            for module, wr in best:
                emoji = "🟢" if wr >= 60 else "🟡" if wr >= 50 else "🔴"
                report += f"  {emoji} {module}: %{wr:.1f}\n"
        
        return report


# Global instance
_optimizer = None

def get_optimizer() -> TemporalModuleOptimizer:
    """Get or create optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = TemporalModuleOptimizer()
    return _optimizer
