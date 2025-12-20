# -*- coding: utf-8 -*-
"""
DEMIR AI - Performance Tracker
===============================
Sinyal performansını takip eden ve doğruluk oranı hesaplayan sistem.

Features:
1. Her sinyali kaydet
2. TP/SL durumunu takip et
3. Doğruluk oranı hesapla
4. En iyi/kötü modülleri belirle
"""
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import requests

logger = logging.getLogger("PERFORMANCE_TRACKER")

# Paths
DATA_DIR = Path("src/brain/models/storage")
SIGNALS_FILE = DATA_DIR / "signal_history.json"
PERFORMANCE_FILE = DATA_DIR / "performance_stats.json"


@dataclass
class SignalRecord:
    """Sinyal kaydı."""
    id: str
    symbol: str
    direction: str  # LONG/SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    sources: List[str]
    created_at: str
    status: str = "ACTIVE"  # ACTIVE, TP_HIT, SL_HIT, EXPIRED
    exit_price: float = 0.0
    exit_time: str = ""
    pnl_percent: float = 0.0
    duration_hours: float = 0.0


class PerformanceTracker:
    """
    Sinyal Performans Takip Sistemi
    
    - Her sinyali kaydeder
    - TP/SL vurulduğunda günceller
    - Doğruluk oranı hesaplar
    - Modül bazlı performans analizi
    """
    
    SIGNAL_EXPIRY_HOURS = 48  # 48 saat sonra sinyal expired olur
    
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.signals: List[SignalRecord] = []
        self.stats: Dict = {}
        self._load_data()
    
    def _load_data(self):
        """Kayıtlı verileri yükle."""
        try:
            if SIGNALS_FILE.exists():
                with open(SIGNALS_FILE, 'r') as f:
                    data = json.load(f)
                    self.signals = [SignalRecord(**s) for s in data]
            
            if PERFORMANCE_FILE.exists():
                with open(PERFORMANCE_FILE, 'r') as f:
                    self.stats = json.load(f)
        except Exception as e:
            logger.warning(f"Data load failed: {e}")
            self.signals = []
            self.stats = {}
    
    def _save_data(self):
        """Verileri kaydet."""
        try:
            with open(SIGNALS_FILE, 'w') as f:
                json.dump([asdict(s) for s in self.signals], f, indent=2)
            
            with open(PERFORMANCE_FILE, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Data save failed: {e}")
    
    def record_signal(self, signal) -> str:
        """
        Yeni sinyali kaydet.
        
        Args:
            signal: unified_brain.Signal object
        
        Returns:
            signal_id
        """
        signal_id = f"{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        record = SignalRecord(
            id=signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            sources=signal.sources if hasattr(signal, 'sources') else [],
            created_at=datetime.now().isoformat()
        )
        
        self.signals.append(record)
        self._save_data()
        
        logger.info(f"📝 Signal recorded: {signal_id}")
        return signal_id
    
    async def check_active_signals(self) -> List[Dict]:
        """
        Aktif sinyalleri kontrol et, TP/SL vuruldu mu?
        
        Returns:
            List of updated signals
        """
        results = []
        
        for signal in self.signals:
            if signal.status != "ACTIVE":
                continue
            
            try:
                # Fiyat kontrolü
                current_price = await self._get_current_price(signal.symbol)
                
                if current_price == 0:
                    continue
                
                # TP/SL kontrolü
                if signal.direction == "LONG":
                    if current_price >= signal.take_profit:
                        self._close_signal(signal, "TP_HIT", current_price)
                        results.append(self._format_result(signal))
                    elif current_price <= signal.stop_loss:
                        self._close_signal(signal, "SL_HIT", current_price)
                        results.append(self._format_result(signal))
                
                elif signal.direction == "SHORT":
                    if current_price <= signal.take_profit:
                        self._close_signal(signal, "TP_HIT", current_price)
                        results.append(self._format_result(signal))
                    elif current_price >= signal.stop_loss:
                        self._close_signal(signal, "SL_HIT", current_price)
                        results.append(self._format_result(signal))
                
                # Expiry kontrolü
                created = datetime.fromisoformat(signal.created_at)
                age_hours = (datetime.now() - created).total_seconds() / 3600
                
                if age_hours >= self.SIGNAL_EXPIRY_HOURS:
                    self._close_signal(signal, "EXPIRED", current_price)
                    results.append(self._format_result(signal))
                    
            except Exception as e:
                logger.debug(f"Signal check error for {signal.id}: {e}")
        
        if results:
            self._update_stats()
            self._save_data()
        
        return results
    
    def _close_signal(self, signal: SignalRecord, status: str, exit_price: float):
        """Sinyali kapat."""
        signal.status = status
        signal.exit_price = exit_price
        signal.exit_time = datetime.now().isoformat()
        
        # PnL hesapla
        if signal.direction == "LONG":
            signal.pnl_percent = ((exit_price - signal.entry_price) / signal.entry_price) * 100
        else:
            signal.pnl_percent = ((signal.entry_price - exit_price) / signal.entry_price) * 100
        
        # Duration hesapla
        created = datetime.fromisoformat(signal.created_at)
        signal.duration_hours = (datetime.now() - created).total_seconds() / 3600
        
        logger.info(f"📊 Signal closed: {signal.id} → {status} ({signal.pnl_percent:+.2f}%)")
    
    def _format_result(self, signal: SignalRecord) -> Dict:
        """Sinyal sonucunu formatla."""
        emoji = "✅" if signal.status == "TP_HIT" else "❌" if signal.status == "SL_HIT" else "⏰"
        
        return {
            'signal_id': signal.id,
            'symbol': signal.symbol,
            'direction': signal.direction,
            'status': signal.status,
            'entry_price': signal.entry_price,
            'exit_price': signal.exit_price,
            'pnl_percent': signal.pnl_percent,
            'duration_hours': signal.duration_hours,
            'emoji': emoji
        }
    
    async def _get_current_price(self, symbol: str) -> float:
        """Güncel fiyatı al."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            if resp.status_code == 200:
                return float(resp.json().get('price', 0))
        except:
            pass
        return 0
    
    def _update_stats(self):
        """İstatistikleri güncelle."""
        closed = [s for s in self.signals if s.status != "ACTIVE"]
        
        if not closed:
            return
        
        tp_count = len([s for s in closed if s.status == "TP_HIT"])
        sl_count = len([s for s in closed if s.status == "SL_HIT"])
        expired_count = len([s for s in closed if s.status == "EXPIRED"])
        
        total_pnl = sum(s.pnl_percent for s in closed)
        avg_pnl = total_pnl / len(closed) if closed else 0
        
        win_rate = (tp_count / len(closed)) * 100 if closed else 0
        
        # Modül bazlı performans
        module_stats = {}
        for signal in closed:
            for source in signal.sources:
                if source not in module_stats:
                    module_stats[source] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
                
                if signal.status == "TP_HIT":
                    module_stats[source]['wins'] += 1
                else:
                    module_stats[source]['losses'] += 1
                
                module_stats[source]['total_pnl'] += signal.pnl_percent
        
        # En iyi modüller
        best_modules = sorted(
            module_stats.items(),
            key=lambda x: x[1]['wins'] / (x[1]['wins'] + x[1]['losses'] + 0.1),
            reverse=True
        )[:5]
        
        self.stats = {
            'total_signals': len(self.signals),
            'active_signals': len([s for s in self.signals if s.status == "ACTIVE"]),
            'closed_signals': len(closed),
            'tp_count': tp_count,
            'sl_count': sl_count,
            'expired_count': expired_count,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'best_modules': [m[0] for m in best_modules],
            'updated_at': datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict:
        """İstatistikleri döndür."""
        self._update_stats()
        return self.stats
    
    def format_stats_message(self) -> str:
        """Telegram için performans mesajı."""
        stats = self.get_stats()
        
        return f"""
📊 DEMIR AI PERFORMANS
━━━━━━━━━━━━━━━━━━
📈 Toplam Sinyal: {stats.get('total_signals', 0)}
🎯 Kazanan: {stats.get('tp_count', 0)}
❌ Kaybeden: {stats.get('sl_count', 0)}
⏰ Süresi Dolan: {stats.get('expired_count', 0)}
━━━━━━━━━━━━━━━━━━
✅ Win Rate: %{stats.get('win_rate', 0):.1f}
💰 Toplam PnL: {stats.get('total_pnl', 0):+.2f}%
📊 Ortalama: {stats.get('avg_pnl', 0):+.2f}%
━━━━━━━━━━━━━━━━━━
🏆 En İyi Modüller:
{chr(10).join(['• ' + m for m in stats.get('best_modules', [])[:3]])}
━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_tracker: Optional[PerformanceTracker] = None

def get_performance_tracker() -> PerformanceTracker:
    """Get or create tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tracker = get_performance_tracker()
    print(tracker.format_stats_message())
