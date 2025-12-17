# -*- coding: utf-8 -*-
"""
DEMIR AI - Smart Timing Filter
Sinyal spam'ını önler, sadece kaliteli sinyalleri geçirir.

PHASE 49: Signal Quality Control
- 4 saat içinde max 1 sinyal
- Güven threshold kontrolü
- Volatilite filtresi
- Piyasa saati kontrolü
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
import os

logger = logging.getLogger("SMART_TIMING")


@dataclass
class SignalRecord:
    """Geçmiş sinyal kaydı"""
    symbol: str
    direction: str
    confidence: float
    timestamp: datetime
    outcome: Optional[str] = None  # WIN / LOSS / PENDING


class SmartTimingFilter:
    """
    Akıllı Zamanlama Filtresi
    
    Kurallar:
    1. Aynı coin için 4 saat içinde max 1 sinyal
    2. Tüm coinler için günde max 8 sinyal
    3. Güven < 65% sinyaller filtrelenir
    4. Yüksek volatilite dönemlerinde dikkatli ol
    5. Hafta sonu düşük likidite uyarısı
    """
    
    # Zaman kısıtlamaları
    MIN_INTERVAL_SAME_COIN = timedelta(hours=4)
    MIN_INTERVAL_ANY_COIN = timedelta(minutes=30)
    MAX_SIGNALS_PER_DAY = 8
    
    # Kalite eşikleri
    MIN_CONFIDENCE = 65
    MIN_CONSENSUS = 60
    
    # Volatilite limitleri
    MAX_VOLATILITY_FOR_SIGNAL = 5.0  # %5'ten fazla saatlik volatilite = dikkat
    
    def __init__(self, history_file: str = 'signal_history.json'):
        self.history_file = history_file
        self.signal_history: List[SignalRecord] = []
        self._load_history()
    
    def _load_history(self):
        """Geçmiş sinyalleri yükle."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.signal_history = [
                        SignalRecord(
                            symbol=d['symbol'],
                            direction=d['direction'],
                            confidence=d['confidence'],
                            timestamp=datetime.fromisoformat(d['timestamp']),
                            outcome=d.get('outcome')
                        )
                        for d in data
                    ]
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
            self.signal_history = []
    
    def _save_history(self):
        """Geçmişi kaydet."""
        try:
            data = [
                {
                    'symbol': s.symbol,
                    'direction': s.direction,
                    'confidence': s.confidence,
                    'timestamp': s.timestamp.isoformat(),
                    'outcome': s.outcome
                }
                for s in self.signal_history[-100:]  # Son 100 sinyal
            ]
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")
    
    def can_send_signal(self, symbol: str, direction: str, confidence: float,
                       consensus: float = 0, volatility: float = 0) -> Dict:
        """
        Sinyal gönderilebilir mi kontrol et.
        
        Returns:
            {
                'allowed': True/False,
                'reason': 'OK' / 'Too soon' / 'Low confidence' / ...,
                'next_allowed': datetime or None,
                'warnings': []
            }
        """
        now = datetime.now()
        warnings = []
        
        # 1. Güven kontrolü
        if confidence < self.MIN_CONFIDENCE:
            return {
                'allowed': False,
                'reason': f'Güven çok düşük: {confidence:.0f}% < {self.MIN_CONFIDENCE}%',
                'next_allowed': None,
                'warnings': []
            }
        
        # 2. Consensus kontrolü
        if consensus > 0 and consensus < self.MIN_CONSENSUS:
            return {
                'allowed': False,
                'reason': f'Konsensüs yetersiz: {consensus:.0f}% < {self.MIN_CONSENSUS}%',
                'next_allowed': None,
                'warnings': []
            }
        
        # 3. Aynı coin için son sinyal kontrolü
        same_coin_signals = [
            s for s in self.signal_history
            if s.symbol == symbol and (now - s.timestamp) < self.MIN_INTERVAL_SAME_COIN
        ]
        
        if same_coin_signals:
            last = max(same_coin_signals, key=lambda x: x.timestamp)
            next_allowed = last.timestamp + self.MIN_INTERVAL_SAME_COIN
            wait_mins = (next_allowed - now).total_seconds() / 60
            return {
                'allowed': False,
                'reason': f'{symbol} için çok erken. {wait_mins:.0f} dk bekle.',
                'next_allowed': next_allowed,
                'warnings': []
            }
        
        # 4. Herhangi bir coin için son sinyal kontrolü
        recent_signals = [
            s for s in self.signal_history
            if (now - s.timestamp) < self.MIN_INTERVAL_ANY_COIN
        ]
        
        if recent_signals:
            last = max(recent_signals, key=lambda x: x.timestamp)
            next_allowed = last.timestamp + self.MIN_INTERVAL_ANY_COIN
            wait_mins = (next_allowed - now).total_seconds() / 60
            return {
                'allowed': False,
                'reason': f'Son sinyalden sonra {wait_mins:.0f} dk bekle.',
                'next_allowed': next_allowed,
                'warnings': []
            }
        
        # 5. Günlük limit kontrolü
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_signals = [
            s for s in self.signal_history
            if s.timestamp >= today_start
        ]
        
        if len(today_signals) >= self.MAX_SIGNALS_PER_DAY:
            return {
                'allowed': False,
                'reason': f'Günlük limit doldu: {len(today_signals)}/{self.MAX_SIGNALS_PER_DAY}',
                'next_allowed': today_start + timedelta(days=1),
                'warnings': []
            }
        
        # 6. Volatilite kontrolü
        if volatility > self.MAX_VOLATILITY_FOR_SIGNAL:
            warnings.append(f'⚠️ Yüksek volatilite: {volatility:.1f}%')
        
        # 7. Hafta sonu kontrolü
        if now.weekday() >= 5:  # Cumartesi veya Pazar
            warnings.append('⚠️ Hafta sonu - düşük likidite riski')
        
        # 8. Gece saatleri (00:00 - 06:00 UTC)
        if 0 <= now.hour < 6:
            warnings.append('⚠️ Düşük hacim saati')
        
        # Tüm kontroller geçti
        return {
            'allowed': True,
            'reason': 'OK',
            'next_allowed': None,
            'warnings': warnings
        }
    
    def record_signal(self, symbol: str, direction: str, confidence: float):
        """Gönderilen sinyali kaydet."""
        record = SignalRecord(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now()
        )
        self.signal_history.append(record)
        self._save_history()
        logger.info(f"📝 Signal recorded: {symbol} {direction} @ {confidence:.0f}%")
    
    def update_outcome(self, symbol: str, timestamp: datetime, outcome: str):
        """Sinyal sonucunu güncelle (WIN/LOSS)."""
        for signal in self.signal_history:
            if signal.symbol == symbol and abs((signal.timestamp - timestamp).total_seconds()) < 60:
                signal.outcome = outcome
                logger.info(f"📊 Outcome updated: {symbol} = {outcome}")
                break
        self._save_history()
    
    def get_statistics(self) -> Dict:
        """Sinyal istatistikleri."""
        if not self.signal_history:
            return {'total': 0, 'win_rate': 0, 'signals_today': 0}
        
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        completed = [s for s in self.signal_history if s.outcome in ['WIN', 'LOSS']]
        wins = [s for s in completed if s.outcome == 'WIN']
        today = [s for s in self.signal_history if s.timestamp >= today_start]
        
        return {
            'total': len(self.signal_history),
            'completed': len(completed),
            'wins': len(wins),
            'losses': len(completed) - len(wins),
            'win_rate': (len(wins) / len(completed) * 100) if completed else 0,
            'signals_today': len(today),
            'remaining_today': self.MAX_SIGNALS_PER_DAY - len(today)
        }
    
    def get_next_allowed_time(self, symbol: str) -> Optional[datetime]:
        """Belirli coin için sonraki izin verilen zaman."""
        now = datetime.now()
        
        same_coin = [
            s for s in self.signal_history
            if s.symbol == symbol and (now - s.timestamp) < self.MIN_INTERVAL_SAME_COIN
        ]
        
        if same_coin:
            last = max(same_coin, key=lambda x: x.timestamp)
            return last.timestamp + self.MIN_INTERVAL_SAME_COIN
        
        return now  # Hemen gönderilebilir
    
    def get_cooldown_status(self) -> Dict:
        """Tüm coinler için bekleme durumu."""
        now = datetime.now()
        coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT']
        
        status = {}
        for coin in coins:
            next_time = self.get_next_allowed_time(coin)
            if next_time and next_time > now:
                wait_mins = (next_time - now).total_seconds() / 60
                status[coin] = {
                    'can_signal': False,
                    'wait_minutes': wait_mins,
                    'next_allowed': next_time.isoformat()
                }
            else:
                status[coin] = {
                    'can_signal': True,
                    'wait_minutes': 0,
                    'next_allowed': None
                }
        
        return status


# Convenience functions
def can_send_now(symbol: str, confidence: float) -> bool:
    """Hızlı kontrol: Sinyal gönderilebilir mi?"""
    filter = SmartTimingFilter()
    result = filter.can_send_signal(symbol, '', confidence)
    return result['allowed']


def record_and_check(symbol: str, direction: str, confidence: float) -> Dict:
    """Kontrol et ve kaydet."""
    filter = SmartTimingFilter()
    result = filter.can_send_signal(symbol, direction, confidence)
    
    if result['allowed']:
        filter.record_signal(symbol, direction, confidence)
    
    return result


def get_timing_stats() -> Dict:
    """Zamanlama istatistikleri."""
    filter = SmartTimingFilter()
    return {
        'stats': filter.get_statistics(),
        'cooldown': filter.get_cooldown_status()
    }
