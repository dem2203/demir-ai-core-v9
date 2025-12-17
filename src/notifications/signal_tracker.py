# -*- coding: utf-8 -*-
"""
DEMIR AI - Signal Tracker
Aktif sinyalleri takip eder, TP/SL izler, sonuç bildirir.

PHASE 50: Advanced Signal Tracking
- Aktif sinyal takibi
- TP/SL monitoring (her 30 saniye)
- Sonuç bildirimi
- Duplicate önleme (TP/SL sonrası serbest)
"""
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
import requests

logger = logging.getLogger("SIGNAL_TRACKER")


@dataclass
class ActiveSignal:
    """Aktif sinyal kaydı"""
    signal_id: str
    symbol: str
    direction: str  # LONG / SHORT
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence: float
    created_at: datetime
    status: str = 'ACTIVE'  # ACTIVE / TP1_HIT / TP2_HIT / SL_HIT / EXPIRED
    result: Optional[str] = None  # WIN / LOSS
    profit_pct: float = 0
    closed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'result': self.result,
            'profit_pct': self.profit_pct,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'ActiveSignal':
        return ActiveSignal(
            signal_id=data['signal_id'],
            symbol=data['symbol'],
            direction=data['direction'],
            entry_price=data['entry_price'],
            stop_loss=data['stop_loss'],
            take_profit_1=data['take_profit_1'],
            take_profit_2=data['take_profit_2'],
            confidence=data['confidence'],
            created_at=datetime.fromisoformat(data['created_at']),
            status=data.get('status', 'ACTIVE'),
            result=data.get('result'),
            profit_pct=data.get('profit_pct', 0),
            closed_at=datetime.fromisoformat(data['closed_at']) if data.get('closed_at') else None
        )


class SignalTracker:
    """
    Aktif Sinyal Takip Sistemi
    
    Özellikler:
    1. Aktif sinyalleri kaydet ve izle
    2. Her 30 saniyede fiyat kontrolü
    3. TP1, TP2 veya SL vurulduğunda bildir
    4. Sinyal kapandıktan sonra yeni sinyal gönderilebilir
    5. 24 saat sonra timeout
    """
    
    STORAGE_FILE = 'active_signals.json'
    SIGNAL_TIMEOUT_HOURS = 24
    
    def __init__(self):
        self.active_signals: Dict[str, ActiveSignal] = {}
        self.closed_signals: List[ActiveSignal] = []
        self._load_signals()
    
    def _load_signals(self):
        """Sinyalleri dosyadan yükle."""
        try:
            if os.path.exists(self.STORAGE_FILE):
                with open(self.STORAGE_FILE, 'r') as f:
                    data = json.load(f)
                    for sig_data in data.get('active', []):
                        sig = ActiveSignal.from_dict(sig_data)
                        self.active_signals[sig.symbol] = sig
                    for sig_data in data.get('closed', []):
                        sig = ActiveSignal.from_dict(sig_data)
                        self.closed_signals.append(sig)
        except Exception as e:
            logger.warning(f"Could not load signals: {e}")
    
    def _save_signals(self):
        """Sinyalleri dosyaya kaydet."""
        try:
            data = {
                'active': [s.to_dict() for s in self.active_signals.values()],
                'closed': [s.to_dict() for s in self.closed_signals[-50:]]  # Son 50
            }
            with open(self.STORAGE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save signals: {e}")
    
    def generate_signal_id(self, symbol: str) -> str:
        """Benzersiz sinyal ID oluştur."""
        now = datetime.now()
        return f"{symbol.replace('USDT', '')}-{now.strftime('%Y%m%d-%H%M%S')}"
    
    def can_send_signal(self, symbol: str) -> Dict:
        """
        Bu coin için yeni sinyal gönderilebilir mi?
        
        Kural: Aktif sinyal yoksa veya önceki sinyal kapandıysa gönderebilir.
        """
        if symbol not in self.active_signals:
            return {
                'allowed': True,
                'reason': 'Aktif sinyal yok'
            }
        
        active = self.active_signals[symbol]
        
        # Aktif mi kontrol et
        if active.status == 'ACTIVE':
            return {
                'allowed': False,
                'reason': f'Aktif sinyal mevcut: {active.signal_id}',
                'active_signal': active.to_dict()
            }
        
        # Kapanmışsa izin ver
        return {
            'allowed': True,
            'reason': f'Önceki sinyal kapandı: {active.status}'
        }
    
    def register_signal(self, symbol: str, direction: str, entry: float,
                       stop_loss: float, tp1: float, tp2: float,
                       confidence: float) -> ActiveSignal:
        """Yeni sinyal kaydet."""
        signal_id = self.generate_signal_id(symbol)
        
        signal = ActiveSignal(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            confidence=confidence,
            created_at=datetime.now()
        )
        
        # Önceki varsa kapalılara taşı
        if symbol in self.active_signals:
            old = self.active_signals[symbol]
            if old.status == 'ACTIVE':
                old.status = 'REPLACED'
                old.closed_at = datetime.now()
            self.closed_signals.append(old)
        
        self.active_signals[symbol] = signal
        self._save_signals()
        
        logger.info(f"📝 Signal registered: {signal_id} {direction} @ ${entry:,.2f}")
        return signal
    
    def check_price_levels(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Fiyat seviyelerini kontrol et - TP veya SL vuruldu mu?
        
        Returns:
            None veya {'event': 'TP1_HIT' / 'TP2_HIT' / 'SL_HIT', 'signal': ...}
        """
        if symbol not in self.active_signals:
            return None
        
        signal = self.active_signals[symbol]
        
        if signal.status != 'ACTIVE':
            return None
        
        # Timeout kontrolü
        age = (datetime.now() - signal.created_at).total_seconds() / 3600
        if age > self.SIGNAL_TIMEOUT_HOURS:
            signal.status = 'EXPIRED'
            signal.closed_at = datetime.now()
            self._save_signals()
            return {
                'event': 'EXPIRED',
                'signal': signal.to_dict(),
                'message': f'{symbol} sinyali 24 saat sonra timeout oldu'
            }
        
        # LONG pozisyon
        if signal.direction == 'LONG':
            # SL kontrolü
            if current_price <= signal.stop_loss:
                signal.status = 'SL_HIT'
                signal.result = 'LOSS'
                signal.profit_pct = ((current_price / signal.entry_price) - 1) * 100
                signal.closed_at = datetime.now()
                self._save_signals()
                return {
                    'event': 'SL_HIT',
                    'signal': signal.to_dict(),
                    'profit_pct': signal.profit_pct,
                    'message': f'{symbol} LONG - Stop Loss vuruldu!'
                }
            
            # TP1 kontrolü
            if current_price >= signal.take_profit_1 and signal.status == 'ACTIVE':
                signal.status = 'TP1_HIT'
                signal.profit_pct = ((current_price / signal.entry_price) - 1) * 100
                self._save_signals()
                return {
                    'event': 'TP1_HIT',
                    'signal': signal.to_dict(),
                    'profit_pct': signal.profit_pct,
                    'message': f'{symbol} LONG - TP1 vuruldu! +{signal.profit_pct:.1f}%'
                }
            
            # TP2 kontrolü
            if current_price >= signal.take_profit_2:
                signal.status = 'TP2_HIT'
                signal.result = 'WIN'
                signal.profit_pct = ((current_price / signal.entry_price) - 1) * 100
                signal.closed_at = datetime.now()
                self._save_signals()
                return {
                    'event': 'TP2_HIT',
                    'signal': signal.to_dict(),
                    'profit_pct': signal.profit_pct,
                    'message': f'{symbol} LONG - TP2 vuruldu! +{signal.profit_pct:.1f}%'
                }
        
        # SHORT pozisyon
        elif signal.direction == 'SHORT':
            # SL kontrolü
            if current_price >= signal.stop_loss:
                signal.status = 'SL_HIT'
                signal.result = 'LOSS'
                signal.profit_pct = ((signal.entry_price / current_price) - 1) * 100
                signal.closed_at = datetime.now()
                self._save_signals()
                return {
                    'event': 'SL_HIT',
                    'signal': signal.to_dict(),
                    'profit_pct': signal.profit_pct,
                    'message': f'{symbol} SHORT - Stop Loss vuruldu!'
                }
            
            # TP1 kontrolü
            if current_price <= signal.take_profit_1 and signal.status == 'ACTIVE':
                signal.status = 'TP1_HIT'
                signal.profit_pct = ((signal.entry_price / current_price) - 1) * 100
                self._save_signals()
                return {
                    'event': 'TP1_HIT',
                    'signal': signal.to_dict(),
                    'profit_pct': signal.profit_pct,
                    'message': f'{symbol} SHORT - TP1 vuruldu! +{signal.profit_pct:.1f}%'
                }
            
            # TP2 kontrolü
            if current_price <= signal.take_profit_2:
                signal.status = 'TP2_HIT'
                signal.result = 'WIN'
                signal.profit_pct = ((signal.entry_price / current_price) - 1) * 100
                signal.closed_at = datetime.now()
                self._save_signals()
                return {
                    'event': 'TP2_HIT',
                    'signal': signal.to_dict(),
                    'profit_pct': signal.profit_pct,
                    'message': f'{symbol} SHORT - TP2 vuruldu! +{signal.profit_pct:.1f}%'
                }
        
        return None  # Hiçbir seviye vurulmadı
    
    def check_all_signals(self) -> List[Dict]:
        """Tüm aktif sinyalleri kontrol et."""
        events = []
        
        for symbol in list(self.active_signals.keys()):
            try:
                # Güncel fiyatı al
                resp = requests.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
                    timeout=5
                )
                if resp.status_code == 200:
                    current_price = float(resp.json()['price'])
                    event = self.check_price_levels(symbol, current_price)
                    if event:
                        events.append(event)
            except Exception as e:
                logger.warning(f"Price check failed for {symbol}: {e}")
        
        return events
    
    def get_active_signals(self) -> List[Dict]:
        """Aktif sinyalleri döndür."""
        return [
            s.to_dict() for s in self.active_signals.values()
            if s.status == 'ACTIVE'
        ]
    
    def get_statistics(self) -> Dict:
        """Performans istatistikleri."""
        wins = [s for s in self.closed_signals if s.result == 'WIN']
        losses = [s for s in self.closed_signals if s.result == 'LOSS']
        
        total_profit = sum(s.profit_pct for s in wins)
        total_loss = sum(abs(s.profit_pct) for s in losses)
        
        return {
            'total_signals': len(self.closed_signals),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(self.closed_signals) * 100) if self.closed_signals else 0,
            'total_profit_pct': total_profit,
            'total_loss_pct': total_loss,
            'net_profit_pct': total_profit - total_loss,
            'active_count': len([s for s in self.active_signals.values() if s.status == 'ACTIVE'])
        }
    
    def close_signal_manually(self, symbol: str, result: str = 'MANUAL'):
        """Sinyali manuel kapat."""
        if symbol in self.active_signals:
            signal = self.active_signals[symbol]
            signal.status = 'MANUAL_CLOSE'
            signal.result = result
            signal.closed_at = datetime.now()
            self._save_signals()
            logger.info(f"Signal manually closed: {signal.signal_id}")


# Convenience functions
def get_tracker() -> SignalTracker:
    """Singleton tracker instance."""
    return SignalTracker()


def can_send_for_coin(symbol: str) -> bool:
    """Hızlı kontrol: Sinyal gönderilebilir mi?"""
    tracker = SignalTracker()
    result = tracker.can_send_signal(symbol)
    return result['allowed']
