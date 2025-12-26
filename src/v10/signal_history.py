# -*- coding: utf-8 -*-
"""
Signal History Module
======================
Sinyal geçmişini yönetir - kaydetme, okuma, istatistik.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("SIGNAL_HISTORY")

HISTORY_FILE = Path("signal_history.json")
MAX_HISTORY = 50  # Son 50 sinyal


@dataclass
class SignalRecord:
    """Tek bir sinyal kaydı"""
    timestamp: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    outcome: Optional[str] = None  # WIN, LOSS, PENDING
    outcome_price: Optional[float] = None
    outcome_time: Optional[str] = None


def load_history() -> List[Dict]:
    """Sinyal geçmişini yükle."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"History load error: {e}")
    return []


def save_history(history: List[Dict]):
    """Geçmişi kaydet."""
    try:
        # Son MAX_HISTORY kaydı tut
        history = history[-MAX_HISTORY:]
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"History save error: {e}")


def add_signal(signal: SignalRecord):
    """Yeni sinyal ekle."""
    history = load_history()
    history.append(asdict(signal))
    save_history(history)
    logger.info(f"[HISTORY] Signal added: {signal.symbol} {signal.action}")


def get_recent_signals(count: int = 10) -> List[Dict]:
    """Son N sinyali getir."""
    history = load_history()
    return history[-count:][::-1]  # En yeni başta


def update_outcome(symbol: str, timestamp: str, outcome: str, price: float):
    """Sinyal sonucunu güncelle."""
    history = load_history()
    for signal in history:
        if signal['symbol'] == symbol and signal['timestamp'] == timestamp:
            signal['outcome'] = outcome
            signal['outcome_price'] = price
            signal['outcome_time'] = datetime.now().isoformat()
            break
    save_history(history)


def get_statistics() -> Dict:
    """İstatistikleri hesapla."""
    history = load_history()
    
    if not history:
        return {
            'total': 0,
            'wins': 0,
            'losses': 0,
            'pending': 0,
            'win_rate': 0,
            'avg_confidence': 0
        }
    
    wins = sum(1 for s in history if s.get('outcome') == 'WIN')
    losses = sum(1 for s in history if s.get('outcome') == 'LOSS')
    pending = sum(1 for s in history if s.get('outcome') in [None, 'PENDING'])
    
    completed = wins + losses
    win_rate = (wins / completed * 100) if completed > 0 else 0
    
    confidences = [s['confidence'] for s in history if s.get('confidence')]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'total': len(history),
        'wins': wins,
        'losses': losses,
        'pending': pending,
        'win_rate': round(win_rate, 1),
        'avg_confidence': round(avg_conf, 1)
    }


# Engine'den çağrılacak fonksiyon
def record_early_signal(early_signal, symbol: str):
    """Early Signal'i kaydet."""
    try:
        record = SignalRecord(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action=early_signal.action,
            confidence=early_signal.confidence,
            entry_price=early_signal.entry_zone[0],
            stop_loss=early_signal.stop_loss,
            take_profit=early_signal.take_profit,
            reasoning=early_signal.reasoning
        )
        add_signal(record)
    except Exception as e:
        logger.error(f"Record signal error: {e}")
