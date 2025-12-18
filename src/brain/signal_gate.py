# -*- coding: utf-8 -*-
"""
DEMIR AI - Smart Signal Gate
Coin başına tek aktif sinyal - TP/SL gelene kadar yeni sinyal yok.

PHASE 93: Smart Signal Gate System
- Coin başına 1 aktif sinyal
- TP/SL gelene kadar gate kapalı
- WebSocket ile her saniye fiyat kontrolü
- Sonuç kaydı ve bildirim
"""
import logging
import json
import os
from datetime import datetime
from typing import Dict, Optional
import requests

logger = logging.getLogger("SIGNAL_GATE")


class SmartSignalGate:
    """
    Akıllı Sinyal Kapısı
    
    Her coin için sadece 1 aktif sinyal olabilir.
    TP veya SL vurulana kadar yeni sinyal gönderilmez.
    """
    
    GATE_FILE = "signal_gate.json"
    
    def __init__(self):
        self.active_signals: Dict[str, Dict] = {}
        self._load_gates()
        logger.info("✅ Smart Signal Gate initialized")
    
    def _load_gates(self):
        """Mevcut gate durumlarını yükle."""
        try:
            if os.path.exists(self.GATE_FILE):
                with open(self.GATE_FILE, 'r') as f:
                    self.active_signals = json.load(f)
                logger.info(f"📂 Loaded {len(self.active_signals)} active signals")
        except Exception as e:
            logger.warning(f"Gate load failed: {e}")
            self.active_signals = {}
    
    def _save_gates(self):
        """Gate durumlarını kaydet."""
        try:
            with open(self.GATE_FILE, 'w') as f:
                json.dump(self.active_signals, f, indent=2)
        except Exception as e:
            logger.warning(f"Gate save failed: {e}")
    
    def can_send_signal(self, symbol: str) -> bool:
        """
        Bu coin için sinyal gönderilebilir mi?
        
        Returns:
            True = Gate açık, sinyal gönderilebilir
            False = Gate kapalı, aktif sinyal var
        """
        if symbol not in self.active_signals:
            return True
        
        active = self.active_signals[symbol]
        
        # Sinyal expired mı kontrol et (24 saat)
        created = datetime.fromisoformat(active['created_at'])
        if (datetime.now() - created).total_seconds() > 86400:  # 24 saat
            logger.info(f"🔓 Signal expired for {symbol}, gate opened")
            del self.active_signals[symbol]
            self._save_gates()
            return True
        
        return False
    
    def open_gate(self, symbol: str, signal: Dict) -> str:
        """
        Sinyal gönderildi, gate'i kapat.
        
        Args:
            symbol: BTCUSDT
            signal: {direction, entry, tp1, tp2, sl, confidence}
            
        Returns:
            signal_id
        """
        signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_signals[symbol] = {
            'id': signal_id,
            'symbol': symbol,
            'direction': signal.get('direction', 'LONG'),
            'entry': signal.get('entry', 0),
            'tp1': signal.get('tp1', 0),
            'tp2': signal.get('tp2', 0),
            'sl': signal.get('sl', 0),
            'confidence': signal.get('confidence', 50),
            'created_at': datetime.now().isoformat(),
            'status': 'ACTIVE'
        }
        
        self._save_gates()
        logger.info(f"🔒 Gate CLOSED for {symbol} - Signal {signal_id}")
        
        return signal_id
    
    def check_and_close(self, symbol: str) -> Optional[Dict]:
        """
        Fiyatı kontrol et, TP/SL vuruldu mu?
        
        Returns:
            None = Hala aktif
            Dict = Sonuç {status, profit_pct, signal}
        """
        if symbol not in self.active_signals:
            return None
        
        signal = self.active_signals[symbol]
        
        # Mevcut fiyatı al
        current_price = self._get_price(symbol)
        if current_price == 0:
            return None
        
        direction = signal['direction']
        entry = signal['entry']
        tp1 = signal['tp1']
        tp2 = signal['tp2']
        sl = signal['sl']
        
        result = None
        
        # LONG pozisyon kontrol
        if direction == 'LONG':
            if current_price >= tp2:
                result = {
                    'status': 'TP2_HIT',
                    'profit_pct': ((tp2 - entry) / entry) * 100,
                    'signal': signal,
                    'exit_price': current_price
                }
            elif current_price >= tp1:
                result = {
                    'status': 'TP1_HIT',
                    'profit_pct': ((tp1 - entry) / entry) * 100,
                    'signal': signal,
                    'exit_price': current_price
                }
            elif current_price <= sl:
                result = {
                    'status': 'SL_HIT',
                    'profit_pct': ((sl - entry) / entry) * 100,
                    'signal': signal,
                    'exit_price': current_price
                }
        
        # SHORT pozisyon kontrol
        elif direction == 'SHORT':
            if current_price <= tp2:
                result = {
                    'status': 'TP2_HIT',
                    'profit_pct': ((entry - tp2) / entry) * 100,
                    'signal': signal,
                    'exit_price': current_price
                }
            elif current_price <= tp1:
                result = {
                    'status': 'TP1_HIT',
                    'profit_pct': ((entry - tp1) / entry) * 100,
                    'signal': signal,
                    'exit_price': current_price
                }
            elif current_price >= sl:
                result = {
                    'status': 'SL_HIT',
                    'profit_pct': ((entry - sl) / entry) * 100,
                    'signal': signal,
                    'exit_price': current_price
                }
        
        # Sonuç varsa gate'i aç
        if result:
            logger.info(f"🎯 {symbol} {result['status']} - {result['profit_pct']:+.2f}%")
            del self.active_signals[symbol]
            self._save_gates()
            return result
        
        return None
    
    def _get_price(self, symbol: str) -> float:
        """Mevcut fiyatı al."""
        try:
            resp = requests.get(
                f"https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=2
            )
            if resp.status_code == 200:
                return float(resp.json()['price'])
        except:
            pass
        return 0
    
    def get_active_signals(self) -> Dict:
        """Tüm aktif sinyalleri getir."""
        return self.active_signals.copy()
    
    def force_close(self, symbol: str, reason: str = 'MANUAL') -> bool:
        """Manuel olarak gate'i aç."""
        if symbol in self.active_signals:
            logger.info(f"🔓 Force closed {symbol}: {reason}")
            del self.active_signals[symbol]
            self._save_gates()
            return True
        return False


# Global instance
_gate = None

def get_gate() -> SmartSignalGate:
    """Get or create gate instance."""
    global _gate
    if _gate is None:
        _gate = SmartSignalGate()
    return _gate
