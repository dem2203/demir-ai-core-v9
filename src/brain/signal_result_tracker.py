# -*- coding: utf-8 -*-
"""
DEMIR AI - Signal Result Tracker
Sinyal sonuçlarını otomatik takip eder.

PHASE 117: Self-Learning - Signal Tracking
- Aktif sinyalleri kontrol et
- TP/SL vurulduğunda sonucu kaydet
- ML eğitimi için veri topla
"""
import logging
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("SIGNAL_TRACKER")


class SignalResultTracker:
    """
    Sinyal Sonuç Takipçisi
    
    Aktif sinyallerin TP/SL durumunu kontrol eder.
    Sonuçları veritabanına kaydeder.
    """
    
    def __init__(self):
        self.last_check = datetime.now()
        logger.info("✅ Signal Result Tracker initialized")
    
    async def check_active_signals(self) -> List[Dict]:
        """Aktif sinyalleri kontrol et ve sonuçları güncelle."""
        try:
            from src.brain.signal_database import get_signal_database
            db = get_signal_database()
            
            active_signals = db.get_active_signals()
            results = []
            
            for signal in active_signals:
                result = await self._check_signal(signal, db)
                if result:
                    results.append(result)
            
            if results:
                logger.info(f"📊 Checked {len(active_signals)} signals, {len(results)} closed")
            
            return results
            
        except Exception as e:
            logger.error(f"Signal check failed: {e}")
            return []
    
    async def _check_signal(self, signal: Dict, db) -> Optional[Dict]:
        """Tek bir sinyali kontrol et."""
        try:
            symbol = signal.get('symbol', 'BTCUSDT')
            entry_price = signal.get('entry_price', 0)
            tp_price = signal.get('tp_price', 0)
            sl_price = signal.get('sl_price', 0)
            direction = signal.get('direction', 'LONG')
            signal_id = signal.get('id')
            
            if not entry_price or not signal_id:
                return None
            
            # Mevcut fiyatı al
            current_price = await self._get_current_price(symbol)
            if current_price <= 0:
                return None
            
            # Zaman kontrolü - 24 saatten eski sinyalleri kapat
            signal_time = datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat()))
            if datetime.now() - signal_time > timedelta(hours=24):
                # Timeout - mevcut fiyatla kapat
                pnl_pct = self._calculate_pnl(entry_price, current_price, direction)
                result = 'WIN' if pnl_pct > 0 else 'LOSS'
                
                db.update_signal_result(signal_id, result, current_price, pnl_pct)
                
                return {
                    'signal_id': signal_id,
                    'symbol': symbol,
                    'result': result,
                    'reason': 'TIMEOUT',
                    'pnl_pct': pnl_pct,
                    'exit_price': current_price
                }
            
            # LONG pozisyon kontrolü
            if direction == 'LONG':
                if current_price >= tp_price:
                    # TP vuruldu
                    pnl_pct = self._calculate_pnl(entry_price, current_price, direction)
                    db.update_signal_result(signal_id, 'WIN', current_price, pnl_pct)
                    
                    return {
                        'signal_id': signal_id,
                        'symbol': symbol,
                        'result': 'WIN',
                        'reason': 'TP_HIT',
                        'pnl_pct': pnl_pct,
                        'exit_price': current_price
                    }
                    
                elif current_price <= sl_price:
                    # SL vuruldu
                    pnl_pct = self._calculate_pnl(entry_price, current_price, direction)
                    db.update_signal_result(signal_id, 'LOSS', current_price, pnl_pct)
                    
                    return {
                        'signal_id': signal_id,
                        'symbol': symbol,
                        'result': 'LOSS',
                        'reason': 'SL_HIT',
                        'pnl_pct': pnl_pct,
                        'exit_price': current_price
                    }
            
            # SHORT pozisyon kontrolü
            elif direction == 'SHORT':
                if current_price <= tp_price:
                    # TP vuruldu (short için fiyat düşmeli)
                    pnl_pct = self._calculate_pnl(entry_price, current_price, direction)
                    db.update_signal_result(signal_id, 'WIN', current_price, pnl_pct)
                    
                    return {
                        'signal_id': signal_id,
                        'symbol': symbol,
                        'result': 'WIN',
                        'reason': 'TP_HIT',
                        'pnl_pct': pnl_pct,
                        'exit_price': current_price
                    }
                    
                elif current_price >= sl_price:
                    # SL vuruldu
                    pnl_pct = self._calculate_pnl(entry_price, current_price, direction)
                    db.update_signal_result(signal_id, 'LOSS', current_price, pnl_pct)
                    
                    return {
                        'signal_id': signal_id,
                        'symbol': symbol,
                        'result': 'LOSS',
                        'reason': 'SL_HIT',
                        'pnl_pct': pnl_pct,
                        'exit_price': current_price
                    }
            
            # Henüz kapanmadı
            return None
            
        except Exception as e:
            logger.debug(f"Signal check error: {e}")
            return None
    
    def _calculate_pnl(self, entry: float, exit: float, direction: str) -> float:
        """PnL yüzdesini hesapla."""
        if entry <= 0:
            return 0
        
        if direction == 'LONG':
            return ((exit - entry) / entry) * 100
        else:  # SHORT
            return ((entry - exit) / entry) * 100
    
    async def _get_current_price(self, symbol: str) -> float:
        """Mevcut fiyatı al."""
        try:
            resp = requests.get(
                f"https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            if resp.status_code == 200:
                return float(resp.json()['price'])
        except:
            pass
        return 0
    
    async def send_result_notification(self, result: Dict):
        """Sonuç bildirimi gönder."""
        try:
            from src.utils.notifications import get_notifier
            notifier = get_notifier()
            
            emoji = "✅" if result['result'] == 'WIN' else "❌"
            reason_text = {
                'TP_HIT': 'Kar Al vuruldu',
                'SL_HIT': 'Zarar Kes vuruldu',
                'TIMEOUT': '24 saat doldu'
            }.get(result['reason'], result['reason'])
            
            msg = f"""
{emoji} SİNYAL SONUCU
━━━━━━━━━━━━━━━━━━━━━━
{result['symbol']} | #{result['signal_id']}
{emoji} {result['result']} | {reason_text}
💰 PnL: {result['pnl_pct']:+.2f}%
💵 Çıkış: ${result['exit_price']:,.2f}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
            
            await notifier.send_message_raw(msg)
            
        except Exception as e:
            logger.debug(f"Result notification failed: {e}")


# Global instance
_tracker = None

def get_signal_tracker() -> SignalResultTracker:
    """Get or create signal tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = SignalResultTracker()
    return _tracker
