# -*- coding: utf-8 -*-
"""
DEMIR AI - INSTANT ALERT SYSTEM
================================
Ani fiyat hareketlerini ANINDA tespit eder.

Saatlik rapor genel analiz icin.
Bu sistem dakika icinde olan sert hareketleri yakalar:
- %2+ hareket 5 dakika icinde
- Buyuk hacim patlamasi
- Ani likidasyonlar

Author: DEMIR AI Core Team
Date: 2024-12
"""
import logging
import asyncio
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("INSTANT_ALERT")


@dataclass
class PriceAlert:
    """Ani fiyat hareketi uyarisi."""
    symbol: str
    alert_type: str  # 'SPIKE_UP', 'SPIKE_DOWN', 'VOLUME_EXPLOSION', 'LIQUIDATION_CASCADE'
    severity: str  # 'WARNING', 'CRITICAL', 'EXTREME'
    
    price_before: float
    price_now: float
    change_percent: float
    
    timeframe_minutes: int
    volume_ratio: float  # Normal hacme gore
    
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


class InstantAlertSystem:
    """
    Anlik Uyari Sistemi
    
    Her 30 saniyede fiyatlari kontrol eder.
    Ani hareket tespit ederse HEMEN bildirim gonderir.
    """
    
    COINS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT']
    
    # Esik degerleri
    THRESHOLDS = {
        'spike_5min': 1.5,    # 5 dakikada %1.5+ hareket
        'spike_15min': 2.5,   # 15 dakikada %2.5+ hareket
        'volume_spike': 3.0,  # Normal hacmin 3 kati
    }
    
    def __init__(self):
        # Fiyat gecmisi (her coin icin son 30 dakika)
        self.price_history: Dict[str, List[Dict]] = {coin: [] for coin in self.COINS}
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_callback: Optional[Callable] = None
        
        self.running = False
        logger.info("Instant Alert System initialized")
    
    def set_callback(self, callback: Callable):
        """Uyari gonderme fonksiyonunu ayarla."""
        self.alert_callback = callback
    
    async def start_monitoring(self):
        """Surekli izlemeyi baslat."""
        self.running = True
        logger.info("Real-time monitoring started...")
        
        while self.running:
            try:
                await self._check_all_coins()
                await asyncio.sleep(30)  # Her 30 saniyede kontrol
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    def stop(self):
        """Izlemeyi durdur."""
        self.running = False
    
    async def _check_all_coins(self):
        """Tum coinleri kontrol et."""
        for symbol in self.COINS:
            try:
                await self._check_coin(symbol)
            except Exception as e:
                logger.debug(f"Check failed for {symbol}: {e}")
    
    async def _check_coin(self, symbol: str):
        """Tek coini kontrol et."""
        # Guncel fiyat ve hacim al
        current = await self._get_current_data(symbol)
        if not current:
            return
        
        # Gecmise ekle
        self.price_history[symbol].append(current)
        
        # Son 30 dakikadan fazlasini sil
        cutoff = datetime.now() - timedelta(minutes=30)
        self.price_history[symbol] = [
            p for p in self.price_history[symbol]
            if p['timestamp'] > cutoff
        ]
        
        # Yeterli veri var mi?
        if len(self.price_history[symbol]) < 10:
            return
        
        # Ani hareket kontrolu
        alerts = []
        
        # 5 dakikalik hareket
        alert_5m = self._check_price_spike(symbol, minutes=5)
        if alert_5m:
            alerts.append(alert_5m)
        
        # 15 dakikalik hareket
        alert_15m = self._check_price_spike(symbol, minutes=15)
        if alert_15m:
            alerts.append(alert_15m)
        
        # Hacim patlamasi
        alert_vol = self._check_volume_spike(symbol)
        if alert_vol:
            alerts.append(alert_vol)
        
        # Uyarilari gonder
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _get_current_data(self, symbol: str) -> Optional[Dict]:
        """Guncel fiyat ve hacim al."""
        try:
            # Son 1 dakikanin verileri
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1m', 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200:
                k = resp.json()[0]
                return {
                    'timestamp': datetime.now(),
                    'price': float(k[4]),
                    'volume': float(k[5]),
                    'high': float(k[2]),
                    'low': float(k[3])
                }
        except:
            pass
        return None
    
    def _check_price_spike(self, symbol: str, minutes: int) -> Optional[PriceAlert]:
        """Fiyat sicramsini kontrol et."""
        history = self.price_history[symbol]
        
        # X dakika onceki fiyati bul
        cutoff = datetime.now() - timedelta(minutes=minutes)
        old_prices = [p for p in history if p['timestamp'] <= cutoff]
        
        if not old_prices:
            return None
        
        old_price = old_prices[-1]['price']
        current_price = history[-1]['price']
        
        change_pct = ((current_price / old_price) - 1) * 100
        
        # Esik kontrolu
        threshold = self.THRESHOLDS.get(f'spike_{minutes}min', 2.0)
        
        if abs(change_pct) < threshold:
            return None
        
        # Cok yakin zamanda ayni uyari gonderdik mi?
        alert_key = f"{symbol}_{minutes}m"
        if alert_key in self.last_alert_time:
            if (datetime.now() - self.last_alert_time[alert_key]).total_seconds() < 300:
                return None  # 5 dakikada bir uyari
        
        self.last_alert_time[alert_key] = datetime.now()
        
        # Uyari olustur
        if change_pct > 0:
            alert_type = 'SPIKE_UP'
            emoji = "🚀"
            direction = "YUKARI"
        else:
            alert_type = 'SPIKE_DOWN'
            emoji = "📉"
            direction = "ASAGI"
        
        # Siddet
        if abs(change_pct) > 5:
            severity = 'EXTREME'
            severity_emoji = "🔴🔴🔴"
        elif abs(change_pct) > 3:
            severity = 'CRITICAL'
            severity_emoji = "🔴🔴"
        else:
            severity = 'WARNING'
            severity_emoji = "🔴"
        
        message = f"""{severity_emoji} ANI HAREKET - {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━
{emoji} {minutes} dakikada {direction}: {change_pct:+.2f}%

📍 Onceki: ${old_price:,.2f}
📍 Simdi: ${current_price:,.2f}

{self._get_action_advice(alert_type, change_pct)}
━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        
        return PriceAlert(
            symbol=symbol,
            alert_type=alert_type,
            severity=severity,
            price_before=old_price,
            price_now=current_price,
            change_percent=change_pct,
            timeframe_minutes=minutes,
            volume_ratio=1.0,
            message=message
        )
    
    def _check_volume_spike(self, symbol: str) -> Optional[PriceAlert]:
        """Hacim patlamasini kontrol et."""
        history = self.price_history[symbol]
        
        if len(history) < 20:
            return None
        
        # Son 5 dakikanin hacmi vs onceki 15 dakika
        recent = [p['volume'] for p in history[-5:]]
        older = [p['volume'] for p in history[-20:-5]]
        
        if not older or sum(older) == 0:
            return None
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        ratio = recent_avg / (older_avg + 0.001)
        
        if ratio < self.THRESHOLDS['volume_spike']:
            return None
        
        # Cok yakin zamanda ayni uyari gonderdik mi?
        alert_key = f"{symbol}_volume"
        if alert_key in self.last_alert_time:
            if (datetime.now() - self.last_alert_time[alert_key]).total_seconds() < 600:
                return None  # 10 dakikada bir uyari
        
        self.last_alert_time[alert_key] = datetime.now()
        
        current_price = history[-1]['price']
        
        message = f"""📊 HACIM PATLAMASI - {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━
⚡ Hacim normalin {ratio:.1f} KATI!

📍 Fiyat: ${current_price:,.2f}

💡 Buyuk oyuncular harekete gecti!
Ani fiyat hareketi olabilir.

━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        
        return PriceAlert(
            symbol=symbol,
            alert_type='VOLUME_EXPLOSION',
            severity='WARNING',
            price_before=current_price,
            price_now=current_price,
            change_percent=0,
            timeframe_minutes=5,
            volume_ratio=ratio,
            message=message
        )
    
    def _get_action_advice(self, alert_type: str, change_pct: float) -> str:
        """Aksiyon onerisi."""
        if alert_type == 'SPIKE_UP':
            if change_pct > 5:
                return "💡 ASIRI hizli yukselis! Dusus icin bekle veya kismi kar al."
            elif change_pct > 3:
                return "💡 Guclu yukselis. Momentum devam edebilir ama dikkatli ol."
            else:
                return "💡 Orta seviye yukselis. Trendi takip et."
        else:
            if change_pct < -5:
                return "💡 SERT dusus! Panik satma. Destek seviyelerini bekle."
            elif change_pct < -3:
                return "💡 Onemli dusus. Stop-loss kontrol et."
            else:
                return "💡 Orta seviye dusus. Ana trend hala yukariysa firsat olabilir."
    
    async def _send_alert(self, alert: PriceAlert):
        """Uyariyi gonder."""
        logger.warning(f"INSTANT ALERT: {alert.symbol} {alert.alert_type} {alert.change_percent:+.2f}%")
        
        if self.alert_callback:
            try:
                await self.alert_callback(alert.message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_alert_system: Optional[InstantAlertSystem] = None

def get_instant_alert_system() -> InstantAlertSystem:
    """Get instance."""
    global _alert_system
    if _alert_system is None:
        _alert_system = InstantAlertSystem()
    return _alert_system


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    import io
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        system = get_instant_alert_system()
        
        # Simule edilmis callback
        async def print_alert(msg):
            print(msg)
        
        system.set_callback(print_alert)
        
        print("Checking all coins once...")
        await system._check_all_coins()
        
        print("\nPrice history collected:")
        for symbol, history in system.price_history.items():
            if history:
                print(f"  {symbol}: {len(history)} data points, last price: ${history[-1]['price']:,.2f}")
    
    asyncio.run(test())
