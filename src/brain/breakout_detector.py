# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - BREAKOUT DETECTOR
=================================
Anlık breakout ve volume spike tespiti.

ÖZELLİKLER:
- Volume Spike (3x ortalama)
- Price Breakout (key level)
- Momentum Burst (5 consecutive green/red)
- Real-time WebSocket desteği
"""
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger("BREAKOUT_DETECTOR")


@dataclass
class BreakoutAlert:
    """Breakout alert"""
    symbol: str
    alert_type: str  # VOLUME_SPIKE, PRICE_BREAKOUT, MOMENTUM_BURST
    direction: str  # BULLISH, BEARISH
    severity: str  # HIGH, MEDIUM, LOW
    current_price: float
    trigger_level: float
    message: str
    timestamp: datetime


class BreakoutDetector:
    """
    Anlık Breakout Tespit Sistemi
    
    Her 1 dakikada bir kontrol:
    1. Volume Spike: Volume > 3x ortalama
    2. Price Breakout: Fiyat key level'ı kırdı
    3. Momentum Burst: 5+ ardışık aynı yön mum
    """
    
    FUTURES_BASE = "https://fapi.binance.com"
    SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    
    # Thresholds
    VOLUME_SPIKE_MULTIPLIER = 3.0  # 3x ortalama
    MOMENTUM_CANDLES = 5  # 5 ardışık mum
    PRICE_MOVE_THRESHOLD = 0.5  # %0.5 ani hareket
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._callback: Optional[Callable] = None
        self._running = False
        self._last_alerts: Dict[str, datetime] = {}  # Spam önleme
        self._cooldown = 300  # 5 dakika cooldown
        
        # Geçmiş veriler (key level tespiti için)
        self._price_history: Dict[str, List[float]] = {}
        self._key_levels: Dict[str, Dict] = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    def set_callback(self, callback: Callable):
        """Alert callback ayarla"""
        self._callback = callback
    
    async def start_monitoring(self):
        """Monitoring başlat"""
        self._running = True
        logger.info("🔍 Breakout Detector started")
        
        while self._running:
            try:
                for symbol in self.SYMBOLS:
                    alerts = await self._check_breakouts(symbol)
                    
                    for alert in alerts:
                        if self._can_send_alert(alert):
                            await self._send_alert(alert)
                
                await asyncio.sleep(60)  # 1 dakika
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)
    
    def stop_monitoring(self):
        """Monitoring durdur"""
        self._running = False
    
    async def _check_breakouts(self, symbol: str) -> List[BreakoutAlert]:
        """Tüm breakout tiplerini kontrol et"""
        alerts = []
        
        try:
            # 1 dakikalık mumları çek
            klines_1m = await self._fetch_klines(symbol, "1m", 60)
            klines_15m = await self._fetch_klines(symbol, "15m", 100)
            
            if not klines_1m or not klines_15m:
                return alerts
            
            current_price = float(klines_1m[-1][4])
            
            # 1. Volume Spike kontrolü
            volume_alert = self._check_volume_spike(symbol, klines_1m, current_price)
            if volume_alert:
                alerts.append(volume_alert)
            
            # 2. Price Breakout kontrolü
            breakout_alert = self._check_price_breakout(symbol, klines_15m, current_price)
            if breakout_alert:
                alerts.append(breakout_alert)
            
            # 3. Momentum Burst kontrolü
            momentum_alert = self._check_momentum_burst(symbol, klines_1m, current_price)
            if momentum_alert:
                alerts.append(momentum_alert)
            
        except Exception as e:
            logger.debug(f"Check error {symbol}: {e}")
        
        return alerts
    
    def _check_volume_spike(self, symbol: str, klines: List, current_price: float) -> Optional[BreakoutAlert]:
        """Volume spike kontrolü"""
        if len(klines) < 30:
            return None
        
        volumes = [float(k[5]) for k in klines]
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
        last_volume = volumes[-1]
        
        if last_volume > avg_volume * self.VOLUME_SPIKE_MULTIPLIER:
            # Yön belirleme
            price_change = (float(klines[-1][4]) - float(klines[-2][4])) / float(klines[-2][4])
            direction = "BULLISH" if price_change > 0 else "BEARISH"
            
            multiplier = last_volume / avg_volume
            severity = "HIGH" if multiplier > 5 else "MEDIUM" if multiplier > 3 else "LOW"
            
            return BreakoutAlert(
                symbol=symbol,
                alert_type="VOLUME_SPIKE",
                direction=direction,
                severity=severity,
                current_price=current_price,
                trigger_level=multiplier,
                message=f"📊 {symbol} Volume Spike: {multiplier:.1f}x ortalama!",
                timestamp=datetime.now()
            )
        
        return None
    
    def _check_price_breakout(self, symbol: str, klines: List, current_price: float) -> Optional[BreakoutAlert]:
        """Key level breakout kontrolü"""
        if len(klines) < 50:
            return None
        
        # Key levels hesapla
        highs = [float(k[2]) for k in klines[:-1]]
        lows = [float(k[3]) for k in klines[:-1]]
        
        # Son 50 mum içindeki en yüksek high ve en düşük low
        resistance = max(highs[-20:])
        support = min(lows[-20:])
        
        # Breakout kontrolü
        prev_close = float(klines[-2][4])
        
        # Resistance kırılması
        if current_price > resistance and prev_close <= resistance:
            pct_above = (current_price - resistance) / resistance * 100
            if pct_above > 0.1:  # %0.1+ kırılma
                return BreakoutAlert(
                    symbol=symbol,
                    alert_type="PRICE_BREAKOUT",
                    direction="BULLISH",
                    severity="HIGH" if pct_above > 0.5 else "MEDIUM",
                    current_price=current_price,
                    trigger_level=resistance,
                    message=f"🚀 {symbol} Resistance Breakout: ${resistance:,.0f} kırıldı!",
                    timestamp=datetime.now()
                )
        
        # Support kırılması
        if current_price < support and prev_close >= support:
            pct_below = (support - current_price) / support * 100
            if pct_below > 0.1:
                return BreakoutAlert(
                    symbol=symbol,
                    alert_type="PRICE_BREAKOUT",
                    direction="BEARISH",
                    severity="HIGH" if pct_below > 0.5 else "MEDIUM",
                    current_price=current_price,
                    trigger_level=support,
                    message=f"📉 {symbol} Support Break: ${support:,.0f} kırıldı!",
                    timestamp=datetime.now()
                )
        
        return None
    
    def _check_momentum_burst(self, symbol: str, klines: List, current_price: float) -> Optional[BreakoutAlert]:
        """Momentum burst kontrolü (ardışık aynı yön mumlar)"""
        if len(klines) < self.MOMENTUM_CANDLES + 1:
            return None
        
        recent = klines[-self.MOMENTUM_CANDLES:]
        
        # Ardışık yeşil (bullish) mum kontrolü
        all_green = all(float(k[4]) > float(k[1]) for k in recent)
        
        # Ardışık kırmızı (bearish) mum kontrolü
        all_red = all(float(k[4]) < float(k[1]) for k in recent)
        
        if all_green:
            total_move = (float(recent[-1][4]) - float(recent[0][1])) / float(recent[0][1]) * 100
            if abs(total_move) > self.PRICE_MOVE_THRESHOLD:
                return BreakoutAlert(
                    symbol=symbol,
                    alert_type="MOMENTUM_BURST",
                    direction="BULLISH",
                    severity="HIGH" if total_move > 1 else "MEDIUM",
                    current_price=current_price,
                    trigger_level=total_move,
                    message=f"🔥 {symbol} Momentum Burst: {self.MOMENTUM_CANDLES} ardışık yeşil mum (+{total_move:.2f}%)",
                    timestamp=datetime.now()
                )
        
        if all_red:
            total_move = (float(recent[-1][4]) - float(recent[0][1])) / float(recent[0][1]) * 100
            if abs(total_move) > self.PRICE_MOVE_THRESHOLD:
                return BreakoutAlert(
                    symbol=symbol,
                    alert_type="MOMENTUM_BURST",
                    direction="BEARISH",
                    severity="HIGH" if abs(total_move) > 1 else "MEDIUM",
                    current_price=current_price,
                    trigger_level=total_move,
                    message=f"❄️ {symbol} Momentum Burst: {self.MOMENTUM_CANDLES} ardışık kırmızı mum ({total_move:.2f}%)",
                    timestamp=datetime.now()
                )
        
        return None
    
    def _can_send_alert(self, alert: BreakoutAlert) -> bool:
        """Spam önleme - aynı tip alert için cooldown"""
        key = f"{alert.symbol}_{alert.alert_type}"
        
        if key in self._last_alerts:
            elapsed = (datetime.now() - self._last_alerts[key]).total_seconds()
            if elapsed < self._cooldown:
                return False
        
        self._last_alerts[key] = datetime.now()
        return True
    
    async def _send_alert(self, alert: BreakoutAlert):
        """Alert gönder"""
        if self._callback:
            msg = self._format_alert(alert)
            try:
                await self._callback(msg)
                logger.info(f"⚡ Alert sent: {alert.symbol} {alert.alert_type}")
            except Exception as e:
                logger.error(f"Alert send error: {e}")
    
    def _format_alert(self, alert: BreakoutAlert) -> str:
        """Telegram formatında alert mesajı"""
        severity_emoji = "🔴" if alert.severity == "HIGH" else "🟡" if alert.severity == "MEDIUM" else "🟢"
        direction_emoji = "📈" if alert.direction == "BULLISH" else "📉"
        
        return f"""⚡ *BREAKOUT ALERT*
━━━━━━━━━━━━━━━━━━━━━━━━

{direction_emoji} *{alert.symbol}* {alert.direction}
🏷️ Tip: {alert.alert_type}
{severity_emoji} Seviye: {alert.severity}

💰 Fiyat: ${alert.current_price:,.2f}
📊 Tetik: {alert.trigger_level:.2f}

💬 {alert.message}

━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {alert.timestamp.strftime('%H:%M:%S')}
"""
    
    async def _fetch_klines(self, symbol: str, interval: str, limit: int) -> List:
        """Binance'dan kline çek"""
        try:
            session = await self._get_session()
            url = f"{self.FUTURES_BASE}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug(f"Klines error: {e}")
        
        return []
    
    async def check_once(self, symbol: str = "BTCUSDT") -> List[BreakoutAlert]:
        """Tek seferlik kontrol (test için)"""
        return await self._check_breakouts(symbol)
    
    async def close(self):
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_detector: Optional[BreakoutDetector] = None

def get_breakout_detector() -> BreakoutDetector:
    global _detector
    if _detector is None:
        _detector = BreakoutDetector()
    return _detector


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        detector = get_breakout_detector()
        
        print("Checking BTCUSDT for breakouts...")
        alerts = await detector.check_once("BTCUSDT")
        
        if alerts:
            for alert in alerts:
                print(detector._format_alert(alert))
        else:
            print("No breakouts detected")
        
        print("\nChecking ETHUSDT...")
        alerts = await detector.check_once("ETHUSDT")
        
        if alerts:
            for alert in alerts:
                print(detector._format_alert(alert))
        else:
            print("No breakouts detected")
        
        await detector.close()
    
    asyncio.run(test())
