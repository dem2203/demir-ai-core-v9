# -*- coding: utf-8 -*-
"""
DEMIR AI - CONTINUOUS MARKET MONITOR
Phase 129: Sürekli Piyasa Takibi

ÖZELLİKLER:
1. 4 coin takibi (BTC, ETH, LTC, SOL)
2. WebSocket entegrasyonu (büyük trade tespiti)
3. Fırsat/Risk sürekli tarama
4. Anti-spam cooldown sistemi
5. Akıllı bildirim yönetimi
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("CONTINUOUS_MONITOR")


@dataclass
class AlertCooldown:
    """Alert cooldown tracker - spam önleme."""
    # Format: {symbol: {alert_type: last_sent_time}}
    cooldowns: Dict[str, Dict[str, datetime]] = field(default_factory=lambda: defaultdict(dict))
    
    # Cooldown süreleri (dakika)
    COOLDOWN_TIMES = {
        'risk_alert': 30,       # Risk uyarısı: 30dk
        'opportunity': 30,      # Fırsat: 30dk
        'whale_trade': 15,      # Büyük işlem: 15dk
        'direction_change': 60, # Yön değişimi: 1 saat
        'reasoning': 15,        # AI Reasoning: 15dk
        'sudden_move': 20,      # Ani hareket: 20dk
        'default': 30           # Varsayılan: 30dk
    }
    
    def can_send(self, symbol: str, alert_type: str) -> bool:
        """Bu alert gönderilebilir mi?"""
        last_sent = self.cooldowns[symbol].get(alert_type)
        if not last_sent:
            return True
        
        cooldown_mins = self.COOLDOWN_TIMES.get(alert_type, self.COOLDOWN_TIMES['default'])
        elapsed = (datetime.now() - last_sent).total_seconds() / 60
        
        return elapsed >= cooldown_mins
    
    def mark_sent(self, symbol: str, alert_type: str):
        """Alert gönderildi olarak işaretle."""
        self.cooldowns[symbol][alert_type] = datetime.now()
    
    def get_remaining(self, symbol: str, alert_type: str) -> int:
        """Kalan cooldown süresi (dakika)."""
        last_sent = self.cooldowns[symbol].get(alert_type)
        if not last_sent:
            return 0
        
        cooldown_mins = self.COOLDOWN_TIMES.get(alert_type, self.COOLDOWN_TIMES['default'])
        elapsed = (datetime.now() - last_sent).total_seconds() / 60
        remaining = cooldown_mins - elapsed
        
        return max(0, int(remaining))


class ContinuousMonitor:
    """
    SÜREKLİ PİYASA TAKİP SİSTEMİ
    
    - 4 coin takibi
    - WebSocket büyük trade tespiti
    - Fırsat/Risk sürekli tarama
    - Anti-spam cooldown
    """
    
    # Takip edilen coinler
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT']
    
    def __init__(self, notification_callback: Optional[Callable] = None):
        self.cooldown = AlertCooldown()
        self.notification_callback = notification_callback
        
        # WebSocket stream
        self.ws_stream = None
        self.ws_running = False
        
        # Son durumlar (değişim tespiti için)
        self.last_states = {}
        self.last_scan_time = datetime.now() - timedelta(hours=1)
        
        # Fırsat/Risk geçmişi
        self.opportunity_history = defaultdict(list)
        self.risk_history = defaultdict(list)
        
        logger.info(f"✅ Continuous Monitor initialized for {len(self.SYMBOLS)} coins")
    
    async def start_websocket(self):
        """WebSocket stream başlat - Hem SPOT hem FUTURES."""
        try:
            from src.brain.websocket_stream import get_websocket_stream
            
            # Callback fonksiyonu
            async def on_large_trade(data):
                await self.handle_large_trade(data)
            
            self.ws_stream = get_websocket_stream(callback=on_large_trade)
            self.ws_running = True
            
            # Her coin için SPOT ve FUTURES stream başlat
            for symbol in self.SYMBOLS:
                # SPOT stream
                asyncio.create_task(self.ws_stream.start(symbol.lower(), market="spot"))
                await asyncio.sleep(0.5)  # Rate limit için
                
                # FUTURES stream
                asyncio.create_task(self.ws_stream.start(symbol.lower(), market="futures"))
                await asyncio.sleep(0.5)
            
            logger.info(f"🔌 WebSocket streams started: {len(self.SYMBOLS)} coins x 2 markets (SPOT + FUTURES)")
            
        except Exception as e:
            logger.warning(f"WebSocket start failed: {e}")
            self.ws_running = False
    
    async def handle_large_trade(self, data: dict):
        """Büyük trade tespiti callback."""
        symbol = data.get('symbol', 'UNKNOWN')
        trade_type = data.get('type', 'UNKNOWN')  # WHALE_BUY, WHALE_SELL, etc.
        amount_usd = data.get('amount_usd', 0)
        price = data.get('price', 0)
        qty = data.get('qty', 0)
        market = data.get('market', 'SPOT')  # SPOT veya FUTURES
        
        # Minimum miktar kontrolü - $100K altındaki işlemleri gösterme
        if amount_usd < 100000:
            logger.debug(f"Trade too small: ${amount_usd:,.0f}")
            return
        
        # Cooldown kontrolü
        if not self.cooldown.can_send(symbol, 'whale_trade'):
            remaining = self.cooldown.get_remaining(symbol, 'whale_trade')
            logger.debug(f"🔇 Whale trade alert blocked for {symbol} ({remaining}dk remaining)")
            return
        
        # Bildirim oluştur
        if amount_usd >= 1000000:  # $1M+
            emoji = "🐋💰"
            size = "DEV"
        elif amount_usd >= 500000:
            emoji = "🐋"
            size = "BÜYÜK"
        else:
            emoji = "🦈"
            size = "ORTA"
        
        side = "ALIŞ" if 'BUY' in trade_type else "SATIŞ"
        market_emoji = "🔶" if market == "FUTURES" else "🔵"
        market_name = "VADELİ" if market == "FUTURES" else "SPOT"
        
        msg = f"""
{emoji} {size} İŞLEM TESPİTİ - {symbol}
━━━━━━━━━━━━━━━━━━━━━━
{market_emoji} Piyasa: {market_name}
💰 Miktar: ${amount_usd:,.0f}
📊 Yön: {side}
💵 Fiyat: ${price:,.2f}
📈 Adet: {qty:.4f}
⏰ {datetime.now().strftime('%H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━
💡 {side} baskısı artabilir!
"""
        
        await self.send_notification(msg)
        self.cooldown.mark_sent(symbol, 'whale_trade')
        logger.info(f"🐋 Whale alert sent: {symbol} {market} {side} ${amount_usd:,.0f}")
    
    async def scan_opportunities(self):
        """Fırsat taraması - tüm coinler için."""
        try:
            from src.brain.ai_reasoning_engine import get_reasoning_engine
            reasoning = get_reasoning_engine()
            
            for symbol in self.SYMBOLS:
                # Cooldown kontrolü
                if not self.cooldown.can_send(symbol, 'opportunity'):
                    continue
                
                # AI analizi
                prediction = await reasoning.think(symbol)
                
                if not prediction:
                    continue
                
                # GÜÇLÜ FIRSAT - %70+ güven
                if prediction.confidence >= 70 and prediction.direction != "YATAY":
                    # Daha önce aynı yönde bildirim gitti mi?
                    last_state = self.last_states.get(symbol, {})
                    if last_state.get('direction') == prediction.direction:
                        continue  # Aynı yön, skip
                    
                    # Fırsat bildirimi
                    msg = f"""
🎯 FIRSAT TESPİTİ - {symbol}
━━━━━━━━━━━━━━━━━━━━━━
{"🟢📈" if prediction.direction == "YUKARI" else "🔴📉"} YÖN: {prediction.direction}
📊 Güven: %{prediction.confidence:.0f}
💰 Fiyat: ${reasoning.context.current_price:,.2f}
━━━━━━━━━━━━━━━━━━━━━━
🎯 ANA SEBEP:
{prediction.primary_reason}
━━━━━━━━━━━━━━━━━━━━━━
🎯 HEDEF: ${prediction.target_price:,.2f}
🛡️ STOP: ${prediction.stop_loss:,.2f}
📊 R:R = 1:{prediction.rr_ratio:.1f}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
"""
                    await self.send_notification(msg)
                    self.cooldown.mark_sent(symbol, 'opportunity')
                    
                    # State güncelle
                    self.last_states[symbol] = {
                        'direction': prediction.direction,
                        'confidence': prediction.confidence,
                        'time': datetime.now()
                    }
                    
                    logger.info(f"🎯 Opportunity sent for {symbol}: {prediction.direction} {prediction.confidence:.0f}%")
                    
        except Exception as e:
            logger.error(f"Opportunity scan failed: {e}")
    
    async def scan_risks(self):
        """Risk taraması - tüm coinler için."""
        try:
            from src.brain.ai_reasoning_engine import get_reasoning_engine
            reasoning = get_reasoning_engine()
            
            for symbol in self.SYMBOLS:
                # Cooldown kontrolü
                if not self.cooldown.can_send(symbol, 'risk_alert'):
                    continue
                
                # Risk analizi
                risk_msg = await reasoning.generate_risk_alerts(symbol)
                
                if risk_msg:  # Sadece risk varsa
                    await self.send_notification(risk_msg)
                    self.cooldown.mark_sent(symbol, 'risk_alert')
                    logger.info(f"🚨 Risk alert sent for {symbol}")
                    
        except Exception as e:
            logger.error(f"Risk scan failed: {e}")
    
    async def check_direction_changes(self):
        """Yön değişimi tespiti."""
        try:
            from src.brain.ai_reasoning_engine import get_reasoning_engine
            reasoning = get_reasoning_engine()
            
            for symbol in self.SYMBOLS:
                # Cooldown kontrolü
                if not self.cooldown.can_send(symbol, 'direction_change'):
                    continue
                
                prediction = await reasoning.think(symbol)
                
                if not prediction:
                    continue
                
                # Önceki durumla karşılaştır
                last = self.last_states.get(symbol, {})
                last_direction = last.get('direction')
                
                # Yön değişti mi?
                if last_direction and last_direction != prediction.direction:
                    if prediction.direction != "YATAY":  # YATAY geçişi atla
                        
                        msg = f"""
🔄 YÖN DEĞİŞİMİ - {symbol}
━━━━━━━━━━━━━━━━━━━━━━
📉 Önceki: {last_direction}
📈 Yeni: {prediction.direction}
📊 Güven: %{prediction.confidence:.0f}
━━━━━━━━━━━━━━━━━━━━━━
💡 Piyasa yönü değişiyor!
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
"""
                        await self.send_notification(msg)
                        self.cooldown.mark_sent(symbol, 'direction_change')
                        
                        # State güncelle
                        self.last_states[symbol] = {
                            'direction': prediction.direction,
                            'confidence': prediction.confidence,
                            'time': datetime.now()
                        }
                        
                        logger.info(f"🔄 Direction change: {symbol} {last_direction} → {prediction.direction}")
                        
        except Exception as e:
            logger.error(f"Direction check failed: {e}")
    
    async def detect_sudden_movements(self):
        """
        ANİ HAREKET TESPİTİ
        
        Coinglass + Binance verileri ile ani hareket tespit:
        1. Volume Spike (3x+ hacim)
        2. OI Delta (hızlı değişim)
        3. Liquidation Cascade riski
        4. Anomaly Detection (ML)
        5. Whale Flow (borsa akışı)
        """
        try:
            for symbol in self.SYMBOLS:
                # Cooldown kontrolü
                if not self.cooldown.can_send(symbol, 'sudden_move'):
                    continue
                
                clean_symbol = symbol.replace('USDT', '')
                alerts = []
                severity = 0  # 0-100
                
                # 1. VOLUME SPIKE
                try:
                    from src.brain.volume_spike import detect_volume_spike
                    spike = detect_volume_spike(symbol)
                    if spike.get('spike_detected'):
                        severity += 25
                        alerts.append(f"📊 Hacim Spike: {spike['spike_strength']:.1f}x normal")
                except Exception:
                    pass
                
                # 2. OI DELTA (Ani değişim)
                try:
                    from src.brain.coinglass_oi_delta import get_oi_delta
                    oi = get_oi_delta(symbol)
                    if oi.get('velocity') == 'INCREASING' and oi.get('delta_1h_pct', 0) > 3:
                        severity += 20
                        alerts.append(f"📈 OI Artışı: +%{oi['delta_1h_pct']:.1f} (1h)")
                    elif oi.get('velocity') == 'DECREASING' and oi.get('delta_1h_pct', 0) < -3:
                        severity += 20
                        alerts.append(f"📉 OI Düşüşü: %{oi['delta_1h_pct']:.1f} (1h)")
                except Exception:
                    pass
                
                # 3. LIQUIDATION CASCADE RİSKİ
                try:
                    from src.brain.coinglass_liquidation import get_liquidation_levels
                    liq = get_liquidation_levels(clean_symbol)
                    if liq.get('cascade_risk') == 'HIGH':
                        severity += 30
                        alerts.append(f"🎯 Cascade Risk: YÜKSEK! Liq %{liq.get('distance_to_long_pct', 0):.1f} uzakta")
                    elif liq.get('cascade_risk') == 'MEDIUM':
                        severity += 15
                        alerts.append(f"⚠️ Cascade Risk: ORTA")
                except Exception:
                    pass
                
                # 4. WHALE FLOW (Borsa akışı)
                try:
                    from src.brain.coinglass_whale_alerts import get_whale_alerts
                    whale = get_whale_alerts()
                    if whale.get('oi_change_pct', 0) > 5:
                        severity += 15
                        alerts.append(f"🐋 Whale Giriş: OI +%{whale['oi_change_pct']:.1f}")
                    elif whale.get('oi_change_pct', 0) < -3:
                        severity += 15
                        alerts.append(f"🐋 Whale Çıkış: OI %{whale['oi_change_pct']:.1f}")
                except Exception:
                    pass
                
                # 5. FİYAT DEĞİŞİMİ KONTROLÜ
                try:
                    import requests
                    resp = requests.get(
                        "https://api.binance.com/api/v3/ticker/24hr",
                        params={'symbol': symbol},
                        timeout=5
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        price = float(data['lastPrice'])
                        change_1h = float(data.get('priceChangePercent', 0))
                        
                        if abs(change_1h) > 3:
                            severity += 25
                            direction = "📈 YUKARI" if change_1h > 0 else "📉 AŞAĞI"
                            alerts.append(f"{direction}: %{change_1h:.1f} (24h)")
                except Exception:
                    pass
                
                # ANİ HAREKET TESPİT EDİLDİ Mİ?
                if severity >= 40 and len(alerts) >= 2:
                    try:
                        current_price = float(data['lastPrice']) if 'data' in dir() else 0
                    except:
                        current_price = 0
                    
                    msg = f"""
⚡ ANİ HAREKET TESPİTİ - {symbol}
━━━━━━━━━━━━━━━━━━━━━━
💰 Fiyat: ${current_price:,.2f}
📊 Şiddet: %{severity}
━━━━━━━━━━━━━━━━━━━━━━
🔍 TESPİTLER:
"""
                    for alert in alerts:
                        msg += f"  • {alert}\n"
                    
                    msg += f"""
━━━━━━━━━━━━━━━━━━━━━━
⚠️ Volatilite artabilir!
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
                    await self.send_notification(msg)
                    self.cooldown.mark_sent(symbol, 'sudden_move')
                    logger.info(f"⚡ Sudden movement detected: {symbol} severity={severity}")
                    
        except Exception as e:
            logger.error(f"Sudden movement detection failed: {e}")
    
    async def run_continuous_scan(self):
        """
        ANA TARAMA DÖNGÜSÜ
        
        Her 5 dakikada bir:
        - Fırsat taraması
        - Risk taraması
        - Yön değişimi kontrolü
        
        Her 1 dakikada bir:
        - Ani hareket taraması
        """
        scan_interval = 300  # 5 dakika
        sudden_check_interval = 60  # 1 dakika
        last_sudden_check = datetime.now() - timedelta(minutes=2)
        
        while True:
            try:
                now = datetime.now()
                
                # Her dakika ani hareket kontrolü
                if (now - last_sudden_check).total_seconds() >= sudden_check_interval:
                    await self.detect_sudden_movements()
                    last_sudden_check = now
                
                # 5dk'da bir tam tarama
                if (now - self.last_scan_time).total_seconds() >= scan_interval:
                    logger.info("🔍 Starting continuous scan...")
                    
                    # Paralel tarama
                    await asyncio.gather(
                        self.scan_opportunities(),
                        self.scan_risks(),
                        self.check_direction_changes()
                    )
                    
                    self.last_scan_time = now
                    logger.info("✅ Continuous scan complete")
                
                await asyncio.sleep(30)  # Her 30 saniye kontrol
                
            except Exception as e:
                logger.error(f"Continuous scan error: {e}")
                await asyncio.sleep(60)
    
    async def send_notification(self, message: str):
        """Bildirim gönder - DISABLED (Legacy)."""
        # DISABLED: ThinkingBrain is the only voice now.
        # if self.notification_callback:
        #    await self.notification_callback(message)
        # else:
        logger.info(f"🔇 SILENCED LEGACY ALERT:\n{message}")
    
    def get_status(self) -> Dict:
        """Monitor durumu."""
        return {
            'websocket_running': self.ws_running,
            'symbols_tracked': self.SYMBOLS,
            'last_scan': self.last_scan_time.strftime('%H:%M:%S'),
            'active_cooldowns': {
                symbol: {
                    alert_type: self.cooldown.get_remaining(symbol, alert_type)
                    for alert_type in self.cooldown.COOLDOWN_TIMES.keys()
                    if self.cooldown.get_remaining(symbol, alert_type) > 0
                }
                for symbol in self.SYMBOLS
            },
            'last_states': {
                symbol: {
                    'direction': state.get('direction', 'N/A'),
                    'time': state.get('time', datetime.now()).strftime('%H:%M')
                }
                for symbol, state in self.last_states.items()
            }
        }


# Global instance
_continuous_monitor = None

def get_continuous_monitor(callback=None) -> ContinuousMonitor:
    """Get or create continuous monitor instance."""
    global _continuous_monitor
    if _continuous_monitor is None:
        _continuous_monitor = ContinuousMonitor(notification_callback=callback)
    return _continuous_monitor
