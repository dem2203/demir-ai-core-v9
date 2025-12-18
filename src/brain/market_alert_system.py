# -*- coding: utf-8 -*-
"""
DEMIR AI - Market Alert System
Ani hareket ve fırsat tespiti - pozisyon olmadan da çalışır.

PHASE 101: Market Alert System
- Ani düşüş/yükseliş tespiti (±3% in 15min)
- Fırsat tespiti (oversold/overbought)
- Spam önleme (1 saat cooldown)
- WebSocket ile gerçek zamanlı
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

logger = logging.getLogger("MARKET_ALERT")


class MarketAlertSystem:
    """
    Piyasa Uyarı Sistemi
    
    Pozisyon olmadan da çalışır.
    Ani hareketleri ve fırsatları tespit eder.
    """
    
    ALERT_FILE = "market_alerts.json"
    
    # Thresholds
    SUDDEN_MOVE_THRESHOLD = 3.0  # %3 hareket
    SUDDEN_MOVE_WINDOW = 15  # Dakika
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    VOLUME_SPIKE_MULTIPLIER = 3.0  # 3x normal hacim
    
    # Cooldowns (dakika)
    COOLDOWN_SUDDEN_MOVE = 60  # 1 saat
    COOLDOWN_OPPORTUNITY = 120  # 2 saat
    
    def __init__(self):
        self.last_alerts: Dict[str, datetime] = {}
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._load_data()
        logger.info("✅ Market Alert System initialized")
    
    def _load_data(self):
        """Mevcut verileri yükle."""
        try:
            if os.path.exists(self.ALERT_FILE):
                with open(self.ALERT_FILE, 'r') as f:
                    data = json.load(f)
                    for key, ts in data.get('last_alerts', {}).items():
                        self.last_alerts[key] = datetime.fromisoformat(ts)
        except Exception as e:
            logger.warning(f"Alert data load failed: {e}")
    
    def _save_data(self):
        """Verileri kaydet."""
        try:
            with open(self.ALERT_FILE, 'w') as f:
                json.dump({
                    'last_alerts': {k: v.isoformat() for k, v in self.last_alerts.items()},
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Alert data save failed: {e}")
    
    def _can_send_alert(self, alert_type: str, cooldown_minutes: int) -> bool:
        """Cooldown kontrolü."""
        if alert_type not in self.last_alerts:
            return True
        
        minutes_since = (datetime.now() - self.last_alerts[alert_type]).total_seconds() / 60
        return minutes_since >= cooldown_minutes
    
    def _mark_alert_sent(self, alert_type: str):
        """Alert gönderildi olarak işaretle."""
        self.last_alerts[alert_type] = datetime.now()
        self._save_data()
    
    async def check_sudden_movement(self, symbol: str = 'BTCUSDT') -> Optional[Dict]:
        """
        Ani hareket kontrolü.
        
        Returns:
            Alert dict if sudden movement detected, None otherwise
        """
        alert_key = f"sudden_{symbol}"
        
        if not self._can_send_alert(alert_key, self.COOLDOWN_SUDDEN_MOVE):
            return None
        
        try:
            # Son 15 dakika mumlarını al
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1m', 'limit': 20},
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            klines = resp.json()
            if len(klines) < 15:
                return None
            
            # 15 dakika önceki fiyat
            price_15m_ago = float(klines[-15][4])  # close
            current_price = float(klines[-1][4])
            
            # Değişim hesapla
            change_pct = ((current_price - price_15m_ago) / price_15m_ago) * 100
            
            # Hacim analizi
            volumes = [float(k[5]) for k in klines[-15:]]
            avg_volume = sum(volumes) / len(volumes)
            recent_volume = sum(volumes[-3:]) / 3
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Ani düşüş
            if change_pct < -self.SUDDEN_MOVE_THRESHOLD:
                self._mark_alert_sent(alert_key)
                
                # Ek analizler
                reasons = [f"📉 Son 15dk: {change_pct:.1f}% düşüş"]
                
                if volume_ratio > 2:
                    reasons.append(f"📊 Hacim {volume_ratio:.1f}x normal")
                
                # Destek/direnç
                low_15m = min([float(k[3]) for k in klines[-15:]])
                reasons.append(f"📍 15dk low: ${low_15m:,.0f}")
                
                return {
                    'type': 'SUDDEN_DROP',
                    'symbol': symbol,
                    'change_pct': change_pct,
                    'current_price': current_price,
                    'volume_ratio': volume_ratio,
                    'reasons': reasons,
                    'severity': 'HIGH' if change_pct < -5 else 'MEDIUM'
                }
            
            # Ani yükseliş
            elif change_pct > self.SUDDEN_MOVE_THRESHOLD:
                self._mark_alert_sent(alert_key)
                
                reasons = [f"📈 Son 15dk: +{change_pct:.1f}% yükseliş"]
                
                if volume_ratio > 2:
                    reasons.append(f"📊 Hacim {volume_ratio:.1f}x normal")
                
                high_15m = max([float(k[2]) for k in klines[-15:]])
                reasons.append(f"📍 15dk high: ${high_15m:,.0f}")
                
                return {
                    'type': 'SUDDEN_PUMP',
                    'symbol': symbol,
                    'change_pct': change_pct,
                    'current_price': current_price,
                    'volume_ratio': volume_ratio,
                    'reasons': reasons,
                    'severity': 'HIGH' if change_pct > 5 else 'MEDIUM'
                }
                
        except Exception as e:
            logger.debug(f"Sudden movement check failed: {e}")
        
        return None
    
    async def check_opportunity(self, symbol: str = 'BTCUSDT') -> Optional[Dict]:
        """
        Fırsat tespiti (oversold/overbought).
        
        Returns:
            Alert dict if opportunity detected, None otherwise
        """
        alert_key = f"opportunity_{symbol}"
        
        if not self._can_send_alert(alert_key, self.COOLDOWN_OPPORTUNITY):
            return None
        
        try:
            # RSI hesapla (14 periyot, 1h mumlar)
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1h', 'limit': 20},
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            klines = resp.json()
            closes = [float(k[4]) for k in klines]
            
            # RSI hesapla
            rsi = self._calculate_rsi(closes, 14)
            if rsi is None:
                return None
            
            current_price = closes[-1]
            
            # Oversold fırsat
            if rsi < self.RSI_OVERSOLD:
                self._mark_alert_sent(alert_key)
                
                # Destek seviyesi
                recent_low = min(closes[-7:])
                
                return {
                    'type': 'OVERSOLD_OPPORTUNITY',
                    'symbol': symbol,
                    'rsi': rsi,
                    'current_price': current_price,
                    'support': recent_low,
                    'reasons': [
                        f"📊 RSI: {rsi:.0f} (aşırı satım)",
                        f"📍 Destek: ${recent_low:,.0f}",
                        "🔄 Bounce potansiyeli yüksek"
                    ],
                    'severity': 'MEDIUM'
                }
            
            # Overbought uyarı
            elif rsi > self.RSI_OVERBOUGHT:
                self._mark_alert_sent(alert_key)
                
                recent_high = max(closes[-7:])
                
                return {
                    'type': 'OVERBOUGHT_WARNING',
                    'symbol': symbol,
                    'rsi': rsi,
                    'current_price': current_price,
                    'resistance': recent_high,
                    'reasons': [
                        f"📊 RSI: {rsi:.0f} (aşırı alım)",
                        f"📍 Direnç: ${recent_high:,.0f}",
                        "⚠️ Düzeltme riski yüksek"
                    ],
                    'severity': 'MEDIUM'
                }
                
        except Exception as e:
            logger.debug(f"Opportunity check failed: {e}")
        
        return None
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """RSI hesapla."""
        if len(prices) < period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def format_sudden_move_alert(self, alert: Dict) -> str:
        """Ani hareket uyarısı formatla."""
        symbol = alert['symbol']
        change = alert['change_pct']
        price = alert['current_price']
        reasons = alert['reasons']
        severity = alert['severity']
        
        if alert['type'] == 'SUDDEN_DROP':
            emoji = "🔴"
            title = "ANİ DÜŞÜŞ"
            direction = "📉"
        else:
            emoji = "🟢"
            title = "ANİ YÜKSELİŞ"
            direction = "📈"
        
        severity_emoji = "🚨🚨" if severity == 'HIGH' else "⚠️"
        
        reasons_text = ""
        for r in reasons:
            reasons_text += f"• {r}\n"
        
        msg = f"""
{severity_emoji} **{title} - {symbol}** {severity_emoji}
━━━━━━━━━━━━━━━━━━━━━━
{direction} **{change:+.1f}%** son 15 dakikada
💰 Şimdi: **${price:,.2f}**
━━━━━━━━━━━━━━━━━━━━━━
**Tespit:**
{reasons_text.strip()}
━━━━━━━━━━━━━━━━━━━━━━
⚡ Bu bir piyasa uyarısıdır
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg
    
    def format_opportunity_alert(self, alert: Dict) -> str:
        """Fırsat uyarısı formatla."""
        symbol = alert['symbol']
        rsi = alert['rsi']
        price = alert['current_price']
        reasons = alert['reasons']
        
        if alert['type'] == 'OVERSOLD_OPPORTUNITY':
            emoji = "🟢"
            title = "FIRSAT TESPİTİ"
            action = "LONG fırsatı olabilir"
        else:
            emoji = "🟠"
            title = "OVERBOUGHT UYARISI"
            action = "SHORT fırsatı veya çıkış zamanı"
        
        reasons_text = ""
        for r in reasons:
            reasons_text += f"• {r}\n"
        
        msg = f"""
{emoji} **{title} - {symbol}**
━━━━━━━━━━━━━━━━━━━━━━
💰 Fiyat: **${price:,.2f}**
📊 RSI: **{rsi:.0f}**
━━━━━━━━━━━━━━━━━━━━━━
**Analiz:**
{reasons_text.strip()}
━━━━━━━━━━━━━━━━━━━━━━
💡 _{action}_
🔔 Sinyal için bekleyin...
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_alert_system = None

def get_alert_system() -> MarketAlertSystem:
    """Get or create alert system instance."""
    global _alert_system
    if _alert_system is None:
        _alert_system = MarketAlertSystem()
    return _alert_system
