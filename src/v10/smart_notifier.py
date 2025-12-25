# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - SMART NOTIFIER
==============================
Telegram'a net, anlaşılır trading sinyalleri gönderir.

FORMAT:
- Entry aralığı
- TP1, TP2, TP3
- SL
- R/R oranı
- Güven skoru
- Reasoning (neden bu sinyal?)
- Risk seviyesi
"""
import logging
import requests
from datetime import datetime
from typing import Optional

from src.config.settings import Config

logger = logging.getLogger("SMART_NOTIFIER")


class SmartNotifier:
    """
    DEMIR AI v10 - Akıllı Telegram Bildirim Sistemi
    
    Net, anlaşılır formatla trading sinyalleri gönderir.
    """
    
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self._error_count = 0
        logger.info("📱 Smart Notifier initialized")
    
    def _send_message(self, text: str) -> bool:
        """
        Telegram'a mesaj gönder.
        HATA YUTULMAZ - başarısız olursa loglar.
        """
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"❌ Telegram API error: {response.status_code} - {response.text}")
                self._error_count += 1
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            self._error_count += 1
            return False
    
    def send_trading_signal(self, signal) -> bool:
        """
        Trading sinyalini formatla ve gönder.
        
        Args:
            signal: TradingSignal from PredictorEngine
        """
        if not signal.is_valid:
            logger.debug(f"Signal not valid for {signal.symbol}: {signal.warnings}")
            return False
        
        # Emoji seçimi
        direction_emoji = "🟢" if signal.signal_type.value == "LONG" else "🔴"
        risk_emoji = {
            "LOW": "🟢",
            "MEDIUM": "🟡",
            "HIGH": "🔴"
        }.get(signal.risk_level.value, "⚪")
        
        # Reasons formatla
        reasons_text = "\n".join([f"✅ {r}" for r in signal.reasons[:6]])
        
        # Warnings
        warnings_text = ""
        if signal.warnings:
            warnings_text = "\n━━━ *UYARILAR* ━━━\n" + "\n".join([f"⚠️ {w}" for w in signal.warnings[:3]])
        
        # Ana mesaj
        message = f"""🎯 *{signal.signal_type.value} SİNYALİ - {signal.symbol}*
━━━━━━━━━━━━━━━━━━━━━━

{direction_emoji} *YÖN: {signal.signal_type.value}*
📍 *ENTRY:* ${signal.entry_low:,.0f} - ${signal.entry_high:,.0f}

🎯 *TP1:* ${signal.tp1:,.0f}
🎯 *TP2:* ${signal.tp2:,.0f}
🎯 *TP3:* ${signal.tp3:,.0f}
🛑 *SL:* ${signal.sl:,.0f}

⚖️ *R/R:* 1:{signal.risk_reward:.1f}
🧠 *Güven:* %{signal.confidence:.0f}
{risk_emoji} *Risk:* {signal.risk_level.value}
💰 *Potansiyel:* ${signal.potential_usd:,.0f}

━━━ *NEDEN BU SİNYAL?* ━━━
{reasons_text}
{warnings_text}

━━━ *VERİ KALİTESİ* ━━━
📡 Kaynak: {signal.data_sources_ok}/{signal.data_sources_total} OK

━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
📡 *DEMIR AI v10 - LIVE DATA*"""
        
        success = self._send_message(message)
        
        if success:
            logger.info(f"✅ Signal sent: {signal.symbol} {signal.signal_type.value} %{signal.confidence:.0f}")
        
        return success
    
    def send_market_summary(self, snapshots: dict) -> bool:
        """
        Tüm coinlerin özet durumunu gönder.
        """
        lines = ["📊 *PİYASA ÖZETİ*\n━━━━━━━━━━━━━━━━━━"]
        
        for symbol, snapshot in snapshots.items():
            if not snapshot.is_valid:
                lines.append(f"❌ {symbol}: VERİ HATASI")
                continue
            
            # Trend emoji
            trend_emoji = {
                "BULLISH": "📈",
                "BEARISH": "📉",
                "NEUTRAL": "➡️"
            }.get(snapshot.trend, "❓")
            
            # Price change emoji
            change = snapshot.price_change_24h
            change_emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            
            lines.append(
                f"\n{trend_emoji} *{symbol}*\n"
                f"💰 ${snapshot.price:,.0f} ({change_emoji}{change:+.1f}%)\n"
                f"📊 RSI: {snapshot.rsi_1h:.0f} | OB: {snapshot.bid_ask_ratio:.2f}x\n"
                f"💰 FR: {snapshot.funding_rate:.3f}% | 🐋: {snapshot.whale_net_flow:+.0f}"
            )
        
        lines.append(f"\n━━━━━━━━━━━━━━━━━━\n⏰ {datetime.now().strftime('%H:%M:%S')}")
        
        return self._send_message("\n".join(lines))
    
    def send_error_alert(self, error_message: str) -> bool:
        """
        Hata bildirimi gönder.
        HATA YUTULMAZ - kullanıcıya bildirilir.
        """
        message = f"""⚠️ *DEMIR AI - HATA*
━━━━━━━━━━━━━━━━━━

❌ {error_message}

━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
📡 *Lütfen kontrol edin!*"""
        
        return self._send_message(message)
    
    def send_startup_message(self) -> bool:
        """
        Bot başlatıldı mesajı.
        """
        message = """🚀 *DEMIR AI v10 - BAŞLATILDI*
━━━━━━━━━━━━━━━━━━━━━━

✅ Data Hub: AKTIF
✅ Predictor: AKTIF
✅ Notifier: AKTIF

📊 *Takip:* BTC, ETH, SOL, LTC
🎯 *Mod:* Prediktif Sinyal

━━━━━━━━━━━━━━━━━━━━━━
📡 *SADECE GERÇEK VERİ - MOCK YOK*"""
        
        return self._send_message(message)
    
    def send_data_quality_alert(self, symbol: str, errors: list) -> bool:
        """
        Veri kalitesi uyarısı gönder.
        """
        error_list = "\n".join([f"❌ {e}" for e in errors[:5]])
        
        message = f"""⚠️ *VERİ HATASI - {symbol}*
━━━━━━━━━━━━━━━━━━

{error_list}

━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}
📡 *Veri kaynakları kontrol ediliyor...*"""
        
        return self._send_message(message)


# Singleton
_notifier: Optional[SmartNotifier] = None

def get_notifier() -> SmartNotifier:
    global _notifier
    if _notifier is None:
        _notifier = SmartNotifier()
    return _notifier
