import logging
import requests
import asyncio
from src.config.settings import Config

logger = logging.getLogger("NOTIFICATION_MANAGER")

class NotificationManager:
    """
    VIP SIGNAL BROADCASTER
    AI'ın analizlerini profesyonel bir sinyal formatında Telegram'a iletir.
    """
    
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    async def send_signal(self, signal: dict):
        """
        Detaylı Sinyal Kartı Gönderir.
        """
        if not self.token or not self.chat_id:
            return

        # Yön İkonu
        side_icon = "🟢 LONG 🚀" if signal['side'] == "BUY" else "🔴 SHORT 🔻"
        
        # Güven Skoru Görseli
        conf = signal['confidence']
        conf_icon = "⭐⭐⭐" if conf > 85 else ("⭐⭐" if conf > 70 else "⭐")

        # Mesaj Şablonu
        message = (
            f"{side_icon} **{signal['symbol']}**\n"
            f"━━━━━━━━━━━━━━\n"
            f"🤖 **AI Confidence:** {conf:.1f}% {conf_icon}\n"
            f"📊 **Setup:** {signal.get('reason', 'AI Prediction Model')}\n"
            f"━━━━━━━━━━━━━━\n"
            f"📍 **ENTRY:** ${signal['entry_price']:.4f}\n"
            f"🎯 **TP (Target):** ${signal['tp_price']:.4f}\n"
            f"🛡️ **SL (Stop):** ${signal['sl_price']:.4f}\n"
            f"━━━━━━━━━━━━━━\n"
            f"⚠️ _Bu bir yatırım tavsiyesi değildir. Son karar sizindir._"
        )
        
        await self.send_message_raw(message)

    async def send_message_raw(self, text: str):
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: requests.post(self.base_url, data=payload))
        except Exception as e:
            logger.error(f"Telegram Error: {e}")