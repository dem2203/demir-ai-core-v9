import logging
import requests
import asyncio
from src.config.settings import Config

logger = logging.getLogger("NOTIFICATION_MANAGER")

class NotificationManager:
    """
    DEMIR AI V20.0 - TELEGRAM ALERT SYSTEM
    
    Sends critical signals through Telegram.
    """
    
    def __init__(self):
        self.telegram_token = Config.TELEGRAM_TOKEN
        self.telegram_chat_id = Config.TELEGRAM_CHAT_ID
        self.telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage" if self.telegram_token else None

    async def send_signal(self, signal: dict):
        """
        Sends signal to Telegram.
        """
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured!")
            return
        
        try:
            side_icon = "🟢 LONG 🚀" if signal['side'] == "BUY" else "🔴 SHORT 🔻"
            conf = signal['confidence']
            conf_icon = "⭐⭐⭐" if conf > 85 else ("⭐⭐" if conf > 70 else "⭐")

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
            logger.info("✅ Signal sent to Telegram")
        except Exception as e:
            logger.error(f"Telegram Error: {e}")

    async def send_message_raw(self, text: str):
        """Send raw text to Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        try:
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: requests.post(self.telegram_url, data=payload))
        except Exception as e:
            logger.error(f"Telegram Error: {e}")