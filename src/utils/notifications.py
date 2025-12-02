import logging
import requests
import asyncio
from src.config.settings import Config

logger = logging.getLogger("NOTIFICATION_MANAGER")

class NotificationManager:
    """
    TELEGRAM ALERT SYSTEM
    Botun dış dünya ile (seninle) iletişim kurduğu yer.
    7/24 çalışırken sana rapor verir.
    """
    
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    async def send_message(self, message: str, level: str = "INFO"):
        """
        Asenkron mesaj gönderimi (Botu yavaşlatmaz).
        Emojilerle mesajın önemini belirtir.
        """
        if not self.token or not self.chat_id:
            logger.warning("Telegram token missing. Notifications disabled.")
            return

        # Mesaj seviyesine göre ikon seç
        icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "CRITICAL": "🚨",
            "PROFIT": "💰",
            "LOSS": "🔻"
        }
        icon = icons.get(level, "📢")
        
        formatted_message = f"{icon} **DEMIR AI v{Config.VERSION}**\n\n{message}"
        
        try:
            # Blocking I/O olmaması için ayrı thread'de çalıştırılır (Requests kütüphanesi senkrondur)
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_message,
                "parse_mode": "Markdown"
            }
            # Basit HTTP isteği
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: requests.post(self.base_url, data=payload))
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def send_trade_alert(self, symbol: str, side: str, price: float, size: float):
        """Özel Alım-Satım Bildirimi"""
        msg = (
            f"*NEW TRADE EXECUTED*\n"
            f"Symbol: `{symbol}`\n"
            f"Side: *{side}*\n"
            f"Price: ${price}\n"
            f"Size: ${size:.2f}\n"
            f"Time: {Config.ENVIRONMENT.upper()}"
        )
        await self.send_message(msg, level="SUCCESS" if side=="BUY" else "WARNING")