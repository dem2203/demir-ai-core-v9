import logging
import asyncio
from telegram import Bot
from src.config import Config

logger = logging.getLogger("TELEGRAM")

class TelegramBot:
    """
    Infrastructure Layer: Notification System
    """
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.bot = None
        
        if self.token:
            self.bot = Bot(token=self.token)
        else:
            logger.warning("⚠️ Telegram Token missing. Notifications disabled.")

    async def send_message(self, message: str):
        """Send a text message to the admin"""
        if not self.bot or not self.chat_id:
            return
            
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def send_alert(self, title: str, body: str, color: str = "⚪"):
        """Send a formatted alert"""
        msg = f"{color} *{title}*\n━━━━━━━━━━━━━━━━\n{body}"
        await self.send_message(msg)
