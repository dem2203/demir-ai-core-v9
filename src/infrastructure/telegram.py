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
        """Send a text message with retry logic for Railway timeout issues"""
        if not self.bot or not self.chat_id:
            return
        
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Increase timeout for Railway's network
                await asyncio.wait_for(
                    self.bot.send_message(
                        chat_id=self.chat_id, 
                        text=message, 
                        parse_mode="Markdown"
                    ),
                    timeout=30.0  # 30 second timeout
                )
                logger.info(f"✅ Telegram message sent successfully (attempt {attempt + 1})")
                return  # Success!
                
            except asyncio.TimeoutError:
                wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Telegram timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Telegram failed after {max_retries} attempts (timeout)")
                    
            except Exception as e:
                wait_time = base_delay * (2 ** attempt)
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Telegram error: {e} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Telegram failed after {max_retries} attempts: {e}")

    async def send_alert(self, title: str, body: str, color: str = "⚪"):
        """Send a formatted alert"""
        msg = f"{color} *{title}*\n━━━━━━━━━━━━━━━━\n{body}"
        await self.send_message(msg)
