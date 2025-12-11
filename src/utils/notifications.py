import logging
import requests
import asyncio
import json
import os
from datetime import datetime, timedelta
from src.config.settings import Config
from src.core.signal_filter import SignalFilter

logger = logging.getLogger("NOTIFICATION_MANAGER")


class DedupCache:
    """
    Prevents spamming the same signal repeatedly.
    Now Price-Aware (Phase 21 Fix).
    """
    def __init__(self, cooldown_minutes: int = 60, price_threshold: float = 0.01):
        self.cache = {}
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.price_threshold = price_threshold # 1% default

    def is_duplicate(self, symbol: str, side: str, current_price: float) -> bool:
        key = f"{symbol}_{side}"
        now = datetime.now()
        
        if key in self.cache:
            last_time, last_price = self.cache[key]
            
            # 1. Time Check
            if now - last_time < self.cooldown:
                # 2. Price Check (If moved > 1%, it's NEW)
                price_diff = abs(current_price - last_price) / last_price
                if price_diff < self.price_threshold:
                    return True # Duplicate (Close in time AND price)
        
        # Update cache
        self.cache[key] = (now, current_price)
        return False

class NotificationManager:
    """
    DEMIR AI V20.0 - TELEGRAM ALERT SYSTEM
    
    Sends critical signals through Telegram.
    """
    
    def __init__(self):
        self.telegram_token = Config.TELEGRAM_TOKEN
        self.telegram_chat_id = Config.TELEGRAM_CHAT_ID
        self.telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage" if self.telegram_token else None
        self.signal_filter = SignalFilter(quality_threshold=70)
        self.rejected_log_path = "rejected_signals.json"
        self.dedup_cache = DedupCache(cooldown_minutes=60) # Phase 21: Dedup

    async def send_signal(self, signal: dict, snapshot: dict = None):
        """
        Sends signal to Telegram (Strictly for STRONG signals).
        """
        if not self.telegram_token or not self.telegram_chat_id:
            return
            
        # 1. STRICT FILTER: Only STRONG signals
        if signal.get('quality') != 'STRONG':
            # Log silent rejection
            return
        
        # 2. DEDUP CHECK
        price = signal.get('entry_price', signal.get('price', 0))
        if self.dedup_cache.is_duplicate(signal['symbol'], signal['side'], price):
            return
        
        try:
            side_icon = "🟢 LONG 🚀" if signal['side'] == "BUY" else "🔴 SHORT 🔻"
            conf = signal['confidence']
            
            # Formatted for optimal readability on mobile
            message = (
                f"{side_icon} **{signal['symbol']}**\n"
                f"━━━━━━━━━━━━━━\n"
                f"🧠 **Decision:** {signal.get('quality', 'STRONG')} ({conf:.1f}%)\n"
                f"📉 **Reason:** {signal.get('reason', 'AI Model Decision')}\n"
                f"━━━━━━━━━━━━━━\n"
                f"🚪 **ENTRY:** ${signal['entry_price']:.4f}\n"
                f"🎯 **TP:** ${signal['tp_price']:.4f}\n"
                f"🛡️ **SL:** ${signal['sl_price']:.4f}\n"
                f"💰 **Size:** {signal.get('kelly_size', 'N/A')}%\n"
                f"━━━━━━━━━━━━━━\n"
                f"⚠️ _Trade at your own risk. AI Beta v9.0_"
            )
            
            await self.send_message_raw(message)
            logger.info(f"✅ STRONG Signal sent: {signal['symbol']}")
            
        except Exception as e:
            logger.error(f"Telegram Error: {e}")

    async def send_heartbeat(self, price_info: dict):
        """Sends hourly heartbeat if no signals were sent."""
        if not self.telegram_token: return
        
        try:
            msg = (
                f"💓 **System Heartbeat**\n"
                f"━━━━━━━━━━━━━━\n"
                f"🤖 AI Engine: **ONLINE**\n"
                f"📅 Time: {datetime.now().strftime('%H:%M')}\n"
                f"━━━━━━━━━━━━━━\n"
            )
            
            for sym, price in price_info.items():
                msg += f"🔹 **{sym}:** ${price:,.2f}\n"
                
            msg += f"━━━━━━━━━━━━━━\n"
            msg += f"Scanning for opportunities..."
            
            await self.send_message_raw(msg)
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")

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
    
    def _log_rejected_signal(self, signal: dict, quality_score: int, reason: str):
        """Log rejected signals to JSON file for dashboard review"""
        try:
            rejected_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": signal['symbol'],
                "side": signal['side'],
                "quality_score": quality_score,
                "reason": reason,
                "pattern": signal.get('pattern', 'None'),
                "confidence": signal.get('confidence', 0)
            }
            
            # Append to log file
            if os.path.exists(self.rejected_log_path):
                with open(self.rejected_log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(rejected_data)
            
            # Keep only last 50 rejected signals
            if len(logs) > 50:
                logs = logs[-50:]
            
            with open(self.rejected_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"📝 Rejected signal logged: {signal['symbol']} (Score: {quality_score})")
        except Exception as e:
            logger.error(f"Error logging rejected signal: {e}")