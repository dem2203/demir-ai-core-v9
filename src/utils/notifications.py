import logging
import requests
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from discord_webhook import DiscordWebhook, DiscordEmbed
from src.config.settings import Config

logger = logging.getLogger("NOTIFICATION_MANAGER")

class NotificationManager:
    """
    DEMIR AI V20.0 - MULTI-CHANNEL ALERT SYSTEM
    
    Sends critical signals through multiple channels:
    - Telegram (Primary)
    - Discord (Secondary)
    - Email (Backup)
    """
    
    def __init__(self):
        # Telegram
        self.telegram_token = Config.TELEGRAM_TOKEN
        self.telegram_chat_id = Config.TELEGRAM_CHAT_ID
        self.telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage" if self.telegram_token else None
        
        # Discord
        self.discord_webhook_url = Config.DISCORD_WEBHOOK_URL
        
        # Email (SMTP)
        self.email_enabled = Config.EMAIL_ENABLED
        self.email_from = Config.EMAIL_FROM
        self.email_to = Config.EMAIL_TO
        self.smtp_server = Config.SMTP_SERVER
        self.smtp_port = Config.SMTP_PORT
        self.smtp_user = Config.SMTP_USER
        self.smtp_password = Config.SMTP_PASSWORD

    async def send_signal(self, signal: dict):
        """
        Sends signal to ALL configured channels.
        """
        tasks = []
        
        # Telegram
        if self.telegram_token and self.telegram_chat_id:
            tasks.append(self._send_telegram(signal))
        
        # Discord
        if self.discord_webhook_url:
            tasks.append(self._send_discord(signal))
        
        # Email
        if self.email_enabled == "true":
            tasks.append(self._send_email(signal))
        
        # Send to all channels concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.warning("No notification channels configured!")

    async def _send_telegram(self, signal: dict):
        """Telegram signal format"""
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

    async def _send_discord(self, signal: dict):
        """Discord embed format"""
        try:
            webhook = DiscordWebhook(url=self.discord_webhook_url)
            
            # Color based on side
            color = 0x00ff00 if signal['side'] == "BUY" else 0xff0000
            
            embed = DiscordEmbed(
                title=f"{'🟢 LONG SIGNAL' if signal['side'] == 'BUY' else '🔴 SHORT SIGNAL'}: {signal['symbol']}",
                description=f"**Setup:** {signal.get('reason', 'AI Model Prediction')}",
                color=color
            )
            
            embed.add_embed_field(name="Entry Price", value=f"${signal['entry_price']:.4f}", inline=True)
            embed.add_embed_field(name="Take Profit", value=f"${signal['tp_price']:.4f}", inline=True)
            embed.add_embed_field(name="Stop Loss", value=f"${signal['sl_price']:.4f}", inline=True)
            embed.add_embed_field(name="AI Confidence", value=f"{signal['confidence']:.1f}%", inline=False)
            
            embed.set_footer(text="DEMIR AI v20.0 | Advisory Mode")
            embed.set_timestamp()
            
            webhook.add_embed(embed)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, webhook.execute)
            
            logger.info("✅ Signal sent to Discord")
        except Exception as e:
            logger.error(f"Discord Error: {e}")

    async def _send_email(self, signal: dict):
        """Email alert format"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🦅 DEMIR AI Signal: {signal['side']} {signal['symbol']}"
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            
            # HTML Email Body
            html_body = f"""
            <html>
              <body style="font-family: Arial, sans-serif; background-color: #0e1117; color: #e0e0e0; padding: 20px;">
                <div style="max-width: 600px; margin: auto; background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px;">
                  <h2 style="color: {'#00ff00' if signal['side'] == 'BUY' else '#ff0000'};">
                    {'🟢 LONG SIGNAL' if signal['side'] == 'BUY' else '🔴 SHORT SIGNAL'}: {signal['symbol']}
                  </h2>
                  <hr style="border-color: #30363d;">
                  <p><strong>Setup:</strong> {signal.get('reason', 'AI Model Prediction')}</p>
                  <p><strong>AI Confidence:</strong> {signal['confidence']:.1f}%</p>
                  <hr style="border-color: #30363d;">
                  <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                      <td style="padding: 10px; border: 1px solid #30363d;"><strong>Entry Price:</strong></td>
                      <td style="padding: 10px; border: 1px solid #30363d;">${signal['entry_price']:.4f}</td>
                    </tr>
                    <tr>
                      <td style="padding: 10px; border: 1px solid #30363d;"><strong>Take Profit:</strong></td>
                      <td style="padding: 10px; border: 1px solid #30363d;">${signal['tp_price']:.4f}</td>
                    </tr>
                    <tr>
                      <td style="padding: 10px; border: 1px solid #30363d;"><strong>Stop Loss:</strong></td>
                      <td style="padding: 10px; border: 1px solid #30363d;">${signal['sl_price']:.4f}</td>
                    </tr>
                  </table>
                  <hr style="border-color: #30363d;">
                  <p style="font-size: 12px; color: #8b949e;">⚠️ This is not financial advice. Make your own decision.</p>
                  <p style="font-size: 12px; color: #8b949e;">DEMIR AI v20.0 | Advisory Mode</p>
                </div>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send via SMTP
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._smtp_send, msg)
            
            logger.info("✅ Signal sent to Email")
        except Exception as e:
            logger.error(f"Email Error: {e}")

    def _smtp_send(self, msg):
        """Synchronous SMTP send"""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
        except Exception as e:
            raise e

    async def send_message_raw(self, text: str):
        """Send raw text to Telegram only"""
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
            logger.error(f"Telegram Raw Error: {e}")