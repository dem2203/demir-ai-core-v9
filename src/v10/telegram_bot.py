# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - Interactive Telegram Bot
=======================================
Kullanıcı komutlarını dinler ve anlık analiz raporları sunar.

Komutlar:
/start  - Botu başlat ve durum kontrolü yap
/analiz <SYMBOL> - Belirtilen coin için anlık AI analizi yap (örn: /analiz BTCUSDT)
/durum - Sistem durumunu göster
"""
import logging
import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

# Load env variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start komutu"""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(
            "🤖 *DEMIR AI v10 Online*\n\n"
            "Komutlarım:\n"
            "🔍 `/analiz BTCUSDT` - Anlık Analiz\n"
            "📊 `/durum` - Sistem Durumu"
        ),
        parse_mode=ParseMode.MARKDOWN
    )

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/analiz command handler"""
    if not context.args:
        await update.message.reply_text("⚠️ Lütfen bir sembol girin. Örnek: `/analiz ETHUSDT`")
        return

    symbol = context.args[0].upper()
    if "USDT" not in symbol:
        symbol += "USDT"

    status_msg = await update.message.reply_text(f"🔍 *{symbol}* analiz ediliyor... Lütfen bekleyin.", parse_mode=ParseMode.MARKDOWN)

    try:
        # Import Engine here to avoid global lock/init issues
        from src.v10.early_signal_engine import EarlySignalEngine
        
        # Initialize Engine (in Quick Mode if possible, but full analysis needed)
        # Note: Engine init calls APIs, might bridge to existing modules
        engine = EarlySignalEngine()
        
        # Run Analysis
        signal = await engine.analyze(symbol)
        
        # Prepare Report
        if signal:
            # Emoji selection
            if signal.action == "BUY":
                emoji = "🟢"
            elif signal.action == "SELL":
                emoji = "🔴"
            else:
                emoji = "⏸️"
                
            risk_info = ""
            if signal.risk_profile:
                rp = signal.risk_profile
                risk_info = (
                    f"\n💰 *KASA YÖNETİMİ:*\n"
                    f"• Kaldıraç: {rp.get('leverage')}x\n"
                    f"• Marjin: %{rp.get('position_size_pct')}\n"
                )

            report = (
                f"🧠 *AI ANALİZ RAPORU - {signal.symbol}*\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📍 Karar: {emoji} *{signal.action}*\n"
                f"🎯 Güven: *%{signal.confidence:.0f}*\n"
                f"💰 Fiyat: ${await engine._get_current_price(symbol):,.2f}\n"
                f"{risk_info}\n"
                f"📝 *AI Mantığı:*\n{signal.reasoning}\n\n"
                f"🤖 *Claude:* {signal.llm_reasoning}\n\n"
                f"⏰ {signal.timestamp.strftime('%H:%M:%S')}"
            )
            
            await update.message.reply_text(report, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"❌ {symbol} için veri alınamadı veya sinyal üretilemedi.")
            
        await engine.close()
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        await update.message.reply_text(f"❌ Analiz hatası oluştu: {str(e)[:100]}")
    finally:
        # Delete "analyzing..." message
        try:
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=status_msg.message_id)
        except:
            pass

async def cmd_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generic command handler - routes to telegram_commands"""
    try:
        from src.v10.telegram_commands import handle_command
        command = update.message.text
        response = await handle_command(command)
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Command error: {e}")
        await update.message.reply_text(f"❌ Komut hatası: {str(e)[:100]}")

def get_application():
    """Bot uygulamasını oluştur ve döndür - TÜM KOMUTLAR"""
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN env variable not found!")
        return None
        
    application = ApplicationBuilder().token(TOKEN).build()
    
    # === TÜM KOMUTLAR ===
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('analiz', analyze))
    
    # telegram_commands.py'den gelen komutlar
    application.add_handler(CommandHandler('info', cmd_handler))
    application.add_handler(CommandHandler('help', cmd_handler))
    application.add_handler(CommandHandler('durum', cmd_handler))
    application.add_handler(CommandHandler('status', cmd_handler))
    application.add_handler(CommandHandler('piyasa', cmd_handler))
    application.add_handler(CommandHandler('market', cmd_handler))
    application.add_handler(CommandHandler('istatistik', cmd_handler))
    application.add_handler(CommandHandler('stats', cmd_handler))
    application.add_handler(CommandHandler('son', cmd_handler))
    application.add_handler(CommandHandler('recent', cmd_handler))
    application.add_handler(CommandHandler('risk', cmd_handler))
    application.add_handler(CommandHandler('positions', cmd_handler))
    application.add_handler(CommandHandler('brain', cmd_handler))
    application.add_handler(CommandHandler('thinking', cmd_handler))
    
    logger.info("✅ Telegram Bot: 16 komut yüklendi")
    return application

if __name__ == '__main__':
    app = get_application()
    if app:
        logger.info("🤖 Telegram Bot Started Polling...")
        app.run_polling()
    else:
        logger.error("Bot başlatılamadı.")

