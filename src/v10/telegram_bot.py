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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler

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


# =============================================================================
# KEYBOARD BUTTONS
# =============================================================================
def get_main_keyboard():
    """Ana menü butonları"""
    keyboard = [
        [
            InlineKeyboardButton("📊 Piyasa", callback_data="piyasa"),
            InlineKeyboardButton("📈 Analiz BTC", callback_data="analiz_btc"),
            InlineKeyboardButton("📈 Analiz ETH", callback_data="analiz_eth"),
        ],
        [
            InlineKeyboardButton("📋 Durum", callback_data="durum"),
            InlineKeyboardButton("🧠 Brain", callback_data="brain"),
            InlineKeyboardButton("📉 Son Sinyaller", callback_data="son"),
        ],
        [
            InlineKeyboardButton("📊 İstatistik", callback_data="istatistik"),
            InlineKeyboardButton("⚠️ Risk", callback_data="risk"),
            InlineKeyboardButton("ℹ️ Yardım", callback_data="info"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


# =============================================================================
# COMMAND HANDLERS
# =============================================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start komutu - Ana menü"""
    await update.message.reply_text(
        "🤖 DEMIR AI v10 Online\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Aşağıdaki butonlardan birini seçin:",
        reply_markup=get_main_keyboard()
    )


async def info_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/info komutu - PRO VERSION"""
    await update.message.reply_text(
        "🤖 DEMIR AI PRO - KOMUTLAR\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "📋 GENEL:\n"
        "  /start → Ana menü (butonlu)\n"
        "  /info → Bu mesajı göster\n"
        "  /durum → Bot + Risk Engine durumu\n"
        "  /brain → 🧠 Thinking Brain durumu\n\n"
        "📊 ANALİZ:\n"
        "  /analiz BTCUSDT → 🧠 AI PRO Analiz\n"
        "  /piyasa → BTC/ETH anlık durum\n"
        "  /son → Son 5 sinyal özeti\n\n"
        "📈 PERFORMANS:\n"
        "  /istatistik → Win rate, rolling metrics\n"
        "  /performans → Detaylı rapor\n"
        "  /risk → Risk Engine durumu\n\n"
        "🛡️ PRO ÖZELLİKLER:\n"
        "  • 4 AI Council (Weighted Voting)\n"
        "  • Kelly Position Sizing\n"
        "  • Auto-Shutdown (%40 limit)\n"
        "  • LSTM Prediction (%83 accuracy)\n\n"
        "━━━━━━━━ DEMIR AI v10 PRO ━━━━━━━━",
        reply_markup=get_main_keyboard()
    )


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/analiz command handler"""
    if not context.args:
        await update.message.reply_text("⚠️ Lütfen bir sembol girin. Örnek: /analiz ETHUSDT")
        return

    symbol = context.args[0].upper()
    if "USDT" not in symbol:
        symbol += "USDT"

    await run_analysis(update, symbol)


async def run_analysis(update: Update, symbol: str):
    """Analiz çalıştır - AI Council V2 ile"""
    # Determine where to send the reply
    if update.callback_query:
        message = update.callback_query.message
        status_msg = await message.reply_text(f"🧠 {symbol} AI COUNCIL analiz ediliyor...\n⏳ 4 AI paralel çalışıyor, lütfen bekleyin...")
    else:
        message = update.message
        status_msg = await message.reply_text(f"🧠 {symbol} AI COUNCIL analiz ediliyor...\n⏳ 4 AI paralel çalışıyor, lütfen bekleyin...")

    try:
        from src.brain.ai_council_v2 import get_ai_council
        
        # AI Council'ı al ve analiz yap
        council = get_ai_council()
        decision = await council.analyze(symbol)
        
        # Rapor formatla ve gönder
        report = council.format_report(decision)
        await message.reply_text(report, reply_markup=get_main_keyboard(), parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"AI Council analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback - eski sisteme dön
        try:
            await status_msg.edit_text("⚠️ AI Council hata verdi, klasik analiz deneniyor...")
            
            from src.v10.early_signal_engine import EarlySignalEngine
            engine = EarlySignalEngine()
            signal = await engine.analyze(symbol)
            
            if signal:
                report_text = (
                    f"🧠 AI ANALİZ RAPORU - {signal.symbol}\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"📍 Karar: {signal.action}\n"
                    f"🎯 Güven: %{signal.confidence:.0f}\n"
                    f"📝 {signal.reasoning[:300] if signal.reasoning else 'N/A'}"
                )
                await message.reply_text(report_text, reply_markup=get_main_keyboard())
            else:
                await message.reply_text(f"❌ {symbol} için analiz yapılamadı.")
            
            await engine.close()
        except Exception as e2:
            await message.reply_text(f"❌ Analiz hatası: {str(e)[:100]}\nFallback hatası: {str(e2)[:100]}")
    finally:
        try:
            await status_msg.delete()
        except:
            pass


# =============================================================================
# BUTTON CALLBACK HANDLER
# =============================================================================
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Buton tıklamalarını işle"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "piyasa":
        from src.v10.telegram_commands import handle_command
        response = await handle_command("/piyasa")
        await query.message.reply_text(response, reply_markup=get_main_keyboard())
        
    elif data == "analiz_btc":
        await run_analysis(update, "BTCUSDT")
        
    elif data == "analiz_eth":
        await run_analysis(update, "ETHUSDT")
        
    elif data == "durum":
        from src.v10.telegram_commands import handle_command
        response = await handle_command("/durum")
        await query.message.reply_text(response, reply_markup=get_main_keyboard())
        
    elif data == "brain":
        from src.v10.telegram_commands import handle_command
        response = await handle_command("/brain")
        await query.message.reply_text(response, reply_markup=get_main_keyboard())
        
    elif data == "son":
        from src.v10.telegram_commands import handle_command
        response = await handle_command("/son")
        await query.message.reply_text(response, reply_markup=get_main_keyboard())
        
    elif data == "istatistik":
        from src.v10.telegram_commands import handle_command
        response = await handle_command("/istatistik")
        await query.message.reply_text(response, reply_markup=get_main_keyboard())
        
    elif data == "risk":
        from src.v10.telegram_commands import handle_command
        response = await handle_command("/risk")
        await query.message.reply_text(response, reply_markup=get_main_keyboard())
        
    elif data == "info":
        await query.message.reply_text(
            "🤖 DEMIR AI - KOMUTLAR\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Yukarıdaki butonları kullanabilir veya\n"
            "manuel komutlar yazabilirsiniz:\n\n"
            "/analiz BTCUSDT\n"
            "/analiz ETHUSDT\n"
            "/piyasa\n"
            "/durum\n",
            reply_markup=get_main_keyboard()
        )


# =============================================================================
# GENERIC COMMAND HANDLER
# =============================================================================
async def cmd_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generic command handler - routes to telegram_commands"""
    try:
        from src.v10.telegram_commands import handle_command
        command = update.message.text
        response = await handle_command(command)
        # Remove markdown formatting
        response = response.replace("*", "").replace("_", "")
        await update.message.reply_text(response, reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Command error: {e}")
        await update.message.reply_text(f"❌ Komut hatası: {str(e)[:100]}")


# =============================================================================
# APPLICATION BUILDER
# =============================================================================
def get_application():
    """Bot uygulamasını oluştur ve döndür - TÜM KOMUTLAR + BUTONLAR"""
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN env variable not found!")
        return None
        
    application = ApplicationBuilder().token(TOKEN).build()
    
    # === KOMUTLAR ===
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('info', info_cmd))
    application.add_handler(CommandHandler('help', info_cmd))
    application.add_handler(CommandHandler('analiz', analyze))
    
    # telegram_commands.py'den gelen komutlar
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
    
    # === BUTON CALLBACK ===
    application.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("✅ Telegram Bot: 16 komut + Inline Butonlar yüklendi")
    return application


if __name__ == '__main__':
    app = get_application()
    if app:
        logger.info("🤖 Telegram Bot Started Polling...")
        app.run_polling()
    else:
        logger.error("Bot başlatılamadı.")

