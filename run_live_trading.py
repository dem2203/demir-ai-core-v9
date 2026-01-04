# -*- coding: utf-8 -*-
"""
DEMIR AI v11.1 - UNIFIED LIVE SYSTEM
====================================
Hem sinyal monitörü hem de Telegram komut handler'ı çalıştırır.

2 Görev Paralel:
1. Sinyal Monitörü (dakikalık tarama)
2. Telegram Bot (komut dinleme: /analiz, /start, vs.)

Komut: python run_live_trading.py

Author: DEMIR AI Team
Date: 2026-01-04
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
import sys
import os

# Windows Unicode Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

from dotenv import load_dotenv
load_dotenv()

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("UNIFIED_RUNNER")


async def signal_monitor_loop():
    """Dakikalık sinyal tarama döngüsü."""
    from src.execution.signal_generator import SignalGenerator
    from src.execution.notifier import send_signal_alert, send_message
    
    logger.info("📡 Signal Monitor starting...")
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    generator = SignalGenerator(symbols, use_advanced=True)
    
    # Initialize async components
    await generator.initialize()
    
    logger.info(f"Monitor edilen semboller: {symbols}")
    
    while True:
        try:
            now = datetime.now()
            
            # Sonraki dakikanın 5. saniyesine kadar bekle
            next_minute = now.replace(second=5, microsecond=0)
            if next_minute <= now:
                next_minute += timedelta(minutes=1)
                
            wait_seconds = (next_minute - now).total_seconds()
            logger.info(f"⏳ Signal check in {wait_seconds:.0f}s...")
            await asyncio.sleep(wait_seconds)
            
            # Sinyal kontrolü
            logger.info(f"⏰ Tick: {datetime.now().strftime('%H:%M:%S')}")
            
            signals = await generator.check_for_signals()
            
            if signals:
                logger.info(f"🔥 {len(signals)} SIGNALS GENERATED!")
                for signal in signals:
                    send_signal_alert(signal)
            else:
                logger.debug("💤 No signals found.")
                
        except asyncio.CancelledError:
            logger.info("Signal monitor stopped.")
            break
        except Exception as e:
            logger.error(f"Signal monitor error: {e}")
            await asyncio.sleep(60)


async def telegram_bot_loop():
    """Telegram bot polling."""
    from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler
    from src.v10.telegram_bot import (
        start, info_cmd, analyze, durum, piyasa, brain_status,
        son_sinyaller, istatistik, risk_status, callback_handler,
        get_application
    )
    
    logger.info("🤖 Telegram Bot starting...")
    
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TOKEN:
        logger.error("❌ TELEGRAM_TOKEN not found!")
        return
    
    try:
        app = ApplicationBuilder().token(TOKEN).build()
        
        # Command Handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("info", info_cmd))
        app.add_handler(CommandHandler("analiz", analyze))
        app.add_handler(CommandHandler("durum", durum))
        app.add_handler(CommandHandler("piyasa", piyasa))
        app.add_handler(CommandHandler("brain", brain_status))
        app.add_handler(CommandHandler("son", son_sinyaller))
        app.add_handler(CommandHandler("istatistik", istatistik))
        app.add_handler(CommandHandler("risk", risk_status))
        app.add_handler(CallbackQueryHandler(callback_handler))
        
        logger.info("✅ Telegram Bot handlers ready")
        
        # Non-blocking polling
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        
        # Keep running until cancelled
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Telegram bot stopped.")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
    except Exception as e:
        logger.error(f"Telegram bot error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Ana çalıştırıcı - iki görevi paralel başlat."""
    from src.execution.notifier import send_message
    
    logger.info("="*50)
    logger.info("🚀 DEMIR AI v11.1 UNIFIED SYSTEM STARTED")
    logger.info("   📡 Signal Monitor + 🤖 Telegram Bot")
    logger.info("="*50)
    
    send_message(
        "🚀 **DEMIR AI v11.1 BAŞLATILDI**\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Mod: Live Monitor (%80 Eşik + Advanced)\n"
        "🐋 Whale Tracking: AKTIF\n"
        "💥 Likidation Hunter: AKTIF\n"
        "📊 Sentiment Analyzer: AKTIF\n"
        "🤖 Telegram Komutları: AKTIF\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📋 Komutlar:\n"
        "  /analiz BTCUSDT\n"
        "  /analiz ETHUSDT\n"
        "  /start - Ana menü"
    )
    
    # İki görevi paralel çalıştır
    try:
        await asyncio.gather(
            signal_monitor_loop(),
            telegram_bot_loop()
        )
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
