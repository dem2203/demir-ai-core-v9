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
    from src.brain.signal_performance_tracker import get_tracker
    from src.execution.paper_trader import get_paper_trader
    
    logger.info("📡 Signal Monitor starting...")
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    generator = SignalGenerator(symbols, use_advanced=True)
    tracker = get_tracker()
    paper_trader = get_paper_trader()
    
    # Initialize async components
    await generator.initialize()
    
    last_report_date = None
    
    logger.info(f"Monitor edilen semboller: {symbols}")
    
    while True:
        try:
            now = datetime.now()
            
            # --- DAILY REPORT CHECK (23:00) ---
            if now.hour == 23 and now.minute == 0:
                current_date = now.date()
                if last_report_date != current_date:
                    try:
                        # Raporu hazırla ve gönder
                        from src.v10.telegram_commands import get_telegram_commands
                        cmd = get_telegram_commands()
                        report = await cmd.cmd_istatistik()
                        
                        full_report = f"📅 *GÜNLÜK KAPANIŞ RAPORU*\n{report}"
                        await send_message(full_report)
                        
                        last_report_date = current_date
                        logger.info("✅ Daily report sent")
                    except Exception as e:
                        logger.error(f"Daily report error: {e}")
            
            # Sonraki dakikanın 5. saniyesine kadar bekle
            next_minute = now.replace(second=5, microsecond=0)
            if next_minute <= now:
                next_minute += timedelta(minutes=1)
                
            wait_seconds = (next_minute - now).total_seconds()
            logger.info(f"⏳ Signal check in {wait_seconds:.0f}s...")
            await asyncio.sleep(wait_seconds)
            
            # --- START LOOP ---
            logger.info(f"⏰ Tick: {datetime.now().strftime('%H:%M:%S')}")
            
            # 1. Tracker Update (TP/SL kontrolü)
            try:
                # Arka planda çalışsın, bloklamasın
                updated_signals = await asyncio.to_thread(tracker.check_signals)
                if updated_signals:
                    logger.info(f"📊 {len(updated_signals)} signals updated in tracker")
            except Exception as e:
                logger.error(f"Tracker check error: {e}")
            
            # 2. Paper Trader Update (Trailing Stop / DCA)
            current_prices = {}
            for sym in symbols:
                price = await generator._get_realtime_price(sym)
                if price:
                    current_prices[sym] = price
            
            if current_prices:
                await paper_trader.update_positions(current_prices)
                
            # 3. Sinyal Kontrolü
            signals = await generator.check_for_signals()
            
            if signals:
                logger.info(f"🔥 {len(signals)} SIGNALS GENERATED!")
                for signal in signals:
                    # Telegram Bildirimi
                    send_signal_alert(signal)
                    
                    # Paper Trade İşlemi
                    await paper_trader.execute_trade(signal)
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
    logger.info("🤖 Telegram Bot starting...")
    
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TOKEN:
        logger.error("❌ TELEGRAM_TOKEN not found!")
        return
    
    try:
        # Use the existing get_application function that sets up all handlers
        from src.v10.telegram_bot import get_application
        
        app = get_application()
        if not app:
            logger.error("❌ Failed to create Telegram application!")
            return
        
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
