# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - LIVE TRADING RUNNER
==================================
Sistemi canlı modda çalıştırır.
Sonsuz döngü içinde piyasayı izler ve sinyal üretir.

Komut: python run_live_trading.py

Author: DEMIR AI Team
Date: 2026-01-04
"""
import asyncio
import logging
import time
from datetime import datetime
import sys

# Windows Unicode Fix
sys.stdout.reconfigure(encoding='utf-8')

from src.execution.signal_generator import SignalGenerator
from src.execution.notifier import send_signal_alert, send_message

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LIVE_RUNNER")

async def main():
    logger.info("="*50)
    logger.info("🚀 DEMIR AI v11.1 LIVE TRADING SYSTEM STARTED")
    logger.info("   🐋 Whale + 💥 Liquidation + 📊 Sentiment ACTIVE")
    logger.info("="*50)
    
    send_message("🚀 **DEMIR AI v11.1 BAŞLATILDI**\n"
                 "Mod: Live Monitor (%80 Eşik + Advanced)\n"
                 "🐋 Whale Tracking: AKTIF\n"
                 "💥 Likidation Hunter: AKTIF\n"
                 "📊 Sentiment Analyzer: AKTIF\n"
                 "Semboller: BTC, ETH")
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    generator = SignalGenerator(symbols, use_advanced=True)
    
    # Initialize async components (Whale WebSocket, etc.)
    await generator.initialize()
    
    logger.info(f"Monitor edilen semboller: {symbols}")
    
    while True:
        try:
            now = datetime.now()
            # Dakika başına senkronizasyon (Örn: Her dakikanın 5. saniyesi)
            # Bu sayede mum kapandıktan hemen sonra veri çekeriz.
            
            # Sonraki dakikanın 5. saniyesine kadar bekle
            next_minute = now.replace(second=5, microsecond=0)
            if next_minute <= now:
                # Eğer zaten geçtikse, bir sonraki dakikaya atla
                from datetime import timedelta
                next_minute += timedelta(minutes=1)
                
            wait_seconds = (next_minute - now).total_seconds()
            logger.info(f"⏳ Waiting {wait_seconds:.1f}s for candle close...")
            await asyncio.sleep(wait_seconds)
            
            # --- LOOP BAŞLANGICI ---
            logger.info(f"⏰ Tick: {datetime.now().strftime('%H:%M:%S')}")
            
            signals = await generator.check_for_signals()
            
            if signals:
                logger.info(f"🔥 {len(signals)} SIGNALS GENERATED!")
                for signal in signals:
                    send_signal_alert(signal)
            else:
                logger.info("💤 No signals found.")
                
            # --- LOOP SONU ---
            
        except KeyboardInterrupt:
            logger.info("🛑 Stopping live trading...")
            break
        except Exception as e:
            logger.error(f"❌ Critical Loop Error: {e}")
            send_message(f"⚠️ **SİSTEM HATASI**: {str(e)[:100]}")
            await asyncio.sleep(60) # Hata durumunda 1 dk bekle

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
