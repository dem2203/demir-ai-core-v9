# CRITICAL: Set thread limits BEFORE importing any libraries
# This prevents Railway resource exhaustion (pthread_create failures)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'

import asyncio
import logging
from src.config.settings import Config

# Configure logging - HATALARI GİZLEME!
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("MAIN")


async def main():
    """
    DEMIR AI v10 - MAIN ENTRY POINT
    
    Yeni prediktif trading sistemi:
    - Hareket ÖNCE sinyal verir
    - Entry/TP/SL hesaplar
    - HATA YUTMAZ
    - Mock/Fallback YOK
    - Telegram Bot entegrasyonu
    """
    print("=" * 60)
    print(f"🚀 DEMIR AI v10 - PREDICTIVE TRADING SYSTEM")
    print(f"📊 Mode: {Config.ENVIRONMENT}")
    print("=" * 60)
    
    # V10 Engine'i import et ve başlat
    from src.v10.engine import get_v10_engine
    
    engine = get_v10_engine()
    
    # Telegram Bot'u başlat
    telegram_app = None
    try:
        from src.v10.telegram_bot import get_application
        telegram_app = get_application()
        if telegram_app:
            await telegram_app.initialize()
            await telegram_app.start()
            await telegram_app.updater.start_polling()
            print("✅ Telegram Bot Başlatıldı (Polling)")
            logger.info("✅ Telegram Bot Başlatıldı (Polling)")
        else:
            print("⚠️ Telegram Bot başlatılamadı - TOKEN eksik olabilir")
            logger.warning("⚠️ Telegram Bot başlatılamadı - TOKEN eksik olabilir")
    except Exception as e:
        print(f"❌ Telegram Bot hatası: {e}")
        logger.error(f"❌ Telegram Bot hatası: {e}")
    
    try:
        print("📡 Starting V10 Engine...")
        print("🎯 Scanning: BTC, ETH, SOL, LTC")
        print("⏱️  Interval: 30 seconds")
        print("💰 Min potential: $500+")
        print("=" * 60)
        
        await engine.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
        await engine.stop()
        if telegram_app:
            await telegram_app.updater.stop()
            await telegram_app.stop()
            await telegram_app.shutdown()
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        # Telegram'a bildir
        from src.v10.smart_notifier import get_notifier
        notifier = get_notifier()
        notifier.send_error_alert(f"FATAL: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
