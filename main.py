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
    """
    print("=" * 60)
    print(f"🚀 DEMIR AI v10 - PREDICTIVE TRADING SYSTEM")
    print(f"📊 Mode: {Config.ENVIRONMENT}")
    print("=" * 60)
    
    # V10 Engine'i import et ve başlat
    from src.v10.engine import get_v10_engine
    
    engine = get_v10_engine()
    
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
