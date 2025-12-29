# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - RAILWAY DEPLOYMENT ENTRYPOINT
============================================
Bu dosya Railway üzerinde 3 servisi tek process'te çalıştırır:
1. Web Dashboard (FastAPI) - Ana Process
2. Trading Engine - Background Task
3. Telegram Bot - Background Task
"""
import asyncio
import logging
from src.dashboard.server import app
from src.v10.engine import run_v10
from src.v10.telegram_bot import get_application

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAILWAY_MAIN")

@app.on_event("startup")
async def startup_event():
    """Server başladığında diğer servisleri de ateşle"""
    logger.info("🚀 RAILWAY STARTUP: Tüm sistemler başlatılıyor...")
    
    # 1. Start Engine (As Sonsuz Döngü)
    # create_task ile background'a atıyoruz, bloklamaz.
    asyncio.create_task(run_v10())
    logger.info("✅ Trading Engine Başlatıldı (Background)")
    
    # 2. Start Telegram Bot
    try:
        bot_app = get_application()
        if bot_app:
            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling()
            logger.info("✅ Telegram Bot Başlatıldı (Polling)")
            
            # Bot instance'ı globalde tutmak gerekebilir mi? 
            # Garbage collection'a uğramaması için app.state içine atabiliriz.
            app.state.bot_app = bot_app
    except Exception as e:
        logger.error(f"❌ Bot başlatma hatası: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Kapanış temizliği"""
    logger.info("🛑 SHUTDOWN: Sistemler kapatılıyor...")
    if hasattr(app.state, 'bot_app'):
        await app.state.bot_app.updater.stop()
        await app.state.bot_app.stop()
        await app.state.bot_app.shutdown()
