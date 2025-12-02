import asyncio
import os
from src.core.engine import BotEngine
from src.utils.logger import setup_logger
from src.config.settings import Config

# --- SİSTEM BAŞLANGIÇ NOKTASI ---

async def main():
    # 1. Loglama Sistemini Kur
    setup_logger()
    
    # 2. Ortam Değişkenlerini Kontrol Et
    print(f">> Starting DEMIR AI v{Config.VERSION}")
    print(f">> Mode: {Config.ENVIRONMENT}")
    
    # 3. Bot Motorunu Oluştur
    bot = BotEngine()
    
    try:
        # 4. Sonsuz Döngüyü Başlat
        await bot.start()
    except KeyboardInterrupt:
        print("\n>> Manual Stop Signal Received.")
        await bot.stop()
    except Exception as e:
        print(f">> FATAL ERROR: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass