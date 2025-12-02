import asyncio
import os
from src.core.engine import BotEngine
from src.utils.logger import setup_logger
from src.config.settings import Config
from src.brain.trainer import AITrainer  # <--- YENİ EKLENEN EĞİTMEN

# --- SİSTEM BAŞLANGIÇ NOKTASI ---

async def main():
    # 1. Loglama Sistemini Kur
    setup_logger()
    
    # 2. Ortam Değişkenlerini Kontrol Et
    print(f">> Starting DEMIR AI v{Config.VERSION}")
    print(f">> Mode: {Config.ENVIRONMENT}")
    
    # 3. YAPAY ZEKA EĞİTİM KONTROLÜ (YENİ!)
    # Botun beyni (Model dosyası) var mı kontrol et. Yoksa oluştur.
    trainer = AITrainer()
    
    if not os.path.exists(AITrainer.MODEL_PATH):
        print(">> 🧠 AI Brain not found (First Run). Starting initial training sequence...")
        print(">> ⏳ Downloading historical data and training Random Forest model...")
        try:
            await trainer.train_new_model()
            print(">> ✅ Training Complete. Brain saved.")
        except Exception as e:
            print(f">> ❌ Training Failed: {e}")
            # Eğitim başarısız olsa bile botu başlatmayı deneyebiliriz (Fallback modunda çalışır)
    else:
        print(f">> 🧠 AI Brain found at {AITrainer.MODEL_PATH}. Skipping training.")

    # 4. Bot Motorunu Oluştur ve Başlat
    bot = BotEngine()
    
    try:
        # 5. Sonsuz Döngüyü Başlat
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