import asyncio
import os
import glob # Dosya silmek için
from src.core.engine import BotEngine
from src.utils.logger import setup_logger
from src.config.settings import Config
from src.brain.trainer import AITrainer as LSTMTrainer
from src.brain.rl_trainer import RLTrainer

async def main():
    setup_logger()
    print(f">> Starting DEMIR AI v{Config.VERSION}")
    print(f">> Mode: {Config.ENVIRONMENT}")
    
    # --- TEMİZLİK MODU (VERSİYON GEÇİŞİ İÇİN) ---
    # Eğer yeni bir özellik eklediysek, eski modelleri silip sıfırdan eğitmek en sağlıklısıdır.
    # Bunu sadece gerektiğinde aktif et veya manuel sil.
    # Otomatik silme için:
    # if os.path.exists("src/brain/models/storage"):
    #     files = glob.glob("src/brain/models/storage/*.h5")
    #     for f in files: os.remove(f)
    #     print(">> 🧹 Old Brains wiped for upgrade.")
    
    # --- 1. LSTM EĞİTİM KONTROLÜ ---
    lstm_trainer = LSTMTrainer()
    print(">> 🧠 Checking LSTM Brains...")
    for symbol in Config.TARGET_COINS:
        model_path, _ = lstm_trainer._get_paths(symbol)
        
        # Model yoksa EĞİT
        if not os.path.exists(model_path):
            print(f">> ⏳ LSTM Brain missing for {symbol}. Training with new features...")
            try:
                await lstm_trainer.train_model_for_symbol(symbol)
            except Exception as e:
                print(f">> ❌ LSTM Training Failed: {e}")

    # --- 2. RL AJAN KONTROLÜ (Eğitimi Atlıyoruz - Çok uzun sürüyor) ---
    rl_trainer = RLTrainer()
    rl_model_path = rl_trainer.MODEL_PATH + ".zip"
    
    if not os.path.exists(rl_model_path):
        print(">> 🤖 RL Agent not found. Will use LSTM fallback until trained.")
        print(">> ℹ️  To train RL manually: python -m src.brain.rl_trainer")
        # RL eğitimi 10+ saat sürüyor, bu yüzden atlıyoruz
        # Engine LSTM fallback ile çalışacak

    # --- 3. BOTU BAŞLAT ---
    print(">> 🚀 All systems ready. Launching Engine.")
    bot = BotEngine()
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
    except Exception as e:
        print(f">> FATAL ERROR: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
