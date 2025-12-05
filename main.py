import asyncio
import os
import glob # Dosya silmek için
from src.core.engine import BotEngine
from src.utils.logger import setup_logger
from src.config.settings import Config
from src.brain.trainer import AITrainer as LSTMTrainer
from src.brain.rl_trainer import RLTrainer  # RL Eğitim YENİDEN AKTİF!

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
    
    # --- 1. LSTM EĞİTİM KONTROLÜ (DİSABLED - Auto-Trainer handles it) ---
    # LSTM artık sadece AutoTrainingScheduler üzerinden haftalık eğitiliyor.
    # Startup'ta eğitimi DİSABLE ettik çünkü Railway'de 10 saat bloke ediyor.
    # RL Agent yeterli, LSTM yoksa fallback logic kullanılır.
    
    print(">> 🧠 LSTM Training: DISABLED (handled by weekly scheduler)")
    print(">> ℹ️ RL Agent will use fallback logic if LSTM models are missing.")

    # --- 2. RL EĞİTİMİ (GERÇEK AI!) ---
    rl_model_path = "src/brain/models/storage/rl_agent_v2_recurrent.zip"
    if not os.path.exists(rl_model_path):
        print(">> 🤖 RL Agent missing. Training TRUE AI Brain...")
        try:
            rl_trainer = RLTrainer()
            await rl_trainer.train_agent(Config.TARGET_COINS[0])  # BTC ile eğit
            print(">> ✅ RL Agent trained successfully!")
        except Exception as e:
            print(f">> ⚠️ RL Training Failed: {e}. Using LSTM fallback.")
    else:
        print(">> ✅ RL Agent already trained. Loading...")


    # --- 3. AUTO-TRAINING SCHEDULER (HAFTALIK OTOMATİK EĞİTİM) ---
    from src.brain.training_scheduler import AutoTrainingScheduler
    
    auto_trainer = AutoTrainingScheduler()
    auto_trainer.start()
    print(f">> 🔄 Auto-Training Scheduled: {auto_trainer.get_next_training_time()}")

    # --- 4. BOTU BAŞLAT ---
    print(">> 🚀 All systems ready. Launching Engine.")
    bot = BotEngine()
    try:
        await bot.start()
    except KeyboardInterrupt:
        auto_trainer.stop()
        await bot.stop()
    except Exception as e:
        print(f">> FATAL ERROR: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
