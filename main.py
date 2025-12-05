import asyncio
import os
import glob # Dosya silmek için
from src.core.engine import BotEngine
from src.utils.logger import setup_logger
from src.config.settings import Config
from src.brain.trainer import AITrainer as LSTMTrainer
from src.brain.rl_trainer import RLTrainer  # RL Eğitim YENİDEN AKTİF!

async def background_train_lstm():
    """
    ARKA PLAN LSTM EĞİTİMİ (Non-Blocking)
    Bot çalışırken arka planda modelleri eğitir.
    5-10 saat sonra otomatik aktif olur.
    """
    print(">> 🧠 Starting BACKGROUND LSTM Training...")
    lstm_trainer = LSTMTrainer()
    
    for symbol in Config.TARGET_COINS:
        model_path, _ = lstm_trainer._get_paths(symbol)
        
        if not os.path.exists(model_path):
            print(f">> ⏳ Training LSTM for {symbol} in background...")
            try:
                await lstm_trainer.train_model_for_symbol(symbol)
                print(f">> ✅ {symbol} LSTM trained successfully!")
            except Exception as e:
                print(f">> ❌ {symbol} LSTM training failed: {e}")
        else:
            print(f">> ✅ {symbol} LSTM already trained. Skipping.")
    
    print(">> 🎉 All LSTM models ready!")

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
    
    # --- 1. LSTM EĞİTİM (BACKGROUND) ---
    # LSTM eğitimi arka planda başlatılıyor (bot'u bloke etmez)
    print(">> 🧠 LSTM Training: BACKGROUND MODE (non-blocking)")
    asyncio.create_task(background_train_lstm())

    # --- 2. RL EĞİTİMİ (ARKA PLANDA!) ---
    rl_model_path = "src/brain/models/storage/rl_agent_v2_recurrent.zip"
    if not os.path.exists(rl_model_path):
        print(">> 🤖 RL Agent missing. Starting training in BACKGROUND...")
        rl_trainer = RLTrainer()
        # BACKGROUND: Don't await - let engine start immediately
        asyncio.create_task(rl_trainer.train_agent(Config.TARGET_COINS[0]))
        print(">> 🧠 RL Training running in background (non-blocking)")
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
