import asyncio
import os
from src.core.engine import BotEngine
from src.utils.logger import setup_logger
from src.config.settings import Config
from src.brain.trainer import AITrainer

async def main():
    setup_logger()
    print(f">> Starting DEMIR AI v{Config.VERSION}")
    
    # --- ÇOKLU MODEL EĞİTİM KONTROLÜ ---
    trainer = AITrainer()
    
    print(">> 🧠 Checking AI Brains for Target Assets...")
    
    for symbol in Config.TARGET_COINS:
        # Model dosyasının yolunu al
        model_path, _ = trainer._get_paths(symbol)
        
        if not os.path.exists(model_path):
            print(f">> ⏳ Brain missing for {symbol}. Starting training...")
            try:
                await trainer.train_model_for_symbol(symbol)
                print(f">> ✅ Training Complete for {symbol}.")
            except Exception as e:
                print(f">> ❌ Training Failed for {symbol}: {e}")
        else:
            print(f">> 🧠 Brain active for {symbol}.")

    # --- BOTU BAŞLAT ---
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
