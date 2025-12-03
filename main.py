import asyncio
import os
from src.core.engine import BotEngine
from src.utils.logger import setup_logger
from src.config.settings import Config
from src.brain.trainer import AITrainer as LSTMTrainer
from src.brain.rl_trainer import RLTrainer # <-- YENİ

async def main():
    setup_logger()
    print(f">> Starting DEMIR AI v{Config.VERSION}")
    print(f">> Mode: {Config.ENVIRONMENT}")
    
    # --- 1. LSTM EĞİTİM KONTROLÜ (ESKİ BEYİN) ---
    lstm_trainer = LSTMTrainer()
    print(">> 🧠 Checking LSTM Brains...")
    for symbol in Config.TARGET_COINS:
        model_path, _ = lstm_trainer._get_paths(symbol)
        if not os.path.exists(model_path):
            print(f">> ⏳ LSTM Brain missing for {symbol}. Training...")
            try:
                await lstm_trainer.train_model_for_symbol(symbol)
            except Exception as e:
                print(f">> ❌ LSTM Training Failed: {e}")

    # --- 2. RL AJAN EĞİTİM KONTROLÜ (YENİ SÜPER ZEKA) ---
    rl_trainer = RLTrainer()
    # .zip uzantısı Stable-Baselines3 tarafından otomatik eklenir
    rl_model_path = rl_trainer.MODEL_PATH + ".zip"
    
    if not os.path.exists(rl_model_path):
        print(">> 🤖 RL Agent (PPO) not found. Starting Reinforcement Learning...")
        print(">> 🎮 The Agent is entering the simulation (Matrix)...")
        try:
            # Genelde BTC verisiyle genel bir ajan eğitmek başlangıç için iyidir
            await rl_trainer.train_agent("BTC/USDT")
            print(">> ✅ RL Agent Trained & Saved.")
        except Exception as e:
            print(f">> ❌ RL Training Failed: {e}")
    else:
        print(f">> 🤖 RL Agent active at {rl_model_path}")

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
