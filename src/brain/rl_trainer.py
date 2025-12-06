import os
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO # LSTM Destekli PPO
from src.brain.trading_env import TradingEnv
from src.brain.feature_engineering import FeatureEngineer
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.macro_connector import MacroConnector
import asyncio

logger = logging.getLogger("RL_TRAINER_PRO")

class RLTrainer:
    """
    PEKİŞTİRMELİ ÖĞRENME EĞİTMENİ (Deep Recurrent RL)
    RecurrentPPO (LSTM) kullanarak piyasanın 'hafızasını' tutar.
    """
    
    MODEL_PATH = "src/brain/models/storage/rl_agent_v2_recurrent"
    
    def __init__(self):
        self.connector = BinanceConnector()
        self.macro = MacroConnector()

    async def prepare_data(self, symbol="BTC/USDT"):
        """Eğitim için veriyi hazırlar ve temizler."""
        logger.info("Fetching data for RL Training...")
        
        # 1. Veri Çek (Daha fazla veri = Daha iyi eğitim)
        raw_crypto = await self.connector.fetch_candles(symbol, limit=5000)
        await self.connector.close()
        
        if not raw_crypto: return None
        
        crypto_df = FeatureEngineer.process_data(raw_crypto)
        
        # Macro data (using helper)
        from src.brain.macro_helpers import fetch_macro_for_training
        df, macro_df = await fetch_macro_for_training(self.macro, crypto_df, period="1y", interval="1h")
        
        # If helper returned separate, merge
        if not macro_df.empty:
            df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        # 2. Rest of processing continues...
        
        # 3. TEMİZLİK
        drop_cols = ['timestamp', 'symbol', 'source', 'target', 'open', 'high', 'low'] 
        numeric_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        numeric_df.fillna(0, inplace=True)
        
        return numeric_df

    async def train_agent(self, symbol="BTC/USDT"):
        logger.info("🚀 Starting Deep Recurrent RL Training (LSTM-PPO)...")
        
        df = await self.prepare_data(symbol)
        if df is None:
            logger.error("No data for RL training.")
            return
        
    def _train_sync(self, df):
        """Blocking training logic to be run in a separate thread."""
        logger.info(f"Training Environment Ready. Data Shape: {df.shape}")
        
        # Ortamı Kur
        env = TradingEnv(df)
        
        # Modeli Tanımla (RecurrentPPO - LSTM Policy)
        try:
            model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, learning_rate=0.0003, n_steps=128)
            logger.info("Using RecurrentPPO (LSTM) Architecture.")
        except ImportError:
            logger.warning("sb3-contrib not found. Falling back to standard PPO.")
            model = PPO("MlpPolicy", env, verbose=1)
        
        # Eğit
        FAST_MODE = True
        timesteps = 10000 if FAST_MODE else 50000
        
        logger.info(f"🏋️ Training for {timesteps} timesteps (FAST_MODE={FAST_MODE})")
        model.learn(total_timesteps=timesteps)
        
        # Kaydet
        if not os.path.exists("src/brain/models/storage"):
            os.makedirs("src/brain/models/storage")
            
        model.save(self.MODEL_PATH)
        logger.info(f"✅ RL AGENT SAVED at {self.MODEL_PATH}.zip")

    async def train_agent(self, symbol="BTC/USDT"):
        logger.info("🚀 Starting Deep Recurrent RL Training (LSTM-PPO)...")
        
        # 1. Async Data Fetch (IO Bound - Keep in Main Thread)
        df = await self.prepare_data(symbol)
        if df is None:
            logger.error("No data for RL training.")
            return
        
        # 2. Sync Training (CPU Bound - Move to Thread)
        logger.info("⏳ Offloading RL training to background thread...")
        await asyncio.to_thread(self._train_sync, df)

if __name__ == "__main__":
    trainer = RLTrainer()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(trainer.train_agent())
