"""
RL Agent Offline Trainer
(RL Ajanı Offline Eğitici)

Trains PPO agent on historical BTC/ETH data
(PPO ajanını tarihsel BTC/ETH verisi üzerinde eğitir)

Usage:
    python -m src.brain.rl_agent.trainer --symbol BTCUSDT --steps 100000
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
import argparse

from src.brain.rl_agent.trading_env import TradingEnv
from src.brain.rl_agent.ppo_agent import RLAgent
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.brain.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RL_TRAINER")

class RLTrainer:
    """
    Offline RL Training Pipeline
    (Offline RL Eğitim Hattı)
    """
    
    def __init__(self, symbol: str = "BTC/USDT"):
        self.symbol = symbol
        self.connector = BinanceConnector()
        
    async def prepare_training_data(self, num_candles: int = 10000) -> np.ndarray:
        """
        Fetch and prepare historical data for training
        (Eğitim için tarihsel veriyi çek ve hazırla)
        
        Returns:
            data: (N, 28) array - Each row is a timestep with 28 features
        """
        logger.info(f"📥 Fetching {num_candles} candles for {self.symbol}...")
        
        # Fetch raw OHLCV (Ham OHLCV verisi çek)
        raw_data = await self.connector.fetch_candles(self.symbol, limit=num_candles)
        await self.connector.close()
        
        if not raw_data or len(raw_data) < 100:
            raise ValueError("Insufficient data fetched!")
        
        # Process with FeatureEngineer (FeatureEngineer ile işle)
        logger.info("⚙️ Processing features...")
        df = await asyncio.to_thread(FeatureEngineer.process_data, raw_data)
        
        # Select features for RL environment (RL ortamı için özellikleri seç)
        # Columns: price features (20) + macro (5) + position (3, will be added by env)
        # For now, use first 25 columns (20 price + 5 macro placeholders)
        # Real implementation should integrate MacroConnector
        
        feature_cols = [
            'close', 'volume', 'rsi', 'macd', 'bb_percent', 'atr',
            'adx', 'obv', 'stoch_rsi', 'ema_20', 'sma_50', 
            'returns_1h', 'returns_4h', 'returns_1d',
            'volume_sma_ratio', 'price_sma_ratio', 'momentum',
            'bollinger_width', 'macd_signal', 'rsi_ma'
        ]
        
        # Pad with zeros for macro (will be replaced with real macro later)
        macro_placeholder = np.zeros((len(df), 5))
        
        # Extract price features (Fiyat özelliklerini çıkar)
        price_features = df[feature_cols].values[:, :20]  # First 20
        
        # Combine (Birleştir)
        data = np.concatenate([price_features, macro_placeholder], axis=1)
        
        logger.info(f"✅ Prepared {data.shape[0]} timesteps with {data.shape[1]} features")
        return data.astype(np.float32)
    
    async def train(
        self, 
        total_timesteps: int = 100_000,
        initial_balance: float = 10_000.0,
        save_name: str = "ppo_trader_v1"
    ):
        """
        Run complete training pipeline
        (Tam eğitim hattını çalıştır)
        """
        logger.info("🚀 Starting RL Agent Training Pipeline...")
        
        # 1. Prepare data (Veri hazırla)
        data = await self.prepare_training_data(num_candles=10000)
        
        # 2. Create environment (Ortam oluştur)
        logger.info("🏗️ Creating TradingEnv...")
        env = TradingEnv(
            data=data,
            initial_balance=initial_balance,
            transaction_fee=0.001  # 0.1% (Binance spot fee)
        )
        
        # 3. Create RL agent (RL ajanı oluştur)
        logger.info("🤖 Creating RL Agent with Transformer policy...")
        agent = RLAgent()
        agent.create_model(env, learning_rate=3e-4)
        
        # 4. Train (Eğit)
        logger.info(f"🎓 Training for {total_timesteps} timesteps...")
        logger.info("📊 TensorBoard: tensorboard --logdir ./tensorboard/")
        
        agent.train(total_timesteps=total_timesteps, tb_log_name="ppo_trader")
        
        # 5. Save model (Modeli kaydet)
        logger.info("💾 Saving trained model...")
        agent.save(filename=save_name)
        
        # 6. Evaluate (Değerlendir)
        logger.info("📈 Running evaluation...")
        await self.evaluate(agent, env)
        
        logger.info("✅ Training pipeline complete!")
        
    async def evaluate(self, agent: RLAgent, env: TradingEnv, num_episodes: int = 10):
        """
        Evaluate trained agent performance
        (Eğitilmiş ajan performansını değerlendir)
        """
        total_rewards = []
        total_sharpes = []
        
        for ep in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            sharpe = info.get('sharpe', 0)
            total_rewards.append(episode_reward)
            total_sharpes.append(sharpe)
            
            logger.info(f"Episode {ep+1}: Reward={episode_reward:.2f}, Sharpe={sharpe:.2f}")
        
        avg_reward = np.mean(total_rewards)
        avg_sharpe = np.mean(total_sharpes)
        
        logger.info(f"\n📊 Evaluation Results ({num_episodes} episodes):")
        logger.info(f"   Avg Reward: {avg_reward:.2f}")
        logger.info(f"   Avg Sharpe: {avg_sharpe:.2f}")
        logger.info(f"   Std Reward: {np.std(total_rewards):.2f}")


async def main():
    """Main training entry point (Ana eğitim giriş noktası)"""
    parser = argparse.ArgumentParser(description="Train RL Trading Agent")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--steps", type=int, default=100_000, help="Training timesteps")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Initial balance")
    parser.add_argument("--name", type=str, default="ppo_trader_v1", help="Model save name")
    
    args = parser.parse_args()
    
    trainer = RLTrainer(symbol=args.symbol)
    await trainer.train(
        total_timesteps=args.steps,
        initial_balance=args.balance,
        save_name=args.name
    )


if __name__ == "__main__":
    asyncio.run(main())
