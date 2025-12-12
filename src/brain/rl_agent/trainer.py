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
        # Use public API - no authentication needed for OHLCV
        # (Public API kullan - OHLCV için auth gerekmez)
        self.exchange = None
        
    async def _get_exchange(self):
        """Create public exchange connection (no API key needed)"""
        if not self.exchange:
            import ccxt.async_support as ccxt
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
        return self.exchange
        
    async def prepare_training_data(self, num_candles: int = 10000) -> np.ndarray:
        """
        Fetch and prepare historical data for training using PUBLIC API
        (Public API kullanarak eğitim için tarihsel veriyi çek)
        
        No API key required!
        
        Returns:
            data: (N, 25) array - Each row is a timestep with 25 features
        """
        logger.info(f"📥 Fetching {num_candles} candles for {self.symbol} (PUBLIC API)...")
        
        exchange = await self._get_exchange()
        
        try:
            # Fetch OHLCV in batches (Binance limit: 1000 per request)
            # Need multiple requests for 2+ years of data
            all_ohlcv = []
            fetched = 0
            since = None  # Start from now, go backwards
            
            # First fetch to get latest data
            ohlcv = await exchange.fetch_ohlcv(
                self.symbol.replace('/', ''), 
                timeframe='1h', 
                limit=1000
            )
            all_ohlcv = ohlcv
            fetched += len(ohlcv)
            logger.info(f"   Batch 1: {len(ohlcv)} candles (total: {fetched})")
            
            # Fetch more batches going backwards in time
            while fetched < num_candles and len(ohlcv) > 0:
                # Calculate timestamp 1000 hours before oldest candle
                oldest_timestamp = all_ohlcv[0][0]
                since = oldest_timestamp - (1000 * 60 * 60 * 1000)  # 1000 hours earlier
                
                # Rate limit protection
                await asyncio.sleep(0.5)
                
                ohlcv = await exchange.fetch_ohlcv(
                    self.symbol.replace('/', ''), 
                    timeframe='1h',
                    since=since,
                    limit=1000
                )
                
                if len(ohlcv) == 0:
                    logger.warning("No more historical data available")
                    break
                
                # Prepend older data
                all_ohlcv = ohlcv + all_ohlcv
                fetched = len(all_ohlcv)
                batch_num = fetched // 1000 + 1
                logger.info(f"   Batch {batch_num}: {len(ohlcv)} candles (total: {fetched})")
            
            # Limit to requested amount
            if len(all_ohlcv) > num_candles:
                all_ohlcv = all_ohlcv[-num_candles:]  # Keep most recent
            
            logger.info(f"✅ Fetched {len(all_ohlcv)} candles (~{len(all_ohlcv)//24} days)")
            
            # Convert to list of dicts for FeatureEngineer
            raw_data = []
            for candle in all_ohlcv:
                raw_data.append({
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'symbol': self.symbol
                })
        finally:
            await exchange.close()
            self.exchange = None
        
        if not raw_data or len(raw_data) < 100:
            raise ValueError("Insufficient data fetched!")
        
        # Process with FeatureEngineer (FeatureEngineer ile işle)
        logger.info("⚙️ Processing features...")
        df = await asyncio.to_thread(FeatureEngineer.process_data, raw_data)
        
        # Select features for RL environment (RL ortamı için özellikleri seç)
        # Use actual columns from FeatureEngineer.process_data()
        
        # Numeric columns that exist in processed DataFrame
        available_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove unwanted columns
        exclude_cols = ['timestamp', 'symbol']
        feature_cols = [c for c in available_numeric_cols if c not in exclude_cols]
        
        # Limit to first 25 features for consistency
        feature_cols = feature_cols[:25]
        
        logger.info(f"📊 Using {len(feature_cols)} features: {feature_cols[:10]}...")
        
        # Extract features (NaN'ları 0 yap)
        data = df[feature_cols].fillna(0).values.astype(np.float32)
        
        logger.info(f"✅ Prepared {data.shape[0]} timesteps with {data.shape[1]} features")
        return data
    
    async def train(
        self, 
        total_timesteps: int = 100_000,
        initial_balance: float = 10_000.0,
        save_name: str = "ppo_trader_v1",
        num_candles: int = 10_000  # 2 yıl = 17,520 candle
    ):
        """
        Run complete training pipeline
        (Tam eğitim hattını çalıştır)
        """
        logger.info(f"🚀 Starting RL Agent Training Pipeline ({num_candles} candles, {total_timesteps} steps)...")
        
        # 1. Prepare data (Veri hazırla)
        data = await self.prepare_training_data(num_candles=num_candles)
        
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
    parser.add_argument("--candles", type=int, default=10_000, help="Number of candles (17520=2 years)")
    
    args = parser.parse_args()
    
    trainer = RLTrainer(symbol=args.symbol)
    await trainer.train(
        total_timesteps=args.steps,
        initial_balance=args.balance,
        save_name=args.name,
        num_candles=args.candles
    )


if __name__ == "__main__":
    asyncio.run(main())
