"""
v5 Training Script - Runs all 4 coin trainings SEQUENTIALLY
Run with: python train_v5_all.py
"""
import sys
sys.path.insert(0, '.')

import asyncio
import logging
from src.brain.rl_agent.trainer import RLTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("V5_TRAINER")

COINS = [
    ("BTC/USDT", "ppo_btc_v5"),
    ("ETH/USDT", "ppo_eth_v5"),
    ("LTC/USDT", "ppo_ltc_v5"),
    ("SOL/USDT", "ppo_sol_v5"),
]

async def train_all():
    """Train all coins sequentially to avoid memory/crash issues."""
    
    for symbol, save_name in COINS:
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting {symbol} v5 Training...")
        logger.info(f"{'='*60}")
        
        try:
            trainer = RLTrainer(symbol)
            await trainer.train(
                total_timesteps=500000,
                save_name=save_name,
                num_candles=43800  # 5 years
            )
            logger.info(f"✅ {symbol} v5 model saved as {save_name}.zip")
        except Exception as e:
            logger.error(f"❌ {symbol} training failed: {e}")
            continue
    
    logger.info("\n" + "="*60)
    logger.info("🎉 ALL v5 TRAININGS COMPLETE!")
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(train_all())
