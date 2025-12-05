import logging
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import json
import os

from src.brain.rl_trainer import RLTrainer
from src.config.settings import Config
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.brain.feature_engineering import FeatureEngineer
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("AUTO_TRAINER")

class AutoTrainingScheduler:
    """
    AUTO-TRAINING SCHEDULER
    
    Retrains the RL Agent weekly (every Sunday at 2 AM) using fresh market data.
    Ensures the AI continuously adapts to changing market conditions.
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.training_history_path = "training_history.json"
        self.is_training = False
        logger.info("🔄 Auto-Training Scheduler initialized")
    
    def start(self):
        """Start the weekly training scheduler"""
        # Schedule training every Sunday at 02:00 AM
        self.scheduler.add_job(
            self.run_training,
            CronTrigger(day_of_week='sun', hour=2, minute=0),
            id='weekly_rl_training',
            name='Weekly RL Agent Retraining',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("✅ Weekly training scheduled: Every Sunday at 02:00 AM")
    
    async def run_training(self):
        """Execute the RL training process"""
        if self.is_training:
            logger.warning("⚠️ Training already in progress. Skipping this cycle.")
            return
        
        self.is_training = True
        training_start = datetime.now()
        
        logger.info("=" * 60)
        logger.info(f"🧠 AUTO-TRAINING STARTED: {training_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        try:
            # Fetch Fresh Data (last 30 days)
            symbol = Config.TARGET_COINS[0]  # Train on BTC
            logger.info(f"📡 Fetching training data for {symbol}...")
            
            binance = BinanceConnector()
            raw_data = await binance.fetch_candles(symbol, timeframe='1h', limit=720)  # 30 days
            await binance.close()
            
            if not raw_data:
                logger.error("❌ Failed to fetch training data. Aborting.")
                self._log_training_result(success=False, error="No data fetched")
                return
            
            logger.info(f"✅ Fetched {len(raw_data)} candles")
            
            # Process Data
            crypto_df = FeatureEngineer.process_data(raw_data)
            macro_connector = MacroConnector()
            macro_df = await macro_connector.fetch_macro_data(period="1mo", interval="1h")
            df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
            
            if df is None or len(df) < 100:
                logger.error("❌ Insufficient data after processing. Aborting.")
                self._log_training_result(success=False, error="Insufficient processed data")
                return
            
            logger.info(f"✅ Processed {len(df)} rows of training data")
            
            # Train RL Agent (wrap sync method in executor to avoid blocking)
            logger.info("🧠 Starting RL Agent training (5k timesteps)...")
            trainer = RLTrainer()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, trainer.train, symbol, df, 5000)
            
            training_duration = (datetime.now() - training_start).total_seconds()
            logger.info(f"✅ TRAINING COMPLETE in {training_duration:.1f}s")
            
            self._log_training_result(success=True, duration=training_duration)
            
        except Exception as e:
            logger.error(f"❌ TRAINING FAILED: {e}")
            self._log_training_result(success=False, error=str(e))
        
        finally:
            self.is_training = False
            logger.info("=" * 60)
    
    def _log_training_result(self, success: bool, duration: float = 0, error: str = ""):
        """Log training results to JSON for dashboard"""
        try:
            result = {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "duration_seconds": duration,
                "error": error if error else None
            }
            
            # Load existing history
            if os.path.exists(self.training_history_path):
                with open(self.training_history_path, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(result)
            
            # Keep only last 10 training results
            if len(history) > 10:
                history = history[-10:]
            
            with open(self.training_history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"📝 Training result logged")
        except Exception as e:
            logger.error(f"Error logging training result: {e}")
    
    def get_next_training_time(self) -> str:
        """Get the timestamp of the next scheduled training"""
        next_run = self.scheduler.get_job('weekly_rl_training').next_run_time
        return next_run.strftime('%Y-%m-%d %H:%M:%S') if next_run else "Not scheduled"
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("🛑 Auto-Training Scheduler stopped")
