import asyncio
import logging
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(".")

from src.data_pipeline.collector import get_data_collector
from src.features.technical import TechnicalFeatures
from src.models.trainer import QuantModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TRAIN_V11")

async def train_symbol(symbol: str, days: int = 180):
    """Train model for a single symbol."""
    logger.info(f"🚀 Starting training for {symbol} (Last {days} days)...")
    
    # 1. Download Data
    collector = get_data_collector()
    logger.info(f"📥 Downloading data for {symbol}...")
    df = await collector.download_symbol(symbol, interval="1m", days=days)
    
    if df is None or len(df) < 1000:
        logger.error(f"❌ Not enough data for {symbol}")
        return
    
    logger.info(f"✅ Data loaded: {len(df)} candles")
    
    # 2. Feature Engineering
    logger.info("🛠️ Calculating features...")
    features = TechnicalFeatures()
    df_features = features.calculate_all(df)
    
    # Remove NaN
    df_features = df_features.dropna()
    logger.info(f"✅ Features ready: {len(df_features)} rows, {len(df_features.columns)} columns")
    
    # 3. Train Model
    logger.info("🧠 Training LightGBM model...")
    model_name = f"quant_{symbol.lower().replace('usdt', '')}"
    trainer = QuantModelTrainer(model_name)
    
    # Train
    result = trainer.train(
        df_features, 
        target_col="label_4h",  # 4 saatlik tahmin
        test_size=0.2
    )
    
    logger.info(f"🏆 Training Complete for {symbol}")
    logger.info(f"   Train Acc: {result.train_accuracy:.1%}")
    logger.info(f"   Val Acc:   {result.val_accuracy:.1%}")
    
    # 4. Backtest (Simple)
    logger.info("📉 Running quick backtest...")
    bt_result = trainer.backtest(df_features, initial_capital=1000)
    
    logger.info(f"📊 Backtest Results ({symbol}):")
    logger.info(f"   Trades: {bt_result.total_trades}")
    logger.info(f"   Win Rate: {bt_result.win_rate:.1f}%")
    logger.info(f"   Profit Factor: {bt_result.profit_factor:.2f}")
    logger.info(f"   Return: {bt_result.avg_return:.2f}%")
    
    return result

async def main():
    logger.info("==========================================")
    logger.info("🔥 DEMIR AI v11 - MODEL RETRAINING STARTED")
    logger.info("==========================================")
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    for symbol in symbols:
        try:
            await train_symbol(symbol)
        except Exception as e:
            logger.error(f"❌ Failed to train {symbol}: {e}")
            import traceback
            traceback.print_exc()
            
    logger.info("✨ ALL TRAINING COMPLETED. Models saved to data/models/")

if __name__ == "__main__":
    try:
        # Windows selector event loop policy fix
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
