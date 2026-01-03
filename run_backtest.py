# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - BACKTEST RUNNER
===============================
Profesyonel quant trading sistemi için ana çalıştırma scripti.

KULLANIM:
1. Veri indir: python run_backtest.py --download
2. Model eğit: python run_backtest.py --train
3. Backtest:   python run_backtest.py --backtest
4. Hepsi:      python run_backtest.py --all

KURAL: Mock veri YOK - Sadece gerçek Binance verisi!

Author: DEMIR AI Team
Date: 2026-01-03
"""
import asyncio
import argparse
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BACKTEST_RUNNER")


async def download_data(symbols: list, days: int = 365):
    """2 yıllık veri indir."""
    from src.data_pipeline.collector import get_data_collector
    
    logger.info("="*50)
    logger.info("📥 DOWNLOADING HISTORICAL DATA")
    logger.info("="*50)
    
    collector = get_data_collector()
    await collector.download_all_symbols(symbols, interval="1m", days=days)
    
    # Özet
    summary = collector.get_data_summary()
    logger.info("\n📊 DATA SUMMARY:")
    for symbol, info in summary.items():
        logger.info(f"  {symbol}: {info['candles']:,} candles | {info['file_size_mb']:.1f} MB")


def generate_features(symbols: list):
    """Feature'ları hesapla."""
    from src.data_pipeline.collector import get_data_collector
    from src.features.technical import get_technical_features
    
    logger.info("="*50)
    logger.info("⚙️ GENERATING FEATURES")
    logger.info("="*50)
    
    collector = get_data_collector()
    features = get_technical_features()
    
    for symbol in symbols:
        df = collector.load_symbol(symbol)
        if df is None:
            logger.warning(f"No data for {symbol}, skipping...")
            continue
        
        logger.info(f"\n📊 Processing {symbol}...")
        result = features.calculate_all(df)
        
        # Kaydet
        output_path = f"data/processed/{symbol}_features.parquet"
        result.to_parquet(output_path, index=False)
        logger.info(f"  Saved: {output_path} ({len(result)} rows, {len(result.columns)} features)")


def train_model(symbols: list):
    """Model eğit."""
    from src.models.trainer import get_model_trainer
    
    logger.info("="*50)
    logger.info("🧠 TRAINING MODEL")
    logger.info("="*50)
    
    # Tüm verileri birleştir
    all_data = []
    for symbol in symbols:
        path = f"data/processed/{symbol}_features.parquet"
        try:
            df = pd.read_parquet(path)
            df['symbol'] = symbol
            all_data.append(df)
            logger.info(f"  Loaded {symbol}: {len(df)} samples")
        except FileNotFoundError:
            logger.warning(f"  {symbol} not found, run --features first")
    
    if not all_data:
        logger.error("No feature data found!")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"\n📊 Combined dataset: {len(combined)} samples")
    
    # Eğit
    trainer = get_model_trainer("quant_btc_eth")
    result = trainer.train(combined, target_col="label_4h")
    
    logger.info("\n" + "="*50)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("="*50)
    logger.info(f"  Model: {result.model_name}_{result.version}")
    logger.info(f"  Train Accuracy: {result.train_accuracy:.2%}")
    logger.info(f"  Validation Accuracy: {result.val_accuracy:.2%}")
    logger.info(f"  Walk-Forward Avg: {result.walk_forward_results.get('avg_accuracy', 0):.2%}")
    
    return result


def run_backtest(symbols: list):
    """Backtest çalıştır."""
    from src.models.trainer import get_model_trainer
    
    logger.info("="*50)
    logger.info("📈 RUNNING BACKTEST")
    logger.info("="*50)
    
    trainer = get_model_trainer("quant_btc_eth")
    
    try:
        trainer.load_model()
    except FileNotFoundError:
        logger.error("No trained model found! Run --train first")
        return None
    
    # Her sembol için backtest
    for symbol in symbols:
        path = f"data/processed/{symbol}_features.parquet"
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            logger.warning(f"  {symbol} not found, skipping...")
            continue
        
        logger.info(f"\n📊 Backtesting {symbol}...")
        result = trainer.backtest(df)
        
        logger.info(f"  Trades: {result.total_trades}")
        logger.info(f"  Win Rate: {result.win_rate:.2%}")
        logger.info(f"  Sharpe: {result.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"  Profit Factor: {result.profit_factor:.2f}")


async def run_all(symbols: list, days: int):
    """Tüm pipeline'ı çalıştır."""
    logger.info("🚀 RUNNING FULL PIPELINE")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Days: {days}")
    
    # 1. Download
    await download_data(symbols, days)
    
    # 2. Features
    generate_features(symbols)
    
    # 3. Train
    train_model(symbols)
    
    # 4. Backtest
    run_backtest(symbols)
    
    logger.info("\n" + "="*50)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("="*50)


def main():
    parser = argparse.ArgumentParser(description="DEMIR AI Quant Backtest")
    parser.add_argument("--download", action="store_true", help="Download historical data")
    parser.add_argument("--features", action="store_true", help="Generate features")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to process")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    
    args = parser.parse_args()
    
    if args.all:
        asyncio.run(run_all(args.symbols, args.days))
    elif args.download:
        asyncio.run(download_data(args.symbols, args.days))
    elif args.features:
        generate_features(args.symbols)
    elif args.train:
        train_model(args.symbols)
    elif args.backtest:
        run_backtest(args.symbols)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
