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
    """Her sembol için ayrı model eğit."""
    from src.models.trainer import QuantModelTrainer
    
    logger.info("="*50)
    logger.info("🧠 TRAINING SEPARATE MODELS")
    logger.info("="*50)
    
    results = {}
    
    for symbol in symbols:
        path = f"data/processed/{symbol}_features.parquet"
        try:
            df = pd.read_parquet(path)
            logger.info(f"\n📊 Training model for {symbol}: {len(df)} samples")
        except FileNotFoundError:
            logger.warning(f"  {symbol} not found, run --features first")
            continue
        
        # Her sembol için ayrı trainer
        model_name = f"quant_{symbol.lower().replace('usdt', '')}"
        trainer = QuantModelTrainer(model_name)
        
        result = trainer.train(df, target_col="label_4h")
        results[symbol] = result
        
        logger.info(f"  ✅ {symbol} Model: {result.model_name}_{result.version}")
        logger.info(f"     Train: {result.train_accuracy:.2%} | Val: {result.val_accuracy:.2%}")
    
    logger.info("\n" + "="*50)
    logger.info("✅ ALL MODELS TRAINED")
    logger.info("="*50)
    
    return results


def run_backtest(symbols: list):
    """Gelişmiş Risk Yönetimi ile Backtest çalıştır."""
    from src.execution.backtester import AdvancedBacktester
    
    logger.info("="*50)
    logger.info("📈 RUNNING ADVANCED BACKTEST (RISK MANAGED)")
    logger.info("="*50)
    
    for symbol in symbols:
        model_name = f"quant_{symbol.lower().replace('usdt', '')}"
        
        try:
            backtester = AdvancedBacktester(symbol, model_name)
        except Exception as e:
            logger.warning(f"Skipping {symbol}: {e}")
            continue
        
        path = f"data/processed/{symbol}_features.parquet"
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            logger.warning(f"  {symbol} data not found, skipping...")
            continue
        
        # Simülasyonu başlat
        results = backtester.run(df)
        
        # Ekstra analiz gerekirse burada yapılabilir
        # Örneğin max drawdown hesaplama (backtester raporluyor ama burada özet geçebiliriz)



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
