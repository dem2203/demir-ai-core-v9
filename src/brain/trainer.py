import pandas as pd
import numpy as np
import os
import logging
import joblib
import asyncio
import argparse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from src.brain.models.transformer import TimeNet
from src.brain.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI_TRAINER_TRANSFORMER")

class AITrainer:
    """
    DEMIR AI v11.1 - MULTI-ASSET TRAINER (Phase X: TimeNet Upgrade)
    Her coin için ayrı bir Transformer (TimeNet) modeli eğitir ve kaydeder.
    Uses PUBLIC API - no authentication required!
    """
    
    MODELS_DIR = "src/brain/models/storage"
    LOOKBACK = 60 

    def __init__(self):
        self.exchange = None
        
        # Klasör yoksa oluştur
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR)
    
    async def _get_exchange(self):
        """Create public exchange connection (no API key needed)"""
        if not self.exchange:
            import ccxt.async_support as ccxt
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
        return self.exchange

    def _get_paths(self, symbol):
        """Coin ismine özel dosya yolları üretir."""
        clean_sym = symbol.replace("/", "")
        model_path = os.path.join(self.MODELS_DIR, f"lstm_v11_{clean_sym}.h5")
        scaler_path = os.path.join(self.MODELS_DIR, f"scaler_{clean_sym}.pkl")
        return model_path, scaler_path

    async def fetch_integrated_data(self, symbol, limit=1000):
        """Fetch market data using PUBLIC API (no auth needed)."""
        logger.info(f"Fetching data for {symbol} using PUBLIC API...")
        
        exchange = await self._get_exchange()
        
        try:
            # Fetch OHLCV using public API
            ohlcv = await exchange.fetch_ohlcv(
                symbol.replace('/', ''), 
                timeframe='1h', 
                limit=min(limit, 1000)
            )
            
            logger.info(f"\u2705 Fetched {len(ohlcv)} candles")
            
            # Convert to list of dicts for FeatureEngineer
            raw_data = []
            for candle in ohlcv:
                raw_data.append({
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'symbol': symbol
                })
        finally:
            await exchange.close()
            self.exchange = None
        
        if not raw_data or len(raw_data) < 100:
            logger.error("Insufficient data fetched!")
            return None
        
        # Process with FeatureEngineer
        logger.info("\u26a1 Processing features...")
        df = await asyncio.to_thread(FeatureEngineer.process_data, raw_data)
        
        return df

    def build_transformer_model(self, input_shape):
        """
        Builds the Phase X TimeNet (Transformer) Model.
        """
        timenet = TimeNet(input_shape=input_shape)
        return timenet.build_model()

    async def train_model_for_symbol(self, symbol):
        """Belirtilen coin için özel model eğitir."""
        model_path, scaler_path = self._get_paths(symbol)
        
        logger.info(f"🚀 Starting FAST Training for {symbol}...")
        
        # 1. Veri (500 = ~21 gün, hızlı eğitim için yeterli)
        df = await self.fetch_integrated_data(symbol, limit=500)
        if df is None: 
            logger.error(f"Data failed for {symbol}")
            return False

        # Hedef
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)

        logger.info("⏳ Offloading Transformer training to background thread...")
        await asyncio.to_thread(self._train_sync, df, model_path, scaler_path)
        
        return True

    def _train_sync(self, df, model_path, scaler_path):
        """Blocking training logic to be run in a separate thread."""
        try:
            feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'source', 'target', 'open', 'high', 'low', 'close', 'volume']]
            data_values = df[feature_cols].values
            target_values = df['target'].values

            # 2. Normalizasyon
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_values)

            # 3. Zaman Serisi
            X, y = [], []
            for i in range(self.LOOKBACK, len(scaled_data)):
                X.append(scaled_data[i-self.LOOKBACK:i])
                y.append(target_values[i])
            
            X, y = np.array(X), np.array(y)

            # 4. Eğitim (Phase X: TimeNet)
            model = self.build_transformer_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=10, batch_size=32, verbose=1)  # Verbose for visibility

            # 5. Kayıt
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"✅ BRAIN SAVED (Background): -> {model_path}")
        except Exception as e:
            logger.error(f"Training failed: {e}")


async def main():
    """Train LSTM models for specified symbols."""
    parser = argparse.ArgumentParser(description="Train LSTM TimeNet Model")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--all", action="store_true", help="Train all 3 coins")
    
    args = parser.parse_args()
    
    trainer = AITrainer()
    
    if args.all:
        symbols = ["BTC/USDT", "ETH/USDT", "LTC/USDT"]
        for symbol in symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧠 Training LSTM for {symbol}...")
            logger.info(f"{'='*50}")
            success = await trainer.train_model_for_symbol(symbol)
            if success:
                logger.info(f"✅ {symbol} LSTM training complete!")
            else:
                logger.error(f"❌ {symbol} training failed!")
    else:
        logger.info(f"🧠 Training LSTM for {args.symbol}...")
        await trainer.train_model_for_symbol(args.symbol)
    
    logger.info("\n🎉 All LSTM training complete!")


if __name__ == "__main__":
    asyncio.run(main())
