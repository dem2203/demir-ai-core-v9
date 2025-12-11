import pandas as pd
import numpy as np
import os
import logging
import joblib
import asyncio
from sklearn.preprocessing import MinMaxScaler
from src.brain.models.transformer import TimeNet

# ... (Imports remain similar)

    def build_transformer_model(self, input_shape):
        """
        Builds the Phase X TimeNet (Transformer) Model.
        """
        timenet = TimeNet(input_shape=input_shape)
        return timenet.build_model()
    
    # ... inside _train_sync ...
        
        # 4. Eğitim (Phase X: TimeNet)
        model = self.build_transformer_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0) 


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

        logger.info("⏳ Offloading LSTM training to background thread...")
        await asyncio.to_thread(self._train_sync, df, model_path, scaler_path)
        
        return True

    def _train_sync(self, df, model_path, scaler_path):
        """Blocking training logic to be run in a separate thread."""
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
        model.fit(X, y, epochs=5, batch_size=32, verbose=0) 

        # 5. Kayıt
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"✅ BRAIN SAVED (Background): -> {model_path}")
