import pandas as pd
import numpy as np
import os
import logging
import joblib
import asyncio
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Kendi modüllerimiz
from src.brain.feature_engineering import FeatureEngineer
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("AI_TRAINER_LSTM")

class AITrainer:
    """
    DEMIR AI v11.1 - MULTI-ASSET TRAINER
    Her coin için ayrı bir LSTM modeli eğitir ve kaydeder.
    """
    
    MODELS_DIR = "src/brain/models/storage"
    LOOKBACK = 60 

    def __init__(self):
        self.connector = BinanceConnector()
        self.macro = MacroConnector()
        
        # Klasör yoksa oluştur
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR)

    def _get_paths(self, symbol):
        """Coin ismine özel dosya yolları üretir."""
        clean_sym = symbol.replace("/", "")
        model_path = os.path.join(self.MODELS_DIR, f"lstm_v11_{clean_sym}.h5")
        scaler_path = os.path.join(self.MODELS_DIR, f"scaler_{clean_sym}.pkl")
        return model_path, scaler_path

    async def fetch_integrated_data(self, symbol, limit=2000):
        """Kripto ve Makro veriyi çeker ve birleştirir."""
        logger.info(f"Fetching integrated data for {symbol}...")
        
        # 1. Kripto
        raw_crypto = await self.connector.fetch_candles(symbol, limit=limit)
        await self.connector.close()
        if not raw_crypto: return None
        
        crypto_df = FeatureEngineer.process_data(raw_crypto)
        
        # 2. Makro
        # Macro data - simplified (not used in training yet)
        macro_df = pd.DataFrame()
        
        # 3. Füzyon
        full_df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        return full_df

    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

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

        # 4. Eğitim
        model = self.build_lstm_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0) # Log kirliliği olmasın diye verbose=0

        # 5. Kayıt
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"✅ BRAIN SAVED: {symbol} -> {model_path}")
        return True
