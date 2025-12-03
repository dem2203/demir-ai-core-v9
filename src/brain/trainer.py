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
    DEMIR AI v11.0 - DEEP LEARNING TRAINER (LSTM + MACRO DATA)
    
    Görevi:
    1. Kripto (Binance) ve Makro (Yahoo) verilerini çekmek.
    2. Bu verileri birleştirmek (Data Fusion).
    3. Verileri 0-1 arasına sıkıştırmak (Normalization).
    4. Geçmiş 60 muma bakarak gelecek mumu tahmin eden LSTM modelini eğitmek.
    """
    
    MODEL_PATH = "src/brain/models/storage/lstm_v11.h5"
    SCALER_PATH = "src/brain/models/storage/scaler.pkl"
    LOOKBACK = 60 # Yapay zeka geriye dönük kaç muma bakacak?

    def __init__(self):
        self.connector = BinanceConnector()
        self.macro = MacroConnector()

    async def fetch_integrated_data(self, symbol="BTC/USDT", limit=2000):
        """
        Kripto ve Makro veriyi çeker ve birleştirir.
        """
        logger.info(f"Fetching integrated data (Crypto + Macro) for {symbol}...")
        
        # 1. Kripto Verisini Çek
        raw_crypto = await self.connector.fetch_candles(symbol, limit=limit)
        await self.connector.close()
        
        if not raw_crypto:
            logger.error("Failed to fetch crypto data.")
            return None
            
        # Kripto verisini işle (RSI, MACD vs. hesapla)
        crypto_df = FeatureEngineer.process_data(raw_crypto)
        if crypto_df is None: return None
        
        # 2. Makro Veriyi Çek (DXY, SPX, VIX)
        # 2000 saat yaklaşık 3 ay eder, garanti olsun diye 1 yıllık çekiyoruz
        macro_df = await self.macro.fetch_macro_data(period="1y", interval="1h")
        
        # 3. VERİ FÜZYONU (BİRLEŞTİRME)
        # Bitcoin verisi ile Dolar verisini zaman damgasına göre eşleştir
        full_df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        logger.info(f"Data Fusion Complete. Final Shape: {full_df.shape}")
        return full_df

    def build_lstm_model(self, input_shape):
        """
        Profesyonel LSTM Mimarisi
        """
        model = Sequential([
            # 1. LSTM Katmanı
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2), # Ezberlemeyi önlemek için %20 nöronu kapat
            
            # 2. LSTM Katmanı
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            
            # 3. Karar Katmanı
            Dense(units=25),
            Dense(units=1, activation='sigmoid') # 0 ile 1 arası olasılık üretir (Yükseliş İhtimali)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    async def train_new_model(self):
        logger.info("Starting Deep Learning Training Session...")
        
        # 1. Veriyi Hazırla
        df = await self.fetch_integrated_data(limit=2000)
        if df is None: return

        # Hedef Belirle (Target): Gelecek mumun kapanışı > Şu anki mumun kapanışı mı?
        # 1: Yükseliş, 0: Düşüş
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # NaN değerleri temizle (Shift işlemi son satırı bozar)
        df.dropna(inplace=True)

        # Eğitime girmeyecek sütunları çıkar (Zaman, Sembol ve Hedef sütunu hariç her şey girer)
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Training features: {feature_cols}")
        
        data_values = df[feature_cols].values
        target_values = df['target'].values

        # 2. Normalizasyon (0-1 arası sıkıştırma)
        # LSTM modelleri büyük sayılarla (90.000$) çalışamaz, 0-1 arası sever.
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_values)

        # 3. Zaman Serisi Oluşturma (Sequence Creation)
        # LSTM'e veriyi tek tek değil, 60'arlı paketler halinde vermeliyiz.
        X, y = [], []
        for i in range(self.LOOKBACK, len(scaled_data)):
            X.append(scaled_data[i-self.LOOKBACK:i]) # Geçmiş 60 mum
            y.append(target_values[i]) # Hedef
        
        X, y = np.array(X), np.array(y)

        # 4. Modeli Eğit
        logger.info(f"Training LSTM Network on {X.shape} shape...")
        
        model = self.build_lstm_model((X.shape[1], X.shape[2]))
        
        # Epochs: Veriyi kaç kere döneceği (5 yeterli, çok yaparsak ezberler)
        model.fit(X, y, epochs=5, batch_size=32, verbose=1)

        # 5. Modeli ve Scaler'ı Kaydet
        if not os.path.exists("src/brain/models/storage"):
            os.makedirs("src/brain/models/storage")
            
        model.save(self.MODEL_PATH)
        joblib.dump(scaler, self.SCALER_PATH) # Scaler'ı da kaydetmeliyiz ki canlı veriyi de aynı oranda küçültelim.
        
        logger.info(f"✅ DEEP LEARNING BRAIN SAVED at {self.MODEL_PATH}")
