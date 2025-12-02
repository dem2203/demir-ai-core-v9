import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from src.brain.feature_engineering import FeatureEngineer
from src.data_ingestion.connectors.binance_connector import BinanceConnector

logger = logging.getLogger("AI_TRAINER")

class AITrainer:
    """
    GERÇEK YAPAY ZEKA EĞİTMENİ
    Binance'den geçmiş veriyi çeker, Random Forest modelini eğitir ve kaydeder.
    """
    
    MODEL_PATH = "src/brain/models/storage/rf_model_v9.pkl"
    
    def __init__(self):
        self.connector = BinanceConnector()
        self.model = RandomForestClassifier(
            n_estimators=200,    # 200 karar ağacı
            max_depth=10,        # Aşırı öğrenmeyi (overfitting) engellemek için sınır
            random_state=42,
            n_jobs=-1            # Tüm işlemci gücünü kullan
        )

    async def fetch_training_data(self, symbol="BTC/USDT", limit=1000):
        """Eğitim için geçmiş veriyi çeker."""
        logger.info(f"Fetching {limit} candles for training...")
        raw_data = await self.connector.fetch_candles(symbol, limit=limit)
        await self.connector.close()
        
        if not raw_data:
            return None
            
        # İndikatörleri hesapla
        df = FeatureEngineer.process_data(raw_data)
        return df

    def prepare_targets(self, df: pd.DataFrame):
        """
        AI'a hedefi öğretir:
        Gelecek 3 mum içinde fiyat %0.5 artacak mı? (1: Evet, 0: Hayır)
        """
        future_close = df['close'].shift(-3) 
        df['target'] = (future_close > df['close'] * 1.005).astype(int)
        df.dropna(inplace=True)
        return df

    async def train_new_model(self):
        """Modeli eğitir ve diske kaydeder."""
        logger.info("Starting AI Training Session...")
        
        # 1. Veri Çek ve Hazırla
        df = await self.fetch_training_data(limit=1000)
        if df is None:
            logger.error("Training aborted due to data error.")
            return
            
        df = self.prepare_targets(df)
        
        # Hedef (target) ve gereksiz sütunları çıkar, geriye sadece matematik kalsın
        features = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        X = df[features]
        y = df['target']
        
        # 2. Eğit
        logger.info(f"Training Random Forest Model with {len(X)} samples...")
        self.model.fit(X, y) # Tüm veriyi kullan
        
        # 3. Kaydet
        if not os.path.exists("src/brain/models/storage"):
            os.makedirs("src/brain/models/storage")
            
        joblib.dump(self.model, self.MODEL_PATH)
        logger.info(f"✅ BRAIN SAVED at {self.MODEL_PATH}")