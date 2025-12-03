import pandas as pd
import numpy as np
import logging
import joblib
import asyncio
import os
from tensorflow.keras.models import load_model
from src.brain.feature_engineering import FeatureEngineer
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("BACKTESTER")

class Backtester:
    """
    DEMIR AI - TIME MACHINE (LSTM EDITION)
    Geçmiş veriler üzerinde LSTM modelini çalıştırarak stratejiyi test eder.
    """
    
    MODEL_PATH = "src/brain/models/storage/lstm_v11.h5"
    SCALER_PATH = "src/brain/models/storage/scaler.pkl"
    LOOKBACK = 60

    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto = BinanceConnector()
        self.macro = MacroConnector()
        self.model = None
        self.scaler = None
        self.trade_log = []

    def load_brain(self):
        """Modeli ve Scaler'ı yükler."""
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.SCALER_PATH):
            try:
                self.model = load_model(self.MODEL_PATH)
                self.scaler = joblib.load(self.SCALER_PATH)
                return True
            except Exception as e:
                logger.error(f"Backtest Load Error: {e}")
                return False
        return False

    async def run_backtest(self, symbol="BTC/USDT", days=30):
        """
        Belirtilen gün sayısı kadar geriye gidip simülasyon yapar.
        """
        logger.info(f"Starting Backtest for {symbol} ({days} days)...")
        
        if not self.load_brain():
            return {"error": "Brain not found. Please wait for initial training to complete."}

        # 1. Veri Hazırlığı
        # Modelin 'Lookback' süresini de hesaba katarak biraz fazla veri çekiyoruz.
        limit = (days * 24) + 200 
        
        # Kripto Verisi
        raw_crypto = await self.crypto.fetch_candles(symbol, limit=limit)
        await self.crypto.close()
        if not raw_crypto: return {"error": "No crypto data fetched."}
        
        crypto_df = FeatureEngineer.process_data(raw_crypto)
        
        # Makro Veri (Stooq)
        macro_df = await self.macro.fetch_macro_data(period="2y", interval="1h")
        
        # Füzyon
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        if df is None or len(df) < self.LOOKBACK:
            return {"error": "Not enough data for simulation."}

        # 2. Toplu Tahmin (Batch Prediction) - Hız için
        # Tek tek döngüde tahmin yapmak yerine, tüm veriyi hazırlayıp tek seferde modele soruyoruz.
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
        
        data_values = df[feature_cols].values
        scaled_data = self.scaler.transform(data_values)
        
        X = []
        # Sadece simülasyon yapılacak kısmı al (Son 'days' kadar)
        # Ancak LSTM için geçmiş 60 veriye ihtiyacımız var, o yüzden indexleri kaydırıyoruz.
        start_index = len(df) - (days * 24)
        if start_index < self.LOOKBACK: start_index = self.LOOKBACK
        
        indices = range(start_index, len(df))
        
        for i in indices:
            X.append(scaled_data[i-self.LOOKBACK:i])
            
        X = np.array(X)
        
        # Yapay Zeka Tahminlerini Al
        predictions = self.model.predict(X, verbose=0) # [[0.5], [0.8], [0.2]...]
        
        # 3. Ticaret Simülasyonu
        position = None # 'LONG', None (Short şimdilik kapalı, sadece Spot mantığı)
        entry_price = 0
        entry_time = ""
        
        sim_df = df.iloc[indices].reset_index(drop=True)
        
        for i, row in sim_df.iterrows():
            current_price = row['close']
            ai_score = predictions[i][0] # 0 ile 1 arası
            
            # AL SİNYALİ
            if position is None:
                if ai_score > 0.65: # Yapay zeka %65'ten fazla eminse
                    position = 'LONG'
                    entry_price = current_price
                    entry_time = row.get('timestamp', 'Unknown')
                    
                    self.trade_log.append({
                        "action": "BUY",
                        "price": entry_price,
                        "time": entry_time,
                        "score": float(ai_score),
                        "balance": self.balance
                    })

            # SAT SİNYALİ (Kar Al / Zarar Durdur / AI Sat Dedi)
            elif position == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Kurallar:
                # 1. Stop Loss: %3 Zarar
                # 2. Take Profit: %6 Kar
                # 3. AI Sat Dedi: Skor < 0.40
                
                if pnl_pct < -0.03 or pnl_pct > 0.06 or ai_score < 0.40:
                    position = None
                    # Bakiyeyi güncelle (Komisyon %0.1 varsayalım)
                    pnl_amount = (self.balance * pnl_pct) - (self.balance * 0.002) 
                    self.balance += pnl_amount
                    
                    self.trade_log.append({
                        "action": "SELL",
                        "price": current_price,
                        "time": row.get('timestamp', 'Unknown'),
                        "score": float(ai_score),
                        "pnl_pct": pnl_pct * 100,
                        "balance": self.balance
                    })

        # 4. Raporlama
        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        total_trades = len([t for t in self.trade_log if t['action'] == 'SELL'])
        winning_trades = len([t for t in self.trade_log if t.get('pnl_pct', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "roi": roi,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "trades": self.trade_log
        }
