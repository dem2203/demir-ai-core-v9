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
    
    Mevcut LSTM modelini ve Scaler'ı kullanarak geçmiş veride simülasyon yapar.
    'MarketAnalyzer'ın canlıda yaptığı işi, geçmiş veride topluca yapar.
    """
    
    MODEL_PATH = "src/brain/models/storage/lstm_v11.h5"
    SCALER_PATH = "src/brain/models/storage/scaler.pkl"
    LOOKBACK = 60 # Modelin ihtiyaç duyduğu geçmiş mum sayısı

    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto = BinanceConnector()
        self.macro = MacroConnector()
        self.model = None
        self.scaler = None
        self.trade_log = []

    def load_brain(self):
        """Eğitilmiş beyni yükler."""
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

        # 1. Veri Hazırlığı (Canlı sistemdeki logic ile birebir aynı)
        # Modelin 'Lookback' süresini hesaba katarak fazla veri çekiyoruz.
        limit = (days * 24) + 200 
        
        # Kripto Verisi
        raw_crypto = await self.crypto.fetch_candles(symbol, limit=limit)
        await self.crypto.close()
        
        if not raw_crypto: return {"error": "No crypto data fetched."}
        
        crypto_df = FeatureEngineer.process_data(raw_crypto)
        
        # Makro Veri (Stooq) - Son 2 yılın verisini alıp eşleyeceğiz
        macro_df = await self.macro.fetch_macro_data(period="2y", interval="1h")
        
        # Füzyon (Birleştirme)
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        if df is None or len(df) < self.LOOKBACK:
            return {"error": "Not enough data for simulation after processing."}

        # 2. Toplu Tahmin (Batch Prediction)
        # Tek tek döngü yerine, tüm veriyi hazırlayıp tek seferde modele soruyoruz (Hız için).
        
        # Eğitimde kullanılan sütunların aynısını seçiyoruz
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
        
        data_values = df[feature_cols].values
        
        # Normalizasyon (Canlıdaki scaler ile aynı)
        scaled_data = self.scaler.transform(data_values)
        
        X = []
        # Sadece simülasyon yapılacak kısmı al (Son 'days' kadar)
        # Ancak LSTM için her satırda geriye dönük 60 veriye ihtiyacımız var.
        start_index = len(df) - (days * 24)
        if start_index < self.LOOKBACK: start_index = self.LOOKBACK
        
        indices = range(start_index, len(df))
        
        for i in indices:
            X.append(scaled_data[i-self.LOOKBACK:i])
            
        X = np.array(X)
        
        # Yapay Zeka Tahminlerini Al (0 ile 1 arası olasılıklar)
        predictions = self.model.predict(X, verbose=0) 
        
        # 3. Ticaret Simülasyonu Döngüsü
        position = None # 'LONG' veya None
        entry_price = 0
        entry_time = ""
        
        # İlgili veri dilimi üzerinde dönüyoruz
        sim_df = df.iloc[indices].reset_index(drop=True)
        
        for i, row in sim_df.iterrows():
            current_price = row['close']
            ai_score = predictions[i][0] # Modelin güveni
            timestamp = row.get('timestamp', 'Unknown')
            
            # --- ALIM MANTIĞI ---
            if position is None:
                # Eğer AI %60'tan fazla eminse AL
                if ai_score > 0.60: 
                    position = 'LONG'
                    entry_price = current_price
                    entry_time = timestamp
                    
                    self.trade_log.append({
                        "action": "BUY",
                        "price": entry_price,
                        "time": entry_time,
                        "score": float(ai_score),
                        "balance": self.balance
                    })

            # --- SATIM MANTIĞI ---
            elif position == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                
                # ÇIKIŞ KURALLARI:
                # 1. AI fikrini değiştirdi: Skor < 0.40 (Düşüş bekliyor)
                # 2. Stop Loss: %2 Zarar
                # 3. Take Profit: %5 Kar
                
                if ai_score < 0.40 or pnl_pct < -0.02 or pnl_pct > 0.05:
                    position = None
                    # Kar/Zarar hesapla (Komisyon %0.1)
                    pnl_amount = (self.balance * pnl_pct) - (self.balance * 0.002) 
                    self.balance += pnl_amount
                    
                    self.trade_log.append({
                        "action": "SELL",
                        "price": current_price,
                        "time": timestamp,
                        "score": float(ai_score),
                        "pnl_pct": pnl_pct * 100,
                        "balance": self.balance
                    })

        # 4. İstatistiksel Rapor
        sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
        total_trades = len(sell_trades)
        winning_trades = len([t for t in sell_trades if t.get('pnl_pct', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "roi": roi,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "trades": self.trade_log
        }
