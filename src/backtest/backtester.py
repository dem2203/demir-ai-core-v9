import pandas as pd
import numpy as np
import logging
import joblib
import asyncio
import os
from tensorflow.keras.models import load_model

# Kendi modüllerimiz
from src.brain.feature_engineering import FeatureEngineer
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("BACKTESTER")

class Backtester:
    """
    DEMIR AI - TIME MACHINE (MULTI-MODEL EDITION)
    
    Seçilen coin için ÖZEL eğitilmiş LSTM modelini bulur ve
    geçmiş veriler (Kripto + Makro) üzerinde simülasyon yapar.
    """
    
    MODELS_DIR = "src/brain/models/storage"
    LOOKBACK = 60

    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto = BinanceConnector()
        self.macro = MacroConnector()
        self.model = None
        self.scaler = None
        self.trade_log = []

    def _get_paths(self, symbol):
        """Coine özel model dosya yollarını oluşturur."""
        clean_sym = symbol.replace("/", "")
        model_path = os.path.join(self.MODELS_DIR, f"lstm_v11_{clean_sym}.h5")
        scaler_path = os.path.join(self.MODELS_DIR, f"scaler_{clean_sym}.pkl")
        return model_path, scaler_path

    def load_brain_for_symbol(self, symbol):
        """İlgili coinin beynini yükler."""
        m_path, s_path = self._get_paths(symbol)
        
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                self.model = load_model(m_path)
                self.scaler = joblib.load(s_path)
                return True
            except Exception as e:
                logger.error(f"Backtest Load Error for {symbol}: {e}")
                return False
        return False

    async def run_backtest(self, symbol="BTC/USDT", days=30):
        """
        Belirtilen gün sayısı kadar geriye gidip simülasyon yapar.
        """
        logger.info(f"Starting Backtest for {symbol} ({days} days)...")
        
        # 1. Doğru Beyni Yükle
        if not self.load_brain_for_symbol(symbol):
            return {"error": f"Brain not found for {symbol}. Please wait for live training to finish."}

        # 2. Veri Hazırlığı (Kripto + Makro)
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

        # 3. Toplu Tahmin (Batch Prediction)
        # Eğitimde kullanılan sütunların aynısını seç (Target vs hariç)
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
        
        data_values = df[feature_cols].values
        
        # Normalizasyon (Kaydedilen Scaler ile)
        try:
            scaled_data = self.scaler.transform(data_values)
        except Exception as e:
            return {"error": f"Scaler mismatch: {e}. Retrain model."}
        
        X = []
        # Sadece simülasyon yapılacak kısmı al
        start_index = len(df) - (days * 24)
        if start_index < self.LOOKBACK: start_index = self.LOOKBACK
        
        indices = range(start_index, len(df))
        
        for i in indices:
            X.append(scaled_data[i-self.LOOKBACK:i])
            
        X = np.array(X)
        
        if len(X) == 0:
             return {"error": "No data points for prediction window."}

        # Yapay Zeka Tahminlerini Al
        predictions = self.model.predict(X, verbose=0) 
        
        # 4. Ticaret Simülasyonu
        position = None 
        entry_price = 0
        
        sim_df = df.iloc[indices].reset_index(drop=True)
        
        for i, row in sim_df.iterrows():
            if i >= len(predictions): break
            
            current_price = row['close']
            ai_score = predictions[i][0] # Modelin yükseliş inancı (0-1)
            timestamp = row.get('timestamp', 'Unknown')
            
            # --- ALIM MANTIĞI ---
            if position is None:
                # Eğer AI %60'tan fazla eminse AL
                if ai_score > 0.60: 
                    position = 'LONG'
                    entry_price = current_price
                    
                    self.trade_log.append({
                        "action": "BUY",
                        "price": entry_price,
                        "time": timestamp,
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

        # 5. Raporlama
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
