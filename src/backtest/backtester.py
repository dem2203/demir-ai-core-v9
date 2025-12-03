import pandas as pd
import numpy as np
import logging
import joblib
import asyncio
import os
from datetime import datetime # Tarih çevirmek için
from tensorflow.keras.models import load_model

# Kendi modüllerimiz
from src.brain.feature_engineering import FeatureEngineer
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("BACKTESTER")

class Backtester:
    """
    DEMIR AI - TIME MACHINE (OPTIMIZED EDITION)
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

    def _get_paths(self, symbol):
        clean_sym = symbol.replace("/", "")
        model_path = os.path.join("src/brain/models/storage", f"lstm_v11_{clean_sym}.h5")
        scaler_path = os.path.join("src/brain/models/storage", f"scaler_{clean_sym}.pkl")
        return model_path, scaler_path

    def load_brain_for_symbol(self, symbol):
        m_path, s_path = self._get_paths(symbol)
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                self.model = load_model(m_path)
                self.scaler = joblib.load(s_path)
                return True
            except: return False
        return False

    async def run_backtest(self, symbol="BTC/USDT", days=30):
        logger.info(f"Starting Optimization Backtest for {symbol}...")
        
        if not self.load_brain_for_symbol(symbol):
            return {"error": f"Brain not found for {symbol}."}

        # 1. Veri Çek
        limit = (days * 24) + 200 
        raw_crypto = await self.crypto.fetch_candles(symbol, limit=limit)
        await self.crypto.close()
        if not raw_crypto: return {"error": "No Data"}
        
        crypto_df = FeatureEngineer.process_data(raw_crypto)
        macro_df = await self.macro.fetch_macro_data(period="2y", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        if df is None or len(df) < self.LOOKBACK: return {"error": "Insufficient Data"}

        # 2. Toplu Tahmin
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
        data_values = df[feature_cols].values
        
        try:
            scaled_data = self.scaler.transform(data_values)
        except: return {"error": "Scaler Mismatch. Retrain model."}
        
        X = []
        start_index = len(df) - (days * 24)
        if start_index < self.LOOKBACK: start_index = self.LOOKBACK
        indices = range(start_index, len(df))
        
        for i in indices:
            X.append(scaled_data[i-self.LOOKBACK:i])
        
        X = np.array(X)
        if len(X) == 0: return {"error": "No X data"}
        
        predictions = self.model.predict(X, verbose=0)
        
        # 3. TİCARET SİMÜLASYONU (STRATEJİ BURADA)
        position = None 
        entry_price = 0
        
        sim_df = df.iloc[indices].reset_index(drop=True)
        
        for i, row in sim_df.iterrows():
            if i >= len(predictions): break
            
            current_price = row['close']
            ai_score = predictions[i][0]
            
            # Unix Timestamp'i okunabilir tarihe çevir
            ts = row.get('timestamp', 0)
            readable_time = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')
            
            # --- OPTİMİZE EDİLMİŞ STRATEJİ ---
            
            # ALIM: Güven eşiğini %60'a çıkardık (Daha az ama öz işlem)
            if position is None:
                if ai_score > 0.60: 
                    position = 'LONG'
                    entry_price = current_price
                    self.trade_log.append({
                        "action": "BUY", "price": entry_price, "time": readable_time, 
                        "score": f"{ai_score:.2f}", "balance": self.balance
                    })

            # SATIM: Kar Al %5, Zarar Durdur %3 (Biraz gevşettik)
            elif position == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                
                # AI "Çöküş var" derse (<0.30) veya Stop/TP tetiklenirse sat
                if ai_score < 0.30 or pnl_pct < -0.03 or pnl_pct > 0.05:
                    position = None
                    pnl_amount = (self.balance * pnl_pct) - (self.balance * 0.001) # %0.1 komisyon
                    self.balance += pnl_amount
                    
                    self.trade_log.append({
                        "action": "SELL", "price": current_price, "time": readable_time, 
                        "score": f"{ai_score:.2f}", "pnl_pct": f"{pnl_pct*100:.2f}%", 
                        "balance": self.balance
                    })

        # Rapor
        sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
        total = len(sell_trades)
        wins = len([t for t in sell_trades if float(t['pnl_pct'].strip('%')) > 0])
        win_rate = (wins / total * 100) if total > 0 else 0
        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "roi": roi,
            "total_trades": total,
            "win_rate": win_rate,
            "trades": self.trade_log
        }
