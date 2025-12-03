import pandas as pd
import numpy as np
import logging
import joblib
import asyncio
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from stable_baselines3 import PPO # <-- YENİ: RL Ajanı için

# Kendi modüllerimiz
from src.brain.feature_engineering import FeatureEngineer
from src.brain.regime_classifier import RegimeClassifier
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("BACKTESTER")

class Backtester:
    """
    DEMIR AI - TIME MACHINE (HYBRID EDITION)
    
    Canlı sistemdeki 'MarketAnalyzer' mantığının aynısını geçmiş veride simüle eder.
    LSTM + RL + Makro Veri + Rejim Filtresi kombinasyonunu test eder.
    """
    
    LSTM_DIR = "src/brain/models/storage"
    RL_MODEL_PATH = "src/brain/models/storage/rl_agent_v1"
    LOOKBACK = 60

    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto = BinanceConnector()
        self.macro = MacroConnector()
        self.regime_classifier = RegimeClassifier()
        
        self.model_lstm = None
        self.scaler = None
        self.agent_rl = None
        
        self.trade_log = []

    def _get_lstm_paths(self, symbol):
        clean_sym = symbol.replace("/", "")
        model_path = os.path.join(self.LSTM_DIR, f"lstm_v11_{clean_sym}.h5")
        scaler_path = os.path.join(self.LSTM_DIR, f"scaler_{clean_sym}.pkl")
        return model_path, scaler_path

    def load_brains(self, symbol):
        """Hem LSTM hem de RL beyinlerini yükler."""
        # 1. LSTM Yükle
        m_path, s_path = self._get_lstm_paths(symbol)
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                self.model_lstm = load_model(m_path)
                self.scaler = joblib.load(s_path)
            except Exception as e:
                logger.error(f"LSTM Load Error: {e}")
                return False
        else:
            return False # LSTM olmadan test yapamayız

        # 2. RL Ajanı Yükle
        if os.path.exists(self.RL_MODEL_PATH + ".zip"):
            try:
                self.agent_rl = PPO.load(self.RL_MODEL_PATH)
            except Exception as e:
                logger.warning(f"RL Agent Load Error: {e}. Running in LSTM-Only mode.")
                self.agent_rl = None
        
        return True

    async def run_backtest(self, symbol="BTC/USDT", days=30):
        logger.info(f"Starting Hybrid Backtest for {symbol} ({days} days)...")
        
        if not self.load_brains(symbol):
            return {"error": f"Brains (LSTM/Scaler) not found for {symbol}."}

        # 1. Veri Çek (Kripto + Makro)
        limit = (days * 24) + 200 
        
        raw_crypto = await self.crypto.fetch_candles(symbol, limit=limit)
        await self.crypto.close()
        if not raw_crypto: return {"error": "No crypto data."}
        
        crypto_df = FeatureEngineer.process_data(raw_crypto)
        macro_df = await self.macro.fetch_macro_data(period="2y", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        if df is None or len(df) < self.LOOKBACK:
            return {"error": "Insufficient Data."}

        # 2. Simülasyon Hazırlığı
        # LSTM için verileri hazırla
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
        data_values = df[feature_cols].values
        scaled_data = self.scaler.transform(data_values)
        
        X_lstm = []
        
        # RL için verileri hazırla (Drop non-numeric)
        drop_cols_rl = ['timestamp', 'symbol', 'target', 'open', 'high', 'low']
        df_rl = df.drop(columns=[c for c in drop_cols_rl if c in df.columns], errors='ignore')
        rl_values = df_rl.values.astype(np.float32)

        start_index = len(df) - (days * 24)
        if start_index < self.LOOKBACK: start_index = self.LOOKBACK
        
        indices = range(start_index, len(df))
        
        # Toplu LSTM Tahmini (Hız için)
        for i in indices:
            X_lstm.append(scaled_data[i-self.LOOKBACK:i])
        
        X_lstm = np.array(X_lstm)
        if len(X_lstm) == 0: return {"error": "No X data."}
        
        lstm_predictions = self.model_lstm.predict(X_lstm, verbose=0)
        
        # 3. Ticaret Döngüsü
        position = None 
        entry_price = 0
        
        sim_df = df.iloc[indices].reset_index(drop=True)
        
        for i, row in sim_df.iterrows():
            if i >= len(lstm_predictions): break
            
            current_price = row['close']
            timestamp = row.get('timestamp', 0)
            try: readable_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            except: readable_time = str(timestamp)
            
            # A. Piyasa Rejimi
            # (Simülasyonda geçmiş veriye bakarak rejim belirlemek için küçük bir dilim alabiliriz ama 
            # hız için şimdilik basitleştirilmiş mantık kullanacağız veya anlık veriyle hesaplayacağız)
            # Burada row'daki indikatörler zaten hesaplı olduğu için direkt kullanabiliriz.
            
            # B. Sinyaller
            lstm_score = lstm_predictions[i][0] # 0-1
            
            rl_action = 0
            if self.agent_rl:
                # RL anlık gözlem istiyor
                obs = rl_values[indices[i]]
                rl_action, _ = self.agent_rl.predict(obs, deterministic=True)
            
            # --- HİBRİT KARAR MANTIĞI (Canlı Botla Aynı) ---
            decision = "NEUTRAL"
            
            # ALIM: RL 'Al' diyor (1) VE LSTM 'Yükseliş' (>0.5) diyor
            if rl_action == 1 and lstm_score > 0.50:
                decision = "BUY"
            # YEDEK ALIM: RL kararsız ama LSTM çok emin (>0.60)
            elif lstm_score > 0.60:
                decision = "BUY"
                
            # SATIM: RL 'Sat' diyor (2) VEYA LSTM 'Düşüş' (<0.4) diyor
            elif rl_action == 2 or lstm_score < 0.40:
                decision = "SELL"

            # --- İŞLEM YÖNETİMİ ---
            if position is None:
                if decision == "BUY":
                    position = 'LONG'
                    entry_price = current_price
                    self.trade_log.append({
                        "action": "BUY", "price": entry_price, "time": readable_time, 
                        "score": f"L:{lstm_score:.2f}|R:{rl_action}", "balance": self.balance
                    })
            
            elif position == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Çıkış: Sinyal Sat'a döndü veya Stop/TP
                if decision == "SELL" or pnl_pct < -0.03 or pnl_pct > 0.06:
                    position = None
                    pnl_amount = (self.balance * pnl_pct) - (self.balance * 0.001)
                    self.balance += pnl_amount
                    
                    reason = "Signal" if decision == "SELL" else ("TP" if pnl_pct > 0 else "SL")
                    
                    self.trade_log.append({
                        "action": "SELL", "price": current_price, "time": readable_time, 
                        "score": f"L:{lstm_score:.2f}|R:{rl_action}", "pnl_pct": f"{pnl_pct*100:.2f}%", 
                        "reason": reason, "balance": self.balance
                    })

        # 4. Rapor
        sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
        total = len(sell_trades)
        wins = len([t for t in sell_trades if float(t['pnl_pct'].replace('%','')) > 0])
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
