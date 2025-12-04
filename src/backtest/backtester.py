import pandas as pd
import numpy as np
import logging
import joblib
import asyncio
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from stable_baselines3 import PPO 

from src.brain.feature_engineering import FeatureEngineer
from src.brain.regime_classifier import RegimeClassifier
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("BACKTESTER")

class Backtester:
    """
    DEMIR AI - TIME MACHINE (VISUAL EDITION)
    
    Grafik çizimi için 'df' (Fiyat Verisi) nesnesini de sonuçlarla birlikte döndürür.
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
        # 1. LSTM
        m_path, s_path = self._get_lstm_paths(symbol)
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                self.model_lstm = load_model(m_path)
                self.scaler = joblib.load(s_path)
            except Exception as e:
                logger.error(f"LSTM Load Error: {e}")
                return False
        else: return False

        # 2. RL Ajanı
        if os.path.exists(self.RL_MODEL_PATH + ".zip"):
            try:
                self.agent_rl = PPO.load(self.RL_MODEL_PATH)
            except: self.agent_rl = None
        
        return True

    async def run_backtest(self, symbol="BTC/USDT", days=30, params=None):
        # Varsayılan Parametreler
        if params is None:
            params = {'sl_mul': 1.5, 'tp_mul': 3.0, 'threshold': 0.55}

        logger.info(f"Backtesting {symbol} with params: {params}")
        
        if not self.load_brains(symbol):
            return {"error": f"Brain not found for {symbol}."}

        # 1. Veri Çek
        limit = (days * 24) + 200 
        raw_crypto = await self.crypto.fetch_candles(symbol, limit=limit)
        await self.crypto.close()
        if not raw_crypto: return {"error": "No crypto data."}
        
        crypto_df = FeatureEngineer.process_data(raw_crypto)
        macro_df = await self.macro.fetch_macro_data(period="2y", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        if df is None or len(df) < self.LOOKBACK: return {"error": "Insufficient Data."}

        # 2. Batch Prediction
        drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 
                     'macro_GOLD', 'macro_SILVER', 'macro_OIL', 'corr_spx', 'corr_dxy']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        try:
            data_values = df[feature_cols].values
            # Check if scaler expects same number of features
            expected_features = self.scaler.n_features_in_
            actual_features = data_values.shape[1]
            if expected_features != actual_features:
                logger.warning(f"Scaler expects {expected_features} features, got {actual_features}. Retraining...")
                # Try to retrain scaler on-the-fly
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(data_values)
                logger.info("Scaler retrained successfully for backtest.")
            else:
                scaled_data = self.scaler.transform(data_values)
        except Exception as e:
            logger.error(f"Scaler transform error: {e}")
            # Fallback: Use new scaler
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df[feature_cols].values)
            logger.warning("Using fresh StandardScaler for backtest.")
        
        X_lstm = []
        start_index = len(df) - (days * 24)
        if start_index < self.LOOKBACK: start_index = self.LOOKBACK
        indices = range(start_index, len(df))
        
        for i in indices:
            X_lstm.append(scaled_data[i-self.LOOKBACK:i])
            
        X_lstm = np.array(X_lstm)
        if len(X_lstm) == 0: return {"error": "No X data."}
        
        lstm_predictions = self.model_lstm.predict(X_lstm, verbose=0)
        
        # RL verisi hazırlığı
        drop_cols_rl = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 
                        'macro_GOLD', 'macro_SILVER', 'macro_OIL', 'corr_spx', 'corr_dxy']
        df_rl = df.drop(columns=[c for c in drop_cols_rl if c in df.columns], errors='ignore')
        rl_values = df_rl.values.astype(np.float32)

        # 3. Simülasyon
        position = None 
        entry_price = 0
        
        sim_df = df.iloc[indices].reset_index(drop=True)
        
        for i, row in sim_df.iterrows():
            if i >= len(lstm_predictions): break
            
            current_price = row['close']
            atr = row['atr']
            lstm_score = lstm_predictions[i][0]
            
            rl_action = 0
            if self.agent_rl:
                obs = rl_values[indices[i]]
                rl_action, _ = self.agent_rl.predict(obs, deterministic=True)
            
            ts = row.get('timestamp', 0)
            try: t_str = datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M')
            except: t_str = str(ts)

            decision = "NEUTRAL"
            threshold = params['threshold']
            
            if rl_action == 1 and lstm_score > threshold: decision = "BUY"
            elif lstm_score > (threshold + 0.1): decision = "BUY"
            elif rl_action == 2 or lstm_score < (1 - threshold): decision = "SELL"

            if position is None:
                if decision == "BUY":
                    position = 'LONG'
                    entry_price = current_price
                    self.trade_log.append({
                        "action": "BUY", "price": entry_price, "time": t_str, 
                        "score": f"L:{lstm_score:.2f}|R:{rl_action}", "balance": self.balance
                    })
            
            elif position == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                stop_pct = (atr * params['sl_mul']) / entry_price
                take_pct = (atr * params['tp_mul']) / entry_price
                
                should_sell = False
                reason = ""
                
                if pnl_pct < -stop_pct: should_sell = True; reason = "SL"
                elif pnl_pct > take_pct: should_sell = True; reason = "TP"
                elif decision == "SELL": should_sell = True; reason = "Signal"
                
                if should_sell:
                    position = None
                    pnl_amount = (self.balance * pnl_pct) - (self.balance * 0.001)
                    self.balance += pnl_amount
                    self.trade_log.append({
                        "action": "SELL", "price": current_price, "time": t_str, 
                        "score": f"L:{lstm_score:.2f}|R:{rl_action}", "pnl_pct": f"{pnl_pct*100:.2f}%", 
                        "reason": reason, "balance": self.balance
                    })

        sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
        total = len(sell_trades)
        wins = len([t for t in sell_trades if float(t['pnl_pct'].strip('%')) > 0])
        win_rate = (wins / total * 100) if total > 0 else 0
        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            "roi": roi,
            "total_trades": total,
            "win_rate": win_rate,
            "final_balance": self.balance,
            "trades": self.trade_log,
            "df": sim_df # <-- YENİ: Grafik çizimi için veriyi döndür
        }
