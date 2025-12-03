import logging
import pandas as pd
import numpy as np
import joblib
import os
import json
import asyncio
from typing import Dict, List, Optional
from tensorflow.keras.models import load_model
from stable_baselines3 import PPO 

from src.brain.feature_engineering import FeatureEngineer
from src.brain.regime_classifier import RegimeClassifier
from src.validation.validator import SignalValidator
from src.data_ingestion.macro_connector import MacroConnector
from src.data_ingestion.connectors.binance_connector import BinanceConnector # Futures için

logger = logging.getLogger("MARKET_ANALYZER_PRO")

class MarketAnalyzer:
    """
    DEMIR AI V18.0 - GOD MODE PREDICTOR
    
    Yenilikler:
    1. Futures Intelligence: Funding Rate > 0.05% ise Long açma (Tuzak).
    2. Anomaly Detection: Hacim anormalliği varsa 'WHALE ALERT' ver.
    """
    
    LSTM_DIR = "src/brain/models/storage"
    RL_MODEL_PATH = "src/brain/models/storage/rl_agent_v1"
    DASHBOARD_DATA_PATH = "dashboard_data.json"
    LOOKBACK = 60 

    def __init__(self):
        self.lstm_models = {} 
        self.scalers = {}
        self.rl_agent = None
        self.macro = MacroConnector()
        self.binance = BinanceConnector() # Futures verisi için
        self.regime_classifier = RegimeClassifier()
        self.load_rl_agent()

    def _get_lstm_paths(self, symbol):
        clean_sym = symbol.replace("/", "")
        return (os.path.join(self.LSTM_DIR, f"lstm_v11_{clean_sym}.h5"),
                os.path.join(self.LSTM_DIR, f"scaler_{clean_sym}.pkl"))

    def load_lstm_for_symbol(self, symbol):
        if symbol in self.lstm_models: return True
        m_path, s_path = self._get_lstm_paths(symbol)
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                self.lstm_models[symbol] = load_model(m_path)
                self.scalers[symbol] = joblib.load(s_path)
                return True
            except: return False
        return False

    def load_rl_agent(self):
        path = self.RL_MODEL_PATH + ".zip"
        if os.path.exists(path):
            try: self.rl_agent = PPO.load(self.RL_MODEL_PATH)
            except: pass

    async def analyze_market(self, symbol: str, raw_data: List[Dict]) -> Optional[Dict]:
        crypto_df = FeatureEngineer.process_data(raw_data)
        if crypto_df is None or len(crypto_df) < self.LOOKBACK + 5: return None

        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        last_row = df.iloc[-1]
        
        current_regime = self.regime_classifier.identify_regime(df)
        regime_settings = self.regime_classifier.get_risk_adjustment(current_regime)
        
        # --- YENİ: FUTURES VERİSİ ÇEK ---
        futures_data = await self.binance.fetch_futures_data(symbol)
        funding_rate = futures_data.get('funding_rate', 0)
        open_interest = futures_data.get('open_interest', 0)
        
        # --- YENİ: ANOMALİ KONTROLÜ ---
        is_anomaly = last_row.get('vol_anomaly', 1) == -1 # -1 Anomali demek

        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        reason = f"Regime: {current_regime}"
        
        # --- TAHMİN ---
        lstm_prob = 0.5
        if self.load_lstm_for_symbol(symbol):
            try:
                drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 
                             'macro_GOLD', 'macro_SILVER', 'macro_OIL', 'corr_spx', 'corr_dxy', 'vol_anomaly',
                             'macro_SPX', 'macro_NDQ', 'macro_TNX', 'macro_DXY', 'macro_VIX']
                feat_cols = [c for c in df.columns if c not in drop_cols]
                recent = df[feat_cols].tail(self.LOOKBACK).values
                scaled = self.scalers[symbol].transform(recent)
                lstm_prob = self.lstm_models[symbol].predict(np.array([scaled]), verbose=0)[0][0]
            except: pass

        # --- KARAR FİLTRELERİ ---
        
        # 1. Funding Rate Tuzağı
        if funding_rate > 0.05: # Çok yüksek fonlama (Long Squeeze Riski)
            if lstm_prob > 0.5:
                ai_decision = "NEUTRAL"
                reason = f"High Funding ({funding_rate*100:.2f}%) - Long Trap Risk"
        
        # 2. Balina Anormalliği
        elif is_anomaly and lstm_prob > 0.60:
             ai_decision = "BUY"
             ai_confidence = 95.0 # Çok güçlü sinyal
             reason = "WHALE ALERT: Volume Anomaly + Uptrend"
             
        elif not regime_settings['trade_allowed']:
             ai_decision = "NEUTRAL"
             reason = f"Blocked ({current_regime})"
        else:
            if lstm_prob > 0.51: # Test için eşik düşük
                ai_decision = "BUY"
                ai_confidence = lstm_prob * 100
                reason = "LSTM Bullish"
            elif lstm_prob < 0.40:
                ai_decision = "SELL"
                ai_confidence = (1 - lstm_prob) * 100

        # --- DASHBOARD VERİLERİ ---
        macro_keys = ['macro_DXY', 'macro_VIX', 'macro_SPX', 'macro_NDQ', 'macro_TNX', 'macro_GOLD', 'macro_SILVER', 'macro_OIL']
        d_vals = {}
        for k in macro_keys:
            val = float(last_row.get(k, 0))
            if val == 0 and k in df.columns: val = float(df[k].ffill().iloc[-1])
            d_vals[k] = val

        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "dxy": d_vals['macro_DXY'], "vix": d_vals['macro_VIX'], "spx": d_vals['macro_SPX'],
            "gold": d_vals['macro_GOLD'], "silver": d_vals['macro_SILVER'], "oil": d_vals['macro_OIL'],
            "funding_rate": funding_rate * 100, # Yüzde
            "open_interest": open_interest,
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "regime": current_regime,
            "rsi": float(last_row['rsi']),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self._save_to_dashboard(snapshot)

        if ai_decision == "NEUTRAL": return None

        price = float(last_row['close'])
        atr = float(last_row['atr'])
        sl = price - (atr * 1.5) if ai_decision == "BUY" else price + (atr * 1.5)

        signal = {
            "symbol": symbol, "side": ai_decision, "entry_price": price,
            "tp_price": 0, "sl_price": sl, "confidence": ai_confidence,
            "reason": reason, "regime": current_regime
        }
        
        return signal if SignalValidator.validate_outgoing_signal(signal) else None

    def _save_to_dashboard(self, data):
        try:
            if os.path.exists(self.DASHBOARD_DATA_PATH):
                with open(self.DASHBOARD_DATA_PATH, 'r') as f:
                    try: db = json.load(f)
                    except: db = {}
            else: db = {}
            db[data['symbol']] = data
            with open(self.DASHBOARD_DATA_PATH, 'w') as f:
                json.dump(db, f, indent=4)
        except: pass
