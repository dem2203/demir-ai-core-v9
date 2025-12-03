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

logger = logging.getLogger("MARKET_ANALYZER_HYBRID")

class MarketAnalyzer:
    """
    DEMIR AI V15.1 - COMMODITY & CORRELATION AWARE
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
        self.regime_classifier = RegimeClassifier()
        self.load_rl_agent()

    def _get_lstm_paths(self, symbol):
        clean_sym = symbol.replace("/", "")
        model_path = os.path.join(self.LSTM_DIR, f"lstm_v11_{clean_sym}.h5")
        scaler_path = os.path.join(self.LSTM_DIR, f"scaler_{clean_sym}.pkl")
        return model_path, scaler_path

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
            try:
                self.rl_agent = PPO.load(self.RL_MODEL_PATH)
                logger.info("🤖 RL AGENT LOADED.")
            except: pass

    async def analyze_market(self, symbol: str, raw_data: List[Dict]) -> Optional[Dict]:
        crypto_df = FeatureEngineer.process_data(raw_data)
        if crypto_df is None or len(crypto_df) < self.LOOKBACK + 5: return None

        # Makro Veri (Genişletilmiş)
        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        last_row = df.iloc[-1]
        current_regime = self.regime_classifier.identify_regime(df)
        regime_settings = self.regime_classifier.get_risk_adjustment(current_regime)
        
        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        reason = f"Regime: {current_regime}"

        # Tahmin
        lstm_prediction = 0.5
        rl_action = 0 
        
        if self.load_lstm_for_symbol(symbol):
            try:
                model = self.lstm_models[symbol]
                scaler = self.scalers[symbol]
                
                # Eğitimde OLMAYAN yeni sütunları (Gold, Silver, Oil, Corr) çıkar
                # Model eski, veri yeni olduğu için shape hatasını önlüyoruz.
                # (Not: Bu yeni verileri kullanmak için modeli RE-TRAIN yapmalıyız, şimdilik bypass ediyoruz)
                drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 
                             'macro_GOLD', 'macro_SILVER', 'macro_OIL', 'corr_spx', 'corr_dxy']
                             
                feature_cols = [c for c in df.columns if c not in drop_cols]
                
                # Shape kontrolü: Eğer modelin beklediği input boyutu ile feature sayısı tutmazsa hata verir.
                # Bu yüzden try-except bloğu hayati.
                recent_data = df[feature_cols].tail(self.LOOKBACK).values
                scaled_input = scaler.transform(recent_data)
                lstm_prediction = model.predict(np.array([scaled_input]), verbose=0)[0][0]
            except Exception as e:
                # Sütun uyuşmazlığı olursa yedek hesaplama veya pass
                pass

        if self.rl_agent:
            try:
                # RL için de aynı temizlik
                drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 
                             'macro_GOLD', 'macro_SILVER', 'macro_OIL', 'corr_spx', 'corr_dxy']
                obs_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
                obs = obs_df.iloc[-1].values.astype(np.float32)
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                rl_action = int(action)
            except: pass

        # Karar
        if not regime_settings['trade_allowed']:
             ai_decision = "NEUTRAL"
             reason = f"Blocked ({current_regime})"
        else:
            if rl_action == 1 and lstm_prediction > 0.50:
                ai_decision = "BUY"
                ai_confidence = lstm_prediction * 100
                reason = "Hybrid Signal (RL+LSTM)"
            elif rl_action == 2 and lstm_prediction < 0.50:
                ai_decision = "SELL"
                ai_confidence = (1 - lstm_prediction) * 100
                reason = "Hybrid Signal (RL+LSTM Bearish)"
            elif lstm_prediction > 0.60:
                 ai_decision = "BUY"
                 reason = "Strong LSTM Trend"

        # --- DASHBOARD İÇİN VERİLER (YAMALI) ---
        # Tüm makro verileri güvenli al
        macro_keys = ['macro_DXY', 'macro_VIX', 'macro_SPX', 'macro_NDQ', 'macro_TNX', 'macro_GOLD', 'macro_SILVER', 'macro_OIL']
        d_vals = {}
        for k in macro_keys:
            val = float(last_row.get(k, 0))
            if val == 0 and k in df.columns:
                val = float(df[k].replace(0, np.nan).ffill().iloc[-1])
            d_vals[k] = val

        corr_spx = float(last_row.get('corr_spx', 0))

        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "dxy": d_vals['macro_DXY'],
            "vix": d_vals['macro_VIX'],
            "spx": d_vals['macro_SPX'],
            "ndq": d_vals['macro_NDQ'],
            "tnx": d_vals['macro_TNX'],
            "gold": d_vals['macro_GOLD'],
            "silver": d_vals['macro_SILVER'],
            "oil": d_vals['macro_OIL'],
            "corr_spx": corr_spx,
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "regime": current_regime,
            "rsi": float(last_row['rsi']),
            "trend": "UP" if lstm_prediction > 0.5 else "DOWN",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self._save_to_dashboard(snapshot)

        if ai_decision == "NEUTRAL": return None

        price = float(last_row['close'])
        atr = float(last_row['atr'])
        stop_multiplier = regime_settings['stop_loss_multiplier']
        
        signal = {
            "symbol": symbol,
            "side": ai_decision,
            "entry_price": price,
            "tp_price": price + (atr * 3) if ai_decision == "BUY" else price - (atr * 3),
            "sl_price": price - (atr * stop_multiplier) if ai_decision == "BUY" else price + (atr * stop_multiplier),
            "confidence": ai_confidence,
            "reason": reason,
            "regime": current_regime
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
