import logging
import pandas as pd
import numpy as np
import joblib
import os
import json
import asyncio
from typing import Dict, List, Optional
from tensorflow.keras.models import load_model

# Kendi modüllerimiz
from src.brain.feature_engineering import FeatureEngineer
from src.brain.regime_classifier import RegimeClassifier
from src.validation.validator import SignalValidator
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("MARKET_ANALYZER_ADAPTIVE")

class MarketAnalyzer:
    """
    DEMIR AI V12.1 - ADAPTIVE INTELLIGENCE & DATA PATCHING
    
    Özellikler:
    1. LSTM Tahmini (Yön)
    2. Piyasa Rejimi Filtresi (Strateji Değişimi)
    3. Makro Veri Füzyonu
    4. Dashboard Veri İyileştirmesi (0.00 hatasını düzeltir)
    """
    
    MODELS_DIR = "src/brain/models/storage"
    DASHBOARD_DATA_PATH = "dashboard_data.json"
    LOOKBACK = 60 

    def __init__(self):
        self.models = {} 
        self.scalers = {}
        self.macro = MacroConnector()
        self.regime_classifier = RegimeClassifier()

    def _get_paths(self, symbol):
        clean_sym = symbol.replace("/", "")
        model_path = os.path.join(self.MODELS_DIR, f"lstm_v11_{clean_sym}.h5")
        scaler_path = os.path.join(self.MODELS_DIR, f"scaler_{clean_sym}.pkl")
        return model_path, scaler_path

    def load_model_for_symbol(self, symbol):
        if symbol in self.models: return True
        m_path, s_path = self._get_paths(symbol)
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                self.models[symbol] = load_model(m_path)
                self.scalers[symbol] = joblib.load(s_path)
                return True
            except: return False
        return False

    async def analyze_market(self, symbol: str, raw_data: List[Dict]) -> Optional[Dict]:
        # 1. Veri İşleme
        crypto_df = FeatureEngineer.process_data(raw_data)
        if crypto_df is None or len(crypto_df) < self.LOOKBACK + 5: return None

        # 2. Makro Veri & Füzyon
        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        last_row = df.iloc[-1]
        
        # --- 3. PİYASA REJİMİNİ BELİRLE ---
        current_regime = self.regime_classifier.identify_regime(df)
        regime_settings = self.regime_classifier.get_risk_adjustment(current_regime)
        
        logger.info(f"MARKET REGIME ({symbol}): {current_regime} | Allowed: {regime_settings['trade_allowed']}")
        
        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        reason = f"Market is {current_regime}"

        # --- 4. LSTM TAHMİNİ ---
        has_brain = self.load_model_for_symbol(symbol)
        
        if has_brain:
            try:
                model = self.models[symbol]
                scaler = self.scalers[symbol]
                
                feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
                recent_data = df[feature_cols].tail(self.LOOKBACK).values
                recent_scaled = scaler.transform(recent_data)
                X_input = np.array([recent_scaled])
                
                prediction = model.predict(X_input, verbose=0)[0][0]
                ai_confidence = prediction * 100
                
                # Dinamik Eşik Değerleri
                threshold = regime_settings['confidence_threshold']
                
                if not regime_settings['trade_allowed']:
                     ai_decision = "NEUTRAL"
                     reason = f"Trade Blocked by Regime ({current_regime})"
                else:
                    if prediction > threshold:
                        ai_decision = "BUY"
                        reason = f"LSTM Bullish in {current_regime}"
                    elif prediction < (1 - threshold):
                        ai_decision = "SELL"
                        reason = f"LSTM Bearish in {current_regime}"
                    else:
                        ai_decision = "NEUTRAL"
                        reason = "Low Conviction"
                        
            except Exception as e:
                logger.error(f"Prediction Error {symbol}: {e}")

        # --- 5. DASHBOARD VERİ DÜZELTME (PATCH) ---
        # Anlık veri 0 gelirse, geçmiş veriden son geçerli değeri bul.
        dxy_val = float(last_row.get('macro_DXY', 0))
        vix_val = float(last_row.get('macro_VIX', 0))
        
        if (dxy_val == 0 or pd.isna(dxy_val)) and 'macro_DXY' in df.columns:
             dxy_val = float(df['macro_DXY'].replace(0, np.nan).ffill().iloc[-1])
             
        if (vix_val == 0 or pd.isna(vix_val)) and 'macro_VIX' in df.columns:
             vix_val = float(df['macro_VIX'].replace(0, np.nan).ffill().iloc[-1])

        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "dxy": dxy_val,
            "vix": vix_val,
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "regime": current_regime,
            "rsi": float(last_row['rsi']),
            "trend": "UP" if last_row['close'] > last_row['vwap'] else "DOWN",
            "volatility": "HIGH" if last_row['bb_width'] > 0.1 else "LOW",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self._save_to_dashboard(snapshot)

        if ai_decision == "NEUTRAL": return None

        # --- 6. Dinamik Stop Loss ---
        price = float(last_row['close'])
        atr = float(last_row['atr'])
        stop_multiplier = regime_settings['stop_loss_multiplier']
        
        tp = price + (atr * 3) if ai_decision == "BUY" else price - (atr * 3)
        sl = price - (atr * stop_multiplier) if ai_decision == "BUY" else price + (atr * stop_multiplier)

        signal = {
            "symbol": symbol,
            "side": ai_decision,
            "entry_price": price,
            "tp_price": tp,
            "sl_price": sl,
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
