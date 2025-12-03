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
    DEMIR AI V13.0 - FULL MACRO AWARENESS
    
    Artık SPX, NDQ ve TNX verilerini de izler ve kaydeder.
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
        crypto_df = FeatureEngineer.process_data(raw_data)
        if crypto_df is None or len(crypto_df) < self.LOOKBACK + 5: return None

        # Makro Veri Çek (Genişletilmiş)
        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        last_row = df.iloc[-1]
        current_regime = self.regime_classifier.identify_regime(df)
        regime_settings = self.regime_classifier.get_risk_adjustment(current_regime)
        
        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        reason = f"Market: {current_regime}"

        has_brain = self.load_model_for_symbol(symbol)
        
        if has_brain:
            try:
                model = self.models[symbol]
                scaler = self.scalers[symbol]
                
                # Eğitimdeki sütunları otomatik bulma mantığı eklenebilir ama şimdilik
                # yeni makro veriler eğitimde yoksa hata verebilir.
                # NOT: Bu yeni verileri kullanmak için MODELİN YENİDEN EĞİTİLMESİ gerekir.
                # Şimdilik sadece Dashboard'da göstermek için çekiyoruz.
                
                # Eski model uyumu için sadece eski sütunları seçelim (Hata almamak için)
                feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 'macro_SPX', 'macro_NDQ', 'macro_TNX']]
                # (İleride modeli bu yeni verilerle tekrar eğiteceğiz)
                
                recent_data = df[feature_cols].tail(self.LOOKBACK).values
                recent_scaled = scaler.transform(recent_data)
                X_input = np.array([recent_scaled])
                
                prediction = model.predict(X_input, verbose=0)[0][0]
                ai_confidence = float(prediction * 100)
                
                threshold = regime_settings['confidence_threshold']
                
                if not regime_settings['trade_allowed']:
                     ai_decision = "NEUTRAL"
                     reason = f"Blocked by Regime ({current_regime})"
                else:
                    if prediction > threshold: ai_decision = "BUY"
                    elif prediction < (1 - threshold): ai_decision = "SELL"
                    
            except Exception as e:
                logger.error(f"Prediction Error {symbol}: {e}")

        # --- DASHBOARD VERİ YAMASI (GÜNCELLENDİ) ---
        macro_vars = ['macro_DXY', 'macro_VIX', 'macro_SPX', 'macro_NDQ', 'macro_TNX']
        dashboard_vals = {}
        
        for var in macro_vars:
            val = float(last_row.get(var, 0))
            if val == 0 and var in df.columns:
                val = float(df[var].replace(0, np.nan).ffill().iloc[-1])
            dashboard_vals[var] = val

        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "dxy": dashboard_vals['macro_DXY'],
            "vix": dashboard_vals['macro_VIX'],
            "spx": dashboard_vals['macro_SPX'],
            "ndq": dashboard_vals['macro_NDQ'],
            "tnx": dashboard_vals['macro_TNX'],
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "regime": current_regime,
            "rsi": float(last_row['rsi']),
            "trend": "UP" if last_row['close'] > last_row['vwap'] else "DOWN",
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
