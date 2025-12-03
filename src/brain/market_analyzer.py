import logging
import pandas as pd
import numpy as np
import joblib
import os
import json
import asyncio
from typing import Dict, List, Optional
from tensorflow.keras.models import load_model

from src.brain.feature_engineering import FeatureEngineer
from src.validation.validator import SignalValidator
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("MARKET_ANALYZER_LSTM")

class MarketAnalyzer:
    """
    DEMIR AI V11.1 - MULTI-MODEL PREDICTOR
    Her coini kendi özel eğitilmiş modeliyle analiz eder.
    """
    
    MODELS_DIR = "src/brain/models/storage"
    DASHBOARD_DATA_PATH = "dashboard_data.json"
    LOOKBACK = 60

    def __init__(self):
        self.models = {}  # { 'BTC/USDT': model_obj }
        self.scalers = {} # { 'BTC/USDT': scaler_obj }
        self.macro = MacroConnector()

    def _get_paths(self, symbol):
        clean_sym = symbol.replace("/", "")
        model_path = os.path.join(self.MODELS_DIR, f"lstm_v11_{clean_sym}.h5")
        scaler_path = os.path.join(self.MODELS_DIR, f"scaler_{clean_sym}.pkl")
        return model_path, scaler_path

    def load_model_for_symbol(self, symbol):
        """İlgili coinin modelini hafızaya yükler (Eğer yoksa)."""
        if symbol in self.models: return True # Zaten yüklü
        
        m_path, s_path = self._get_paths(symbol)
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                self.models[symbol] = load_model(m_path)
                self.scalers[symbol] = joblib.load(s_path)
                logger.info(f"🧠 Loaded Brain for {symbol}")
                return True
            except:
                return False
        return False

    async def analyze_market(self, symbol: str, raw_data: List[Dict]) -> Optional[Dict]:
        # 1. Veri İşleme
        crypto_df = FeatureEngineer.process_data(raw_data)
        if crypto_df is None or len(crypto_df) < self.LOOKBACK + 5: return None

        # 2. Makro Veri
        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        last_row = df.iloc[-1]
        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        reason = "Insufficient Data"

        # --- ÇOKLU MODEL TAHMİNİ ---
        # Önce bu coin için model var mı kontrol et
        has_brain = self.load_model_for_symbol(symbol)
        
        if has_brain:
            try:
                model = self.models[symbol]
                scaler = self.scalers[symbol]
                
                # Veriyi Hazırla
                feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
                recent_data = df[feature_cols].tail(self.LOOKBACK).values
                
                # Normalize Et (Kendi Scaler'ı ile)
                recent_scaled = scaler.transform(recent_data)
                X_input = np.array([recent_scaled])
                
                # Tahmin
                prediction = model.predict(X_input, verbose=0)[0][0]
                
                logger.info(f"🧠 {symbol} AI Score: {prediction:.4f}")
                
                if prediction > 0.51:
                    ai_decision = "BUY"
                    ai_confidence = prediction * 100
                    reason = "Deep Learning Bullish Pattern"
                elif prediction < 0.40:
                    ai_decision = "SELL"
                    ai_confidence = (1 - prediction) * 100
                    reason = "Deep Learning Bearish Pattern"
            except Exception as e:
                logger.error(f"Prediction Error {symbol}: {e}")
        else:
            # Model henüz eğitilmemişse yedek strateji (RSI)
            if last_row['rsi'] < 30: ai_decision = "BUY"; reason = "Fallback RSI Strategy"

        # --- Dashboard Kayıt ---
        dxy_val = float(last_row.get('macro_DXY', 0))
        vix_val = float(last_row.get('macro_VIX', 0))

        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "rsi": float(last_row['rsi']),
            "macd": float(last_row['macd']),
            "dxy": dxy_val,
            "vix": vix_val,
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "trend": "UP" if last_row['close'] > last_row['vwap'] else "DOWN",
            "volatility": "HIGH" if last_row['bb_width'] > 0.1 else "LOW",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self._save_to_dashboard(snapshot)

        if ai_decision == "NEUTRAL": return None

        price = float(last_row['close'])
        atr = float(last_row['atr'])
        
        signal = {
            "symbol": symbol,
            "side": ai_decision,
            "entry_price": price,
            "tp_price": price + (atr * 3) if ai_decision == "BUY" else price - (atr * 3),
            "sl_price": price - (atr * 1.5) if ai_decision == "BUY" else price + (atr * 1.5),
            "confidence": ai_confidence,
            "reason": reason
        }
        
        if SignalValidator.validate_outgoing_signal(signal):
            return signal
        return None

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
