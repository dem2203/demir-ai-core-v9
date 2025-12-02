import logging
import pandas as pd
import joblib
import os
import json # <-- EKLENDİ
from typing import Dict, List, Optional
from src.brain.feature_engineering import FeatureEngineer
from src.validation.validator import SignalValidator

logger = logging.getLogger("MARKET_ANALYZER_AI")

class MarketAnalyzer:
    """
    DEMIR AI V10.1 - DASHBOARD INTEGRATED
    Analiz sonuçlarını hem Telegram'a gönderir hem de Dashboard için kaydeder.
    """
    
    MODEL_PATH = "src/brain/models/storage/rf_model_v9.pkl"
    DASHBOARD_DATA_PATH = "dashboard_data.json" # <-- Dashboard'un okuyacağı dosya

    def __init__(self):
        self.model = None
        self.load_brain()

    def load_brain(self):
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model = joblib.load(self.MODEL_PATH)
            except: pass

    async def analyze_market(self, symbol: str, raw_data: List[Dict]) -> Optional[Dict]:
        df = FeatureEngineer.process_data(raw_data)
        if df is None or df.empty: return None

        current_data = df.iloc[[-1]]
        last_row = df.iloc[-1]
        
        # --- AI KARARI ---
        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        reason = "Market Noise"
        
        if self.model:
            features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            try:
                pred = self.model.predict(current_data[features])[0]
                proba = self.model.predict_proba(current_data[features])[0]
                
                if pred == 1 and proba[1] > 0.60:
                    ai_decision = "BUY"
                    ai_confidence = proba[1] * 100
                    reason = "AI Model High Conviction (Bullish)"
                elif pred == 0 and proba[0] > 0.60:
                    ai_decision = "SELL"
                    ai_confidence = proba[0] * 100
                    reason = "AI Model High Conviction (Bearish)"
            except: pass
        else:
            # Yedek Strateji (Eğitim yoksa)
            if last_row['rsi'] < 30: 
                ai_decision = "BUY"
                ai_confidence = 65.0
                reason = "RSI Oversold Strategy"

        # --- DASHBOARD İÇİN VERİ KAYDETME ---
        # Analiz edilen tüm verileri bir JSON dosyasına yazıyoruz
        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "rsi": float(last_row['rsi']),
            "macd": float(last_row['macd']),
            "adx": float(last_row['adx']),
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "trend": "UP" if last_row['close'] > last_row['vwap'] else "DOWN",
            "volatility": "HIGH" if last_row['bb_width'] > 0.1 else "LOW",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self._save_to_dashboard(snapshot)
        # ------------------------------------

        if ai_decision == "NEUTRAL": return None

        # Sinyal Paketi
        price = float(last_row['close'])
        atr = float(last_row['atr'])
        tp = price + (atr * 3) if ai_decision == "BUY" else price - (atr * 3)
        sl = price - (atr * 1.5) if ai_decision == "BUY" else price + (atr * 1.5)

        signal = {
            "symbol": symbol,
            "side": ai_decision,
            "entry_price": price,
            "tp_price": tp,
            "sl_price": sl,
            "confidence": ai_confidence,
            "reason": reason
        }
        
        if SignalValidator.validate_outgoing_signal(signal):
            return signal
        return None

    def _save_to_dashboard(self, data):
        """Basit bir JSON veritabanı gibi çalışır."""
        try:
            # Mevcut veriyi oku
            if os.path.exists(self.DASHBOARD_DATA_PATH):
                with open(self.DASHBOARD_DATA_PATH, 'r') as f:
                    try: db = json.load(f)
                    except: db = {}
            else:
                db = {}
            
            # Yeni veriyi ekle/güncelle
            db[data['symbol']] = data
            
            # Kaydet
            with open(self.DASHBOARD_DATA_PATH, 'w') as f:
                json.dump(db, f, indent=4)
        except Exception as e:
            logger.error(f"Dashboard Data Save Error: {e}")