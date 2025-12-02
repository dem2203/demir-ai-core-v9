import logging
import pandas as pd
import joblib
import os
from typing import Dict, List, Optional
from src.brain.feature_engineering import FeatureEngineer
from src.validation.validator import SignalValidator

logger = logging.getLogger("MARKET_ANALYZER_AI")

class MarketAnalyzer:
    """
    DEMIR AI V10.0 - ML POWERED ENGINE
    İstatistiksel Öğrenme (Machine Learning) kullanarak tahmin yapar.
    """
    
    MODEL_PATH = "src/brain/models/storage/rf_model_v9.pkl"

    def __init__(self):
        self.model = None
        self.load_brain()

    def load_brain(self):
        """Eğitilmiş modeli yükler."""
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model = joblib.load(self.MODEL_PATH)
                logger.info("🧠 AI BRAIN LOADED SUCCESSFULLY.")
            except Exception as e:
                logger.error(f"Brain load failed: {e}")
        else:
            logger.warning("🧠 BRAIN NOT FOUND! Waiting for training...")

    async def analyze_market(self, symbol: str, raw_data: List[Dict]) -> Optional[Dict]:
        df = FeatureEngineer.process_data(raw_data)
        if df is None or df.empty: return None

        current_data = df.iloc[[-1]] # Son satır
        
        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        
        # --- YAPAY ZEKA TAHMİNİ ---
        if self.model:
            features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            try:
                # Modelden tahmin iste
                prediction = self.model.predict(current_data[features])[0]
                proba = self.model.predict_proba(current_data[features])[0] # Olasılık [Düşüş%, Yükseliş%]
                
                if prediction == 1 and proba[1] > 0.60: # %60'tan fazla eminse AL
                    ai_decision = "BUY"
                    ai_confidence = proba[1] * 100
                elif prediction == 0 and proba[0] > 0.60: # %60'tan fazla eminse SAT
                    ai_decision = "SELL"
                    ai_confidence = proba[0] * 100
                    
                logger.info(f"AI PREDICTION ({symbol}): {ai_decision} (Prob: {ai_confidence:.2f}%)")
            except Exception as e:
                logger.error(f"AI Prediction Error: {e}")
        else:
            # Model yoksa, dosya yüklenene kadar bekle (veya basit RSI kontrolü yap)
            if df.iloc[-1]['rsi'] < 30: return {"symbol": symbol, "side": "BUY", "entry_price": df.iloc[-1]['close'], "tp_price": 0, "sl_price": 0, "confidence": 50}
            return None

        if ai_decision == "NEUTRAL":
            return None

        # Sinyal Paketi
        price = current_data['close'].values[0]
        atr = current_data['atr'].values[0]
        
        tp = price + (atr * 3) if ai_decision == "BUY" else price - (atr * 3)
        sl = price - (atr * 1.5) if ai_decision == "BUY" else price + (atr * 1.5)

        signal = {
            "symbol": symbol,
            "side": ai_decision,
            "entry_price": price,
            "tp_price": tp,
            "sl_price": sl,
            "confidence": ai_confidence
        }
        
        if SignalValidator.validate_outgoing_signal(signal):
            return signal
        return None