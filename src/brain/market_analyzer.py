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
from src.validation.validator import SignalValidator
from src.data_ingestion.macro_connector import MacroConnector # <--- Makro veriyi canlıda da çekeceğiz

logger = logging.getLogger("MARKET_ANALYZER_LSTM")

class MarketAnalyzer:
    """
    DEMIR AI V11.0 - LSTM PREDICTOR & MACRO AWARENESS
    
    Özellikler:
    1. Canlı Kripto ve Makro veriyi birleştirir.
    2. Eğitilmiş LSTM modelini kullanır.
    3. Dashboard için veriyi kaydeder.
    4. Telegram sinyali üretir.
    """
    
    MODEL_PATH = "src/brain/models/storage/lstm_v11.h5"
    SCALER_PATH = "src/brain/models/storage/scaler.pkl"
    DASHBOARD_DATA_PATH = "dashboard_data.json"
    LOOKBACK = 60 # Modelin ihtiyaç duyduğu geçmiş mum sayısı

    def __init__(self):
        self.model = None
        self.scaler = None
        self.macro = MacroConnector() # Makro veri bağlantısı
        self.load_brain()

    def load_brain(self):
        """Modeli ve Scaler'ı diskten yükler"""
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.SCALER_PATH):
            try:
                self.model = load_model(self.MODEL_PATH)
                self.scaler = joblib.load(self.SCALER_PATH)
                logger.info("🧠 LSTM BRAIN & SCALER LOADED.")
            except Exception as e:
                logger.error(f"Brain load error: {e}")
        else:
            logger.warning("🧠 Brain not found! Waiting for training...")

    async def analyze_market(self, symbol: str, raw_data: List[Dict]) -> Optional[Dict]:
        """
        Canlı veriyi analiz eder.
        """
        # 1. Kripto Verisini İşle
        crypto_df = FeatureEngineer.process_data(raw_data)
        if crypto_df is None or len(crypto_df) < self.LOOKBACK + 5:
            return None

        # 2. Canlı Makro Veriyi Çek (Hızlı olması için son 5 gün yeterli)
        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        
        # 3. Verileri Birleştir (Füzyon)
        # Eğer makro veri çekilemezse, sadece kripto verisiyle devam etmeye çalışır
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        last_row = df.iloc[-1]
        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        reason = "Market Noise"

        # --- LSTM TAHMİNİ ---
        if self.model and self.scaler:
            try:
                # Modeli eğitirken kullandığımız sütunları seç (Target vs hariç)
                feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume']]
                
                # Son 60 mumu al (Lookback)
                recent_data = df[feature_cols].tail(self.LOOKBACK).values
                
                # Veriyi Normalize Et (Eğitimdeki aynı oranlarla)
                recent_scaled = self.scaler.transform(recent_data)
                
                # Boyutlandır: (1, 60, Features) -> Model tek bir tahmin yapacak
                X_input = np.array([recent_scaled])
                
                # Tahmin Et
                prediction = self.model.predict(X_input, verbose=0)[0][0] # 0 ile 1 arası bir sayı döner
                
                logger.info(f"🧠 AI RAW OUTPUT ({symbol}): {prediction:.4f}")
                
                # Karar Mantığı
                if prediction > 0.60: # %60'tan fazla yükseliş ihtimali
                    ai_decision = "BUY"
                    ai_confidence = prediction * 100
                    reason = "Deep Learning Bullish Pattern"
                    
                elif prediction < 0.40: # %40'tan az (Yani %60 düşüş ihtimali)
                    ai_decision = "SELL"
                    ai_confidence = (1 - prediction) * 100
                    reason = "Deep Learning Bearish Pattern"

            except Exception as e:
                logger.error(f"Prediction Error: {e}")
                # Hata durumunda yedek (Fallback) strateji
                if last_row['rsi'] < 30: ai_decision = "BUY"
        else:
            # Beyin yoksa yedek strateji
            if last_row['rsi'] < 30: ai_decision = "BUY"

        # --- DASHBOARD İÇİN VERİ KAYDETME ---
        # Makro verilerin varlığını kontrol et, yoksa 0 yaz (Hata almamak için)
        dxy_val = float(last_row.get('macro_DXY', 0))
        vix_val = float(last_row.get('macro_VIX', 0))

        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "rsi": float(last_row['rsi']),
            "macd": float(last_row['macd']),
            "dxy": dxy_val,  # Dashboard'da Dolar Endeksini de görelim
            "vix": vix_val,  # Korku Endeksini de görelim
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "trend": "UP" if last_row['close'] > last_row['vwap'] else "DOWN",
            "volatility": "HIGH" if last_row['bb_width'] > 0.1 else "LOW",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self._save_to_dashboard(snapshot)

        # Eğer Nötr ise sinyal üretme
        if ai_decision == "NEUTRAL": return None

        # Sinyal Paketi Hazırla
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
        
        # Validatörden geçir ve gönder
        if SignalValidator.validate_outgoing_signal(signal):
            return signal
        return None

    def _save_to_dashboard(self, data):
        """Veriyi JSON dosyasına yazar"""
        try:
            if os.path.exists(self.DASHBOARD_DATA_PATH):
                with open(self.DASHBOARD_DATA_PATH, 'r') as f:
                    try: db = json.load(f)
                    except: db = {}
            else:
                db = {}
            
            db[data['symbol']] = data
            
            with open(self.DASHBOARD_DATA_PATH, 'w') as f:
                json.dump(db, f, indent=4)
        except Exception as e:
            pass # Dashboard hatası botu durdurmasın
