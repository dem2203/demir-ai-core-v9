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

logger = logging.getLogger("MARKET_ANALYZER_PRO")

class MarketAnalyzer:
    """
    DEMIR AI V17.0 - OMNISCIENT ENGINE (HER ŞEYİ BİLEN)
    
    Kripto Grafiği + 8 Küresel Makro Faktör + LSTM + RL + Rejim
    hepsini tek potada eritip karar veren nihai beyin.
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
        # 1. TEMEL VERİ İŞLEME
        crypto_df = FeatureEngineer.process_data(raw_data)
        if crypto_df is None or len(crypto_df) < self.LOOKBACK + 5: return None

        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        last_row = df.iloc[-1]
        
        # 2. PİYASA REJİMİ
        current_regime = self.regime_classifier.identify_regime(df)
        regime_settings = self.regime_classifier.get_risk_adjustment(current_regime)
        
        # 3. MAKRO ANALİZ PUANLAMASI (YENİ MANTIK)
        # Tüm küresel verileri topla (Eksikleri doldur)
        macro_vars = ['macro_DXY', 'macro_VIX', 'macro_SPX', 'macro_NDQ', 'macro_TNX', 'macro_GOLD', 'macro_SILVER', 'macro_OIL']
        m = {}
        for k in macro_vars:
            val = float(last_row.get(k, 0))
            if val == 0 and k in df.columns: val = float(df[k].ffill().iloc[-1])
            m[k] = val

        # Puanlama Sistemi: Pozitif puanlar AL'ı, Negatifler SAT'ı destekler.
        macro_score = 0
        reasons = []

        # a) Dolar ve Korku (En önemli düşmanlar)
        if m['macro_DXY'] > 106: macro_score -= 3; reasons.append("DXY High")
        elif m['macro_DXY'] < 102: macro_score += 2; reasons.append("DXY Low")
        
        if m['macro_VIX'] > 25: macro_score -= 3; reasons.append("VIX Panic")
        elif m['macro_VIX'] < 15: macro_score += 1; reasons.append("VIX Calm")

        # b) Borsalar (Risk İştahı)
        # Son 24 saatlik değişime bakmak daha doğru olur ama anlık seviye kontrolü:
        # (Basitlik için SPX 4000 üstü güvenli kabulü - örnek)
        # Burada korelasyona bakıyoruz:
        corr_spx = float(last_row.get('corr_spx', 0))
        if corr_spx > 0.7: # Eğer Bitcoin borsayla birlikte hareket ediyorsa
            macro_score += 1; reasons.append("High SPX Corr")

        # c) Tahvil Faizi (TNX)
        if m['macro_TNX'] > 4.5: macro_score -= 2; reasons.append("Yields High")

        # d) Emtialar
        if m['macro_GOLD'] > 2100 and m['macro_DXY'] < 103: 
            macro_score += 2; reasons.append("Gold/Weak Dollar Rally")

        # FİLTRE KARARI
        # Skor -3'ten kötüyse piyasa çok risklidir -> İşlem Yasak
        # Skor +3'ten iyiyse rüzgar arkamızda -> İşlem Serbest
        macro_status = "NEUTRAL"
        if macro_score <= -3: macro_status = "BEARISH_MACRO"
        elif macro_score >= 3: macro_status = "BULLISH_MACRO"

        # 4. YAPAY ZEKA TAHMİNİ (LSTM)
        lstm_prob = 0.5
        if self.load_lstm_for_symbol(symbol):
            try:
                # Sadece temel 44 sütunu kullan (Eğitimle uyumlu)
                drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 
                             'macro_GOLD', 'macro_SILVER', 'macro_OIL', 'corr_spx', 'corr_dxy',
                             'macro_SPX', 'macro_NDQ', 'macro_TNX', 'macro_DXY', 'macro_VIX'] # Tüm makroları çıkar, model sadece teknik biliyor
                             
                # Not: İleride modeli makro verilerle eğitirsek bunları çıkarmayacağız.
                # Şimdilik "Hybrid Logic" (Mantık Birleştirme) yapıyoruz: Model Teknik bakar, Kod Makro bakar.
                
                feat_cols = [c for c in df.columns if c not in drop_cols]
                recent = df[feat_cols].tail(self.LOOKBACK).values
                scaled = self.scalers[symbol].transform(recent)
                lstm_prob = self.lstm_models[symbol].predict(np.array([scaled]), verbose=0)[0][0]
            except: pass

        # 5. FİNAL KARAR (FÜZYON)
        ai_decision = "NEUTRAL"
        ai_confidence = lstm_prob * 100
        final_reason = f"{current_regime} | Macro: {macro_score}"

        # Eğer Makro Ortam ÇOK KÖTÜYSE, LSTM ne derse desin ALMA.
        if macro_status == "BEARISH_MACRO":
            if lstm_prob < 0.40:
                ai_decision = "SELL"
                final_reason = f"Macro Bearish + LSTM Bearish ({reasons})"
            else:
                ai_decision = "NEUTRAL"
                final_reason = f"Blocked by Macro Risk ({reasons})"
        
        # Eğer Rejim Yasaklıysa
        elif not regime_settings['trade_allowed']:
             ai_decision = "NEUTRAL"
             final_reason = f"Blocked by Regime ({current_regime})"
             
        else:
            # Normal veya İyi Makro Ortam
            threshold = regime_settings['confidence_threshold']
            
            # Makro destekliyorsa eşiği düşür (Cesur ol)
            if macro_status == "BULLISH_MACRO": threshold -= 0.05
            
            if lstm_prob > threshold:
                ai_decision = "BUY"
                final_reason = f"Strong LSTM + Macro OK ({reasons})"
            elif lstm_prob < (1 - threshold):
                ai_decision = "SELL"
                final_reason = "LSTM Sell Signal"

        # 6. DASHBOARD KAYIT
        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "dxy": m['macro_DXY'], "vix": m['macro_VIX'], "spx": m['macro_SPX'],
            "ndq": m['macro_NDQ'], "tnx": m['macro_TNX'], "gold": m['macro_GOLD'],
            "silver": m['macro_SILVER'], "oil": m['macro_OIL'], "corr_spx": corr_spx,
            "ai_decision": ai_decision, "ai_confidence": ai_confidence,
            "regime": current_regime, "rsi": float(last_row['rsi']),
            "trend": "UP" if lstm_prob > 0.5 else "DOWN",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self._save_to_dashboard(snapshot)

        if ai_decision == "NEUTRAL": return None

        # 7. HEDEF BELİRLEME (Pivot)
        price = float(last_row['close'])
        r1 = float(last_row.get('r1', price*1.02))
        s1 = float(last_row.get('s1', price*0.98))
        
        tp = r1 if ai_decision == "BUY" else s1
        sl = s1 if ai_decision == "BUY" else r1
        
        # Güvenlik (Stop çok yakınsa ATR kullan)
        atr = float(last_row['atr'])
        if abs(price - sl) < atr * 0.5:
            sl = price - (atr * 1.5) if ai_decision == "BUY" else price + (atr * 1.5)

        signal = {
            "symbol": symbol, "side": ai_decision, "entry_price": price,
            "tp_price": tp, "sl_price": sl, "confidence": ai_confidence,
            "reason": final_reason, "regime": current_regime
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
