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

# Kendi modüllerimiz
from src.brain.feature_engineering import FeatureEngineer
from src.brain.regime_classifier import RegimeClassifier
from src.validation.validator import SignalValidator
from src.data_ingestion.macro_connector import MacroConnector

logger = logging.getLogger("MARKET_ANALYZER_PRO")

class MarketAnalyzer:
    """
    DEMIR AI V16.0 - LOGIC FUSION ENGINE
    
    Bu motor, sadece bir tahminci değil, bir "Stratejist"tir.
    LSTM (Yön) + Macro (Ortam) + Pivot (Seviye) verilerini birleştirerek
    sana 'Nokta Atışı' fırsatlar sunar.
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
        # 1. VERİ HAZIRLIĞI (ENGINEERING)
        crypto_df = FeatureEngineer.process_data(raw_data)
        if crypto_df is None or len(crypto_df) < self.LOOKBACK + 5: return None

        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        last_row = df.iloc[-1]
        
        # 2. PİYASA ORTAMI (REGIME & MACRO FILTER)
        current_regime = self.regime_classifier.identify_regime(df)
        
        # --- MAKRO FİLTRELER (YENİ) ---
        # Eğer Dolar (DXY) veya Korku (VIX) çok yüksekse, bot "AL" vermemeli.
        dxy = float(last_row.get('macro_DXY', 0))
        vix = float(last_row.get('macro_VIX', 0))
        
        # Veri 0 ise geçmişten tamamla (Patching)
        if dxy == 0 and 'macro_DXY' in df.columns: dxy = df['macro_DXY'].ffill().iloc[-1]
        if vix == 0 and 'macro_VIX' in df.columns: vix = df['macro_VIX'].ffill().iloc[-1]

        macro_risk_high = False
        macro_reason = ""
        if dxy > 106.5: 
            macro_risk_high = True; macro_reason = "DXY Extreme High"
        if vix > 30: 
            macro_risk_high = True; macro_reason = "VIX Panic Mode"

        # 3. YAPAY ZEKA TAHMİNİ (LSTM + RL)
        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        reason = "Waiting for setup..."
        
        lstm_prob = 0.5
        
        if self.load_lstm_for_symbol(symbol):
            try:
                # LSTM Tahmini
                drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 
                             'macro_GOLD', 'macro_SILVER', 'macro_OIL', 'corr_spx', 'corr_dxy']
                feat_cols = [c for c in df.columns if c not in drop_cols]
                
                recent = df[feat_cols].tail(self.LOOKBACK).values
                scaled = self.scalers[symbol].transform(recent)
                lstm_prob = self.lstm_models[symbol].predict(np.array([scaled]), verbose=0)[0][0]
            except: pass

        # 4. KARAR MEKANİZMASI (LOGIC FUSION)
        
        # Eğer Makro Risk varsa, sadece SHORT (Satış) yönlü sinyallere izin ver veya bekle
        if macro_risk_high:
            if lstm_prob < 0.40: # Düşüş sinyali
                ai_decision = "SELL"
                ai_confidence = (1 - lstm_prob) * 100
                reason = f"Macro Risk ({macro_reason}) + LSTM Bearish"
            else:
                ai_decision = "NEUTRAL"
                reason = f"Trade Blocked: {macro_reason}"
        
        else:
            # Normal Piyasa Koşulları
            if lstm_prob > 0.60: # %60 Eminse
                # Trend Kontrolü (Feature Engineering'den gelen 4H Trend)
                if last_row.get('trend_4h', 0) == -1:
                     ai_decision = "NEUTRAL"
                     reason = "LSTM Bullish but 4H Trend is Bearish"
                else:
                    ai_decision = "BUY"
                    ai_confidence = lstm_prob * 100
                    reason = "Strong LSTM Signal + Trend Aligned"
            
            elif lstm_prob < 0.40:
                if last_row.get('trend_4h', 0) == 1:
                     ai_decision = "NEUTRAL"
                     reason = "LSTM Bearish but 4H Trend is Bullish"
                else:
                    ai_decision = "SELL"
                    ai_confidence = (1 - lstm_prob) * 100
                    reason = "Strong LSTM Signal + Trend Aligned"

        # 5. HEDEF BELİRLEME (PIVOT POINTS)
        # ATR yerine Pivot seviyelerini kullanıyoruz (Daha profesyonel)
        price = float(last_row['close'])
        
        # Pivotlar (Feature Engineering'den geliyor: r1, r2, s1, s2)
        r1 = float(last_row.get('r1', price * 1.02))
        s1 = float(last_row.get('s1', price * 0.98))
        
        if ai_decision == "BUY":
            tp = r1 # İlk direnç hedef
            sl = s1 # İlk destek altı stop
            # Eğer hedef çok yakınsa (Risk/Reward kötü), işlemi iptal et
            if (tp - price) < (price - sl):
                ai_decision = "NEUTRAL"
                reason = "Bad Risk/Reward Ratio (Target too close)"
                
        elif ai_decision == "SELL":
            tp = s1
            sl = r1
            if (price - tp) < (sl - price):
                ai_decision = "NEUTRAL"
                reason = "Bad Risk/Reward Ratio (Target too close)"
        else:
            tp, sl = 0.0, 0.0

        # 6. DASHBOARD VERİSİ (PATCHED)
        # ... (Eksik verileri doldurma) ...
        snapshot = self._create_snapshot(symbol, last_row, ai_decision, ai_confidence, current_regime, lstm_prob, df)
        self._save_to_dashboard(snapshot)

        if ai_decision == "NEUTRAL": return None
        
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

    def _create_snapshot(self, symbol, row, decision, conf, regime, prob, df):
        """Dashboard için temiz veri paketi hazırlar."""
        macro_vars = ['macro_DXY', 'macro_VIX', 'macro_SPX', 'macro_NDQ', 'macro_TNX', 'macro_GOLD', 'macro_SILVER', 'macro_OIL']
        vals = {}
        for k in macro_vars:
            v = float(row.get(k, 0))
            if v == 0 and k in df.columns: v = float(df[k].ffill().iloc[-1])
            vals[k] = v
            
        return {
            "symbol": symbol,
            "price": float(row['close']),
            "dxy": vals['macro_DXY'], "vix": vals['macro_VIX'], "spx": vals['macro_SPX'],
            "gold": vals['macro_GOLD'], "silver": vals['macro_SILVER'], "oil": vals['macro_OIL'],
            "corr_spx": float(row.get('corr_spx', 0)),
            "ai_decision": decision, "ai_confidence": conf,
            "regime": regime, "rsi": float(row['rsi']),
            "trend": "UP" if prob > 0.5 else "DOWN",
            "timestamp": pd.Timestamp.now().isoformat()
        }

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
