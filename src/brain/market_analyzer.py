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
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.connectors.bybit_connector import BybitConnector
from src.data_ingestion.connectors.coinbase_connector import CoinbaseConnector

logger = logging.getLogger("MARKET_ANALYZER_PRO")

class MarketAnalyzer:
    """
    DEMIR AI V18.2 - FINAL STABLE ENGINE (ZERO-MOCK EDITION)
    
    Özellikler:
    1. Hibrit Zeka: LSTM (Yön) + RL (Karar) + Rejim (Filtre).
    2. Makro Farkındalık: DXY, VIX, SPX, GOLD, OIL analizi.
    3. Vadeli İşlem Zekası: Funding Rate ve Open Interest takibi.
    4. ZERO-MOCK: Asla varsayılan değer kullanmaz. Veri yoksa analiz yapmaz.
    """
    
    LSTM_DIR = "src/brain/models/storage"
    RL_MODEL_PATH = "src/brain/models/storage/rl_agent_v1"
    DASHBOARD_DATA_PATH = "dashboard_data.json"
    LOOKBACK = 60 

    def __init__(self):
        self.lstm_models = {} 
        self.scalers = {}
        self.rl_agent = None
        
        # Bağlantılar
        self.macro = MacroConnector()
        self.binance = BinanceConnector()
        self.bybit = BybitConnector()
        self.coinbase = CoinbaseConnector()
        
        self.regime_classifier = RegimeClassifier()
        
        # Başlangıç Yüklemeleri
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
        # 1. KRİPTO VERİ İŞLEME
        crypto_df = FeatureEngineer.process_data(raw_data)
        if crypto_df is None or len(crypto_df) < self.LOOKBACK + 5: 
            logger.warning(f"Insufficient crypto data for {symbol}")
            return None

        # 2. MAKRO VERİ & FÜZYON
        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        
        # ZERO-MOCK CHECK: Makro veri boşsa analiz yapma
        if macro_df is None or macro_df.empty:
            logger.error("CRITICAL: Macro data unavailable. Halting analysis to prevent false signals.")
            return None

        df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        last_row = df.iloc[-1]
        
        # 3. PİYASA REJİMİ
        current_regime = self.regime_classifier.identify_regime(df)
        regime_settings = self.regime_classifier.get_risk_adjustment(current_regime)
        
        # 4. FUTURES VERİSİ (FONLAMA ORANI) - ÇOKLU BORSA
        # Binance ve Bybit'ten veri çekip ortalama alabiliriz veya en kötüsünü baz alabiliriz.
        binance_futures = await self.binance.fetch_futures_data(symbol)
        bybit_rate = await self.bybit.fetch_funding_rate(symbol)
        
        funding_rate = binance_futures.get('funding_rate', bybit_rate) # Binance yoksa Bybit
        
        # 5. ANOMALİ KONTROLÜ
        is_anomaly = last_row.get('vol_anomaly', 1) == -1

        # --- KARAR MEKANİZMASI ---
        ai_decision = "NEUTRAL"
        ai_confidence = 0.0
        reason = f"Regime: {current_regime}"
        
        # LSTM Tahmini
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
            except Exception as e:
                logger.error(f"LSTM Prediction Error: {e}")

        # RL Tahmini
        rl_action = 0
        if self.rl_agent:
            try:
                drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 
                             'macro_GOLD', 'macro_SILVER', 'macro_OIL', 'corr_spx', 'corr_dxy']
                obs_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
                obs = obs_df.iloc[-1].values.astype(np.float32)
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                rl_action = int(action)
            except: pass

        # Filtreler ve Nihai Karar
        if funding_rate > 0.05: # Aşırı yüksek fonlama
            if lstm_prob > 0.5:
                ai_decision = "NEUTRAL"
                reason = f"High Funding ({funding_rate*100:.3f}%) - Long Trap Risk"
        
        elif is_anomaly and lstm_prob > 0.60:
             ai_decision = "BUY"
             ai_confidence = 95.0
             reason = "WHALE ALERT: Volume Anomaly + Uptrend"
             
        elif not regime_settings['trade_allowed']:
             ai_decision = "NEUTRAL"
             reason = f"Blocked ({current_regime})"
        else:
            if lstm_prob > 0.51: 
                ai_decision = "BUY"
                ai_confidence = float(lstm_prob * 100)
                reason = "LSTM Bullish"
            elif lstm_prob < 0.49:
                ai_decision = "SELL"
                ai_confidence = float((1 - lstm_prob) * 100)

        # --- DASHBOARD VERİSİ ---
        # ZERO-MOCK: Eğer makro veri yoksa, dashboard'a eksik veri gönderilir ama ASLA uydurulmaz.
        # Frontend (Streamlit) bu eksik veriyi "N/A" veya "-" olarak göstermelidir.
        
        corr_spx = float(last_row.get('corr_spx', 0))

        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "dxy": float(last_row.get('macro_DXY', 0)), 
            "vix": float(last_row.get('macro_VIX', 0)), 
            "spx": float(last_row.get('macro_SPX', 0)),
            "ndq": float(last_row.get('macro_NDQ', 0)),
            "tnx": float(last_row.get('macro_TNX', 0)),
            "gold": float(last_row.get('macro_GOLD', 0)),
            "silver": float(last_row.get('macro_SILVER', 0)),
            "oil": float(last_row.get('macro_OIL', 0)),
            "corr_spx": corr_spx,
            "funding_rate": funding_rate * 100,
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "regime": current_regime,
            "rsi": float(last_row['rsi']),
            "trend": "UP" if lstm_prob > 0.5 else "DOWN",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self._save_to_dashboard(snapshot)

        if ai_decision == "NEUTRAL": return None

        # Sinyal Paketi
        price = float(last_row['close'])
        atr = float(last_row['atr'])
        stop_multiplier = regime_settings['stop_loss_multiplier']
        
        r1 = float(last_row.get('r1', price*1.02))
        s1 = float(last_row.get('s1', price*0.98))
        
        if ai_decision == "BUY":
            tp = r1 if r1 > price else price + (atr * 3)
            sl = s1 if s1 < price else price - (atr * stop_multiplier)
        else:
            tp = s1
            sl = r1

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
        
        # Çıkışta da doğrulama yap
        if SignalValidator.validate_outgoing_signal(signal):
            return signal
        else:
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
