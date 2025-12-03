import logging
import pandas as pd
import numpy as np
import joblib
import os
import json
import asyncio
from typing import Dict, List, Optional, Tuple
from tensorflow.keras.models import load_model
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

# Kendi modüllerimiz
from src.brain.feature_engineering import FeatureEngineer
from src.brain.regime_classifier import RegimeClassifier
from src.validation.validator import SignalValidator
from src.data_ingestion.macro_connector import MacroConnector
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.data_ingestion.connectors.bybit_connector import BybitConnector
from src.data_ingestion.connectors.coinbase_connector import CoinbaseConnector
from src.data_ingestion.orderbook_analyzer import OrderBookAnalyzer
from src.brain.correlation_engine import CorrelationEngine

logger = logging.getLogger("MARKET_ANALYZER_PRO")

class MarketAnalyzer:
    """
    DEMIR AI V20.0 - INSTITUTIONAL STRATEGIST
    
    Yenilikler (Faz 4A):
    1. Order Book Depth Analysis: Balina duvarlarını tespit eder.
    2. Correlation Matrix: Varlıklar arası korelasyon riski kontrolü.
    """
    
    LSTM_DIR = "src/brain/models/storage"
    RL_MODEL_PATH = "src/brain/models/storage/rl_agent_v2_recurrent"
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
        
        # PHASE 4A: Market Intelligence
        self.orderbook_analyzer = OrderBookAnalyzer()
        self.correlation_engine = CorrelationEngine()
        
        # PHASE 7: True AI Decision Engine
        self.state_builder = StateVectorBuilder()
        
        self.regime_classifier = RegimeClassifier()
        
        # Başlangıç Yüklemeleri
        self.load_rl_agent()
    
    def get_lstm_prediction(self, symbol: str, df: pd.DataFrame) -> float:
        """
        Helper method to get LSTM prediction probability.
        Returns 0.5 if model not available.
        """
        lstm_prob = 0.5
        if self.load_lstm_for_symbol(symbol):
            try:
                drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 
                             'macro_GOLD', 'macro_SILVER', 'macro_OIL', 'corr_spx', 'corr_dxy', 'vol_anomaly',
                             'macro_SPX', 'macro_NDQ', 'macro_TNX', 'macro_DXY', 'macro_VIX',
                             'hurst', 'vah', 'val', 'poc', 'atr_upper', 'atr_lower']
                
                feat_cols = [c for c in df.columns if c not in drop_cols]
                recent = df[feat_cols].tail(self.LOOKBACK).values
                scaled = self.scalers[symbol].transform(recent)
                lstm_prob = self.lstm_models[symbol].predict(np.array([scaled]), verbose=0)[0][0]
            except Exception as e:
                logger.debug(f"LSTM prediction failed: {e}")
        return lstm_prob

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
            try: 
                self.rl_agent = RecurrentPPO.load(self.RL_MODEL_PATH)
                logger.info("Loaded RecurrentPPO Agent.")
            except: 
                try:
                    self.rl_agent = PPO.load(self.RL_MODEL_PATH)
                    logger.info("Loaded Standard PPO Agent.")
                except: pass

    async def fetch_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        15m, 1H ve 4H verilerini çeker ve işler.
        """
        timeframes = ['15m', '1h', '4h']
        data_map = {}
        
        for tf in timeframes:
            raw = await self.binance.fetch_candles(symbol, timeframe=tf, limit=200)
            if raw:
                df = FeatureEngineer.process_data(raw)
                if df is not None and not df.empty:
                    data_map[tf] = df
            else:
                logger.warning(f"Missing data for {symbol} {tf}")
        
        return data_map

    def check_fractal_confluence(self, data_map: Dict[str, pd.DataFrame]) -> Tuple[str, float]:
        """
        3 Zaman dilimini analiz eder ve ortak yönü bulur.
        Dönüş: (Yön, Güç Skoru)
        """
        if '15m' not in data_map or '1h' not in data_map or '4h' not in data_map:
            return "NEUTRAL", 0.0
            
        df_15m = data_map['15m'].iloc[-1]
        df_1h = data_map['1h'].iloc[-1]
        df_4h = data_map['4h'].iloc[-1]
        
        # Basit Trend Analizi (EMA Cross veya RSI)
        # 4H: Ana Nehir (Trend)
        trend_4h = "UP" if df_4h['close'] > df_4h['vwap'] else "DOWN"
        
        # 1H: Dalga (Sinyal)
        trend_1h = "UP" if df_1h['rsi'] > 50 else "DOWN"
        
        # 15m: Dalgalanma (Giriş)
        trend_15m = "UP" if df_15m['rsi'] > 45 else "DOWN" # Biraz daha hassas
        
        score = 0
        decision = "NEUTRAL"
        
        if trend_4h == "UP" and trend_1h == "UP" and trend_15m == "UP":
            decision = "BUY"
            score = 90.0
        elif trend_4h == "DOWN" and trend_1h == "DOWN" and trend_15m == "DOWN":
            decision = "SELL"
            score = 90.0
        elif trend_4h == trend_1h: # 4H ve 1H uyumlu, 15m bekliyor
            decision = trend_4h
            score = 60.0 # Daha düşük güven
        else:
            decision = "NEUTRAL"
            score = 0.0
            
        return decision, score

    async def analyze_market(self, symbol: str, raw_data_1h: List[Dict]) -> Optional[Dict]:
        # NOT: raw_data_1h parametresi engine'den geliyor ama biz burada kendi verimizi çekeceğiz
        # Engine'i güncellemek yerine burada override ediyoruz.
        
        # 1. MULTI-TIMEFRAME VERİ ÇEK
        mtf_data = await self.fetch_multi_timeframe_data(symbol)
        if not mtf_data or '1h' not in mtf_data:
            return None
            
        df_1h = mtf_data['1h'] # Ana analiz 1H üzerinden döner
        last_row = df_1h.iloc[-1]

        # 2. MAKRO VERİ & FÜZYON
        macro_df = await self.macro.fetch_macro_data(period="5d", interval="1h")
        if macro_df is None or macro_df.empty:
            return None

        df = FeatureEngineer.merge_crypto_and_macro(df_1h, macro_df)
        
        # 3. PİYASA REJİMİ
        current_regime = self.regime_classifier.identify_regime(df)
        regime_settings = self.regime_classifier.get_risk_adjustment(current_regime)
        
        # 4. FUTURES VERİSİ
        binance_futures = await self.binance.fetch_futures_data(symbol)
        funding_rate = binance_futures.get('funding_rate', 0)
        
        # 5. SUPERHUMAN METRİKLERİ
        hurst = float(last_row.get('hurst', 0.5))
        vah = float(last_row.get('vah', 0))
        
        # 6. FRACTAL CONFLUENCE
        fractal_decision, fractal_score = self.check_fractal_confluence(mtf_data)
        
        # 7. PHASE 4A: ORDER BOOK ANALYSIS
        orderbook_data = await self.orderbook_analyzer.analyze_orderbook(symbol, float(last_row['close']))
        whale_support = orderbook_data.get('whale_support', 0) if orderbook_data else 0
        whale_resistance = orderbook_data.get('whale_resistance', 0) if orderbook_data else 0
        orderbook_imbalance = orderbook_data.get('orderbook_imbalance', 0) if orderbook_data else 0
        
        # 8. PHASE 4A: CORRELATION RISK CHECK
        # Build data dict for correlation (BTC, ETH, SPX, etc)
        corr_data = {
            symbol.replace('/', ''): df[['close']],
        }
        if not macro_df.empty:
            for col in ['macro_SPX', 'macro_NDQ', 'macro_DXY']:
                if col in df.columns:
                    corr_data[col.replace('macro_', '')] = df[[col]].rename(columns={col: 'close'})
        
        corr_matrix = self.correlation_engine.calculate_correlation_matrix(corr_data)
        corr_risk = self.correlation_engine.check_signal_correlation_risk(
            corr_matrix, 
            symbol.replace('/', ''),
            ['SPX', 'NDQ', 'DXY']
        )

        # --- PHASE 7: TRUE AI DECISION ENGINE ---
        # NO MORE HUMAN RULES - AI DECIDES EVERYTHING!
        
        # Get LSTM prediction
        lstm_prob = self.get_lstm_prediction(symbol, df)
        
        # Build unified 42-dimension state vector
        state_vector = self.state_builder.build(
            lstm_output={
                'prediction': lstm_prob,
                'confidence': abs(lstm_prob - 0.5) * 2,
                'trend_strength': abs(lstm_prob - 0.5) * 10
            },
            fractal_data={
                '15m': 'BULLISH' if mtf_data.get('15m') is not None else 'NEUTRAL',
                '1H': 'BULLISH' if mtf_data.get('1h') is not None else 'NEUTRAL',
                '4H': 'BULLISH' if mtf_data.get('4h') is not None else 'NEUTRAL'
            },
            orderbook_data={
                'whale_support': whale_support,
                'whale_resistance': whale_resistance,
                'flow_imbalance': orderbook_imbalance,
                'bid_ask_ratio': orderbook_data.get('imbalance_ratio', 1.0) if orderbook_data else 1.0,
                'depth_score': 0.5,
                'current_price': float(last_row['close'])
            },
            correlation_data={
                'btc_spx_corr': 0.0,  # TODO: extract from corr_matrix
                'btc_dxy_corr': 0.0,
                'corr_stability': 0.5,
                'regime_shift': not corr_risk.get('safe_to_trade', True) if corr_risk else False
            },
            funding_data={
                'binance_rate': funding_rate,
                'bybit_rate': 0.0,
                'divergence': 0.0
            },
            volatility_data={
                'garch_forecast': 0.02,
                'hurst': hurst,
                'regime': current_regime,
                'atr_position': 0.5,
                'volume_profile_position': 0.5
            },
            anomaly_data={
                'is_anomaly': False,
                'volume_surge': 1.0,
                'price_change_pct': 0.0
            },
            macro_data={
                'dxy': float(last_row.get('macro_DXY', 100)),
                'vix': float(last_row.get('macro_VIX', 20))
            },
            position_data={
                'position': 0,
                'days_in_position': 0,
                'unrealized_pnl_pct': 0.0
            },
            performance_data={
                'win_rate': 0.5,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        )
        
        # AI MAKES THE FINAL DECISION (replaces 100+ lines of if-then rules!)
        ai_decision = "NEUTRAL"
        ai_confidence = 50.0
        reason = f"Regime: {current_regime}"
        
        if self.rl_agent:
            try:
                # RL agent predicts using the unified state vector
                action, _ = self.rl_agent.predict(state_vector, deterministic=True)
                rl_action = int(action)
                
                # Decode action: 0=SELL, 1=HOLD, 2=BUY
                if rl_action == 2:
                    ai_decision = "BUY"
                    ai_confidence = 70.0 + (lstm_prob - 0.5) * 60  # Boosted by LSTM
                    reason = "AI Agent: BUY Signal"
                elif rl_action == 0:
                    ai_decision = "SELL"
                    ai_confidence = 70.0 + (0.5 - lstm_prob) * 60
                    reason = "AI Agent: SELL Signal"
                else:
                    ai_decision = "NEUTRAL"
                    ai_confidence = 30.0
                    reason = "AI Agent: HOLD/WAIT"
                
                logger.info(f"🧠 AI DECISION: {ai_decision} (Confidence: {ai_confidence:.1f}%) | State dim: {state_vector.shape}")
                
            except Exception as e:
                logger.error(f"RL prediction failed: {e}, falling back to LSTM")
                # Fallback to LSTM-only if RL fails
                if lstm_prob > 0.6:
                    ai_decision = "BUY"
                    ai_confidence = (lstm_prob - 0.5) * 200
                    reason = "Fallback: LSTM Bullish"
                elif lstm_prob < 0.4:
                    ai_decision = "SELL"
                    ai_confidence = (0.5 - lstm_prob) * 200
                    reason = "Fallback: LSTM Bearish"
        else:
            # No RL agent - use LSTM as fallback
            if lstm_prob > 0.6:
                ai_decision = "BUY"
                ai_confidence = (lstm_prob - 0.5) * 200
                reason = "LSTM Only: Bullish"
            elif lstm_prob < 0.4:
                ai_decision = "SELL"
                ai_confidence = (0.5 - lstm_prob) * 200
                reason = "LSTM Only: Bearish"

        # --- DASHBOARD VERİSİ ---
        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "dxy": float(last_row.get('macro_DXY', 0)), 
            "vix": float(last_row.get('macro_VIX', 0)), 
            "funding_rate": funding_rate * 100,
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "regime": current_regime,
            "hurst": hurst,
            "fractal_score": fractal_score,
            "whale_support": whale_support if whale_support else 0,
            "whale_resistance": whale_resistance if whale_resistance else 0,
            "orderbook_imbalance": orderbook_imbalance,
            "correlations": corr_risk.get('correlations', {}) if corr_risk else {},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self._save_to_dashboard(snapshot)

        if ai_decision == "NEUTRAL": return None

        # Sinyal Paketi (ATR Bands ile Dinamik Stop)
        price = float(last_row['close'])
        atr_upper = float(last_row.get('atr_upper', price*1.05))
        atr_lower = float(last_row.get('atr_lower', price*0.95))
        
        if ai_decision == "BUY":
            tp = atr_upper
            sl = atr_lower
        else:
            tp = atr_lower
            sl = atr_upper

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
