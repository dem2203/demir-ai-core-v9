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
from src.brain.state_builder import StateVectorBuilder  # PHASE 7: True AI

# PHASE 8: AI Superpowers
from src.brain.onchain_intel import OnChainIntelligence
from src.brain.liquidation_hunter import LiquidationHunter
from src.brain.pattern_engine import PatternRecognition
from src.brain.adaptive_intel import AdaptiveIntelligence
from src.brain.technical_analyzer import TechnicalAnalyzer  # PHASE 9: Advanced TA

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
        
        # PHASE 8: AI Superpowers
        self.onchain_intel = OnChainIntelligence()
        self.liquidation_hunter = LiquidationHunter()
        self.pattern_engine = PatternRecognition()
        self.adaptive_intel = AdaptiveIntelligence()
        
        # PHASE 9: Advanced Technical Analysis
        self.technical_analyzer = TechnicalAnalyzer()
        
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
            logger.warning("⚠️ Macro data unavailable. Using crypto-only data.")
            df = df_1h.copy()
            # Add dummy macro columns to prevent errors
            for col in ['macro_DXY', 'macro_VIX', 'macro_SPX', 'macro_NDQ', 'macro_TNX', 'macro_GOLD', 'macro_SILVER', 'macro_OIL']:
                df[col] = 0.0
            df['macro_DXY'] = 100.0  # Default DXY
            df['macro_VIX'] = 20.0   # Default VIX
        else:
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
        if macro_df is not None and not macro_df.empty:
            for col in ['macro_SPX', 'macro_NDQ', 'macro_DXY']:
                if col in df.columns:
                    corr_data[col.replace('macro_', '')] = df[[col]].rename(columns={col: 'close'})
        
        corr_matrix = self.correlation_engine.calculate_correlation_matrix(corr_data)
        corr_risk = self.correlation_engine.check_signal_correlation_risk(
            corr_matrix, 
            symbol.replace('/', ''),
            ['SPX', 'NDQ', 'DXY']
        )

        # --- PHASE 8: AI SUPERPOWERS ---
        clean_symbol = symbol.replace('/', '')
        
        # 8.1 ON-CHAIN INTELLIGENCE (Whale tracking, exchange flow)
        try:
            onchain_data = await self.onchain_intel.get_full_onchain_analysis(clean_symbol)
            onchain_signal = onchain_data.get('signal', 'NEUTRAL')
            onchain_score = onchain_data.get('composite_score', 0)
            logger.info(f"🐋 On-Chain: {onchain_signal} (Score: {onchain_score})")
        except Exception as e:
            logger.warning(f"On-chain analysis failed: {e}")
            onchain_data = {}
            onchain_signal = 'NEUTRAL'
            onchain_score = 0
        
        # 8.2 LIQUIDATION HUNTER (Liq levels, funding, L/S ratio)
        try:
            liq_data = await self.liquidation_hunter.get_full_liquidation_analysis(clean_symbol)
            liq_signal = liq_data.get('signal', 'NEUTRAL')
            liq_score = liq_data.get('composite_score', 0)
            magnet_price = liq_data.get('liquidation_levels', {}).get('magnet_price', 0)
            logger.info(f"🎯 Liquidation: {liq_signal} (Score: {liq_score}) | Magnet: ${magnet_price:,.0f}")
        except Exception as e:
            logger.warning(f"Liquidation analysis failed: {e}")
            liq_data = {}
            liq_signal = 'NEUTRAL'
            liq_score = 0
            magnet_price = 0
        
        # 8.3 PATTERN RECOGNITION (Wyckoff, SMC, Structure)
        try:
            pattern_data = self.pattern_engine.get_full_pattern_analysis(df)
            wyckoff_phase = pattern_data.get('wyckoff', {}).get('phase', 'UNKNOWN')
            pattern_bias = pattern_data.get('final_bias', 'NEUTRAL')
            structure = pattern_data.get('market_structure', {}).get('structure', 'UNKNOWN')
            logger.info(f"📊 Pattern: Wyckoff={wyckoff_phase} | Bias={pattern_bias} | Structure={structure}")
        except Exception as e:
            logger.warning(f"Pattern analysis failed: {e}")
            pattern_data = {}
            wyckoff_phase = 'UNKNOWN'
            pattern_bias = 'NEUTRAL'
            structure = 'UNKNOWN'
        
        # 8.4 ADAPTIVE INTELLIGENCE (Regime, calibration)
        try:
            adaptive_regime = self.adaptive_intel.detect_regime(df)
            adaptive_strategy = adaptive_regime.get('strategy', 'WAIT')
            risk_multiplier = adaptive_regime.get('risk_multiplier', 1.0)
            logger.info(f"🧠 Adaptive: Regime={adaptive_regime.get('regime')} | Strategy={adaptive_strategy}")
        except Exception as e:
            logger.warning(f"Adaptive analysis failed: {e}")
            adaptive_regime = {}
            adaptive_strategy = 'WAIT'
            risk_multiplier = 1.0

        # --- PHASE 9: ADVANCED TECHNICAL ANALYSIS ---
        try:
            tech_analysis = self.technical_analyzer.get_full_analysis(df)
            tech_bias = tech_analysis.get('technical_bias', 'NEUTRAL')
            candlestick_patterns = tech_analysis.get('candlestick_patterns', [])
            chart_patterns = tech_analysis.get('chart_patterns', [])
            divergences = tech_analysis.get('divergences', [])
            fib_data = tech_analysis.get('fibonacci', {})
            volume_analysis = tech_analysis.get('volume', {})
            pivot_data = tech_analysis.get('pivot_points', {})
            
            # En güçlü patternleri logla
            if candlestick_patterns:
                strongest = max(candlestick_patterns, key=lambda x: x.get('strength', 0))
                logger.info(f"🕯️ Candlestick: {strongest.get('pattern')} ({strongest.get('signal')})")
            if chart_patterns:
                strongest = max(chart_patterns, key=lambda x: x.get('strength', 0))
                logger.info(f"📐 Chart: {strongest.get('pattern')} ({strongest.get('signal')})")
            if divergences:
                logger.info(f"📊 Divergence detected: {divergences[0].get('type')}")
            
            logger.info(f"🎯 Technical Bias: {tech_bias} | Patterns: {tech_analysis.get('active_patterns_count', 0)}")
        except Exception as e:
            logger.warning(f"Technical analysis failed: {e}")
            tech_analysis = {}
            tech_bias = 'NEUTRAL'
            candlestick_patterns = []
            chart_patterns = []
            divergences = []
            fib_data = {}
            volume_analysis = {}
            pivot_data = {}

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
                logger.error(f"RL prediction failed: {e}, using Secondary Model")
                # Fallback to LSTM-only if RL fails
                if lstm_prob > 0.6:
                    ai_decision = "BUY"
                    ai_confidence = (lstm_prob - 0.5) * 200
                    reason = "Secondary: LSTM Bullish"
                elif lstm_prob < 0.4:
                    ai_decision = "SELL"
                    ai_confidence = (0.5 - lstm_prob) * 200
                    reason = "Secondary: LSTM Bearish"
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
        
        # ========================================
        # CONFLUENCE-BASED TRADING (3-5 trades/day)
        # Sadece Technical + Pattern AYNI YÖNÜ gösterdiğinde işlem aç
        # ========================================
        
        # Sinyal Yönlerini Hesapla
        tech_direction = 0  # -1: bearish, 0: neutral, 1: bullish
        pattern_direction = 0
        onchain_direction = 0
        
        # Technical Bias → Direction
        if tech_bias in ['STRONG_BEARISH']:
            tech_direction = -1
        elif tech_bias in ['BEARISH']:
            tech_direction = -0.5
        elif tech_bias in ['STRONG_BULLISH']:
            tech_direction = 1
        elif tech_bias in ['BULLISH']:
            tech_direction = 0.5
        
        # Pattern Bias → Direction
        if pattern_bias in ['STRONG_BEARISH']:
            pattern_direction = -1
        elif pattern_bias in ['BEARISH']:
            pattern_direction = -0.5
        elif pattern_bias in ['STRONG_BULLISH']:
            pattern_direction = 1
        elif pattern_bias in ['BULLISH']:
            pattern_direction = 0.5
        
        # On-Chain → Direction
        if onchain_signal == 'SELL':
            onchain_direction = -0.5
        elif onchain_signal == 'BUY':
            onchain_direction = 0.5
        
        # LSTM → Direction
        lstm_direction = (lstm_prob - 0.5) * 2  # -1 to 1 range

        # HTF Trend (4H) → Direction
        htf_direction = 0
        if '4h' in mtf_data:
            df_4h = mtf_data['4h'].iloc[-1]
            # 4H Trend: Fiyat VWAP'ın üstündeyse BULLISH
            if df_4h['close'] > df_4h.get('vwap', df_4h['close']):
                htf_direction = 1
            else:
                htf_direction = -1
        
        # ========================================
        # CONFLUENCE SCORE HESAPLA
        # ========================================
        
        # ========================================
        # CONFLUENCE SCORE HESAPLA (Reasoning için her zaman hesapla)
        # ========================================
        
        # Ağırlıklar (Swing Trading için - Multi-Timeframe)
        weights = {
            'tech': 0.30,      # Technical (LTF)
            'pattern': 0.25,   # Pattern
            'lstm': 0.20,      # AI Model
            'htf': 0.15,       # HTF Trend (Kartal Gözü)
            'onchain': 0.10    # On-Chain
        }
        
        confluence_score = (
            tech_direction * weights['tech'] +
            pattern_direction * weights['pattern'] +
            lstm_direction * weights['lstm'] +
            htf_direction * weights['htf'] +
            onchain_direction * weights['onchain']
        )
        
        # Uyum Skoru: Sinyallerin aynı yönü gösterme oranı
        signals = [tech_direction, pattern_direction, lstm_direction, htf_direction]
        bullish_count = sum(1 for s in signals if s > 0)
        bearish_count = sum(1 for s in signals if s < 0)
        
        total_signals = len([s for s in signals if s != 0])
        agreement_ratio = max(bullish_count, bearish_count) / total_signals if total_signals > 0 else 0
        
        # ========================================
        # AI KARAR MANTIĞI (TRUE AI vs RULE BASED)
        # ========================================
        
        ai_decision = "NEUTRAL"
        signal_quality = "WEAK"
        ai_confidence = 50.0
        reason_parts = []
        
        # 1. RL AGENT (TRUE AI) - ÖNCELİKLİ
        if self.rl_agent:
            try:
                # RL agent predicts using the unified state vector
                action, _ = self.rl_agent.predict(state_vector, deterministic=True)
                rl_action = int(action)
                
                if rl_action == 2:
                    ai_decision = "BUY"
                    ai_confidence = 70.0 + (lstm_prob - 0.5) * 60
                    reason_parts.append("🤖 RL Agent: BUY")
                elif rl_action == 0:
                    ai_decision = "SELL"
                    ai_confidence = 70.0 + (0.5 - lstm_prob) * 60
                    reason_parts.append("🤖 RL Agent: SELL")
                else:
                    ai_decision = "NEUTRAL"
                    ai_confidence = 40.0
                    reason_parts.append("🤖 RL Agent: WAIT")
                
                # RL kararını Confluence ile destekle (Confidence Boost)
                if (ai_decision == "BUY" and confluence_score > 0.2) or \
                   (ai_decision == "SELL" and confluence_score < -0.2):
                    ai_confidence += 15
                    signal_quality = "STRONG"
                    reason_parts.append("⭐ Confluence Confirmed")
                elif (ai_decision == "BUY" and confluence_score < -0.1) or \
                     (ai_decision == "SELL" and confluence_score > 0.1):
                    ai_confidence -= 20
                    signal_quality = "CONFLICTING"
                    reason_parts.append("⚠️ Confluence Disagrees")
                
                logger.info(f"🧠 TRUE AI DECISION: {ai_decision} | RL Action: {rl_action}")
                
            except Exception as e:
                logger.error(f"RL Agent failed: {e}. Falling back to Confluence Logic.")
                self.rl_agent = None # Fallback tetikle
        
        # 2. CONFLUENCE LOGIC (FALLBACK / INDICATOR MODE)
        if not self.rl_agent:
            # GÜÇLÜ SELL: Technical + Pattern ikisi de bearish
            if tech_direction < 0 and pattern_direction < 0:
                if confluence_score < -0.30:
                    ai_decision = "SELL"
                    signal_quality = "STRONG" if agreement_ratio >= 0.66 else "MODERATE"
                elif confluence_score < -0.15:
                    ai_decision = "SELL"
                    signal_quality = "MODERATE"
            
            # GÜÇLÜ BUY: Technical + Pattern ikisi de bullish
            elif tech_direction > 0 and pattern_direction > 0:
                if confluence_score > 0.30:
                    ai_decision = "BUY"
                    signal_quality = "STRONG" if agreement_ratio >= 0.66 else "MODERATE"
                elif confluence_score > 0.15:
                    ai_decision = "BUY"
                    signal_quality = "MODERATE"
            
            # ÇELİŞKİLİ
            elif tech_direction * pattern_direction < 0:
                ai_decision = "NEUTRAL"
                signal_quality = "CONFLICTING"
            
            # Confidence Calculation for Fallback
            if signal_quality == "STRONG":
                ai_confidence = 80 + abs(confluence_score) * 20
            elif signal_quality == "MODERATE":
                ai_confidence = 65 + abs(confluence_score) * 20
            elif signal_quality == "CONFLICTING":
                ai_confidence = 40
            else:
                ai_confidence = 50 + abs(confluence_score) * 30
            
            reason_parts.append("Rule-Based Fallback")

        ai_confidence = max(30, min(95, ai_confidence))
        
        # Generate detailed reason
        if current_regime != 'UNKNOWN': reason_parts.append(f"Regime: {current_regime}")
        if htf_direction != 0: reason_parts.append(f"4H Trend: {'BULL' if htf_direction > 0 else 'BEAR'}")
        if tech_bias and tech_bias != 'NEUTRAL': reason_parts.append(f"Tech: {tech_bias}")
        if pattern_bias != 'NEUTRAL': reason_parts.append(f"Pattern: {pattern_bias}")
        
        reason = " | ".join(reason_parts)
        # Log the decision
        logger.info(f"🎯 FINAL DECISION: {ai_decision} ({signal_quality}) | Conf: {ai_confidence:.1f}%")
        
        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "dxy": float(last_row.get('macro_DXY', 0)), 
            "vix": float(last_row.get('macro_VIX', 0)), 
            "funding_rate": funding_rate * 100,
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "reason": reason,
            "regime": current_regime,
            "hurst": hurst,
            "fractal_score": fractal_score,
            "whale_support": whale_support if whale_support else 0,
            "whale_resistance": whale_resistance if whale_resistance else 0,
            "orderbook_imbalance": orderbook_imbalance,
            "correlations": corr_risk.get('correlations', {}) if corr_risk else {},
            # PHASE 8: New Superpowers
            "onchain_signal": onchain_signal,
            "onchain_score": onchain_score,
            "liq_signal": liq_signal,
            "liq_score": liq_score,
            "magnet_price": magnet_price,
            "wyckoff_phase": wyckoff_phase,
            "pattern_bias": pattern_bias,
            "market_structure": structure,
            "adaptive_strategy": adaptive_strategy,
            "risk_multiplier": risk_multiplier,
            # PHASE 9: Advanced Technical Analysis
            "tech_bias": tech_bias,
            "candlestick_count": len(candlestick_patterns),
            "candlestick_latest": candlestick_patterns[-1].get('pattern') if candlestick_patterns else None,
            "chart_pattern_count": len(chart_patterns),
            "chart_pattern_latest": chart_patterns[-1].get('pattern') if chart_patterns else None,
            "divergence_count": len(divergences),
            "divergence_latest": divergences[-1].get('type') if divergences else None,
            "fib_support": fib_data.get('nearest_support', 0),
            "fib_resistance": fib_data.get('nearest_resistance', 0),
            "volume_signal": volume_analysis.get('price_volume_signal', 'N/A'),
            "pivot": pivot_data.get('pivot', 0),
            "pivot_support": pivot_data.get('nearest_support', 0),
            "pivot_resistance": pivot_data.get('nearest_resistance', 0),
            "timestamp": pd.Timestamp.now().isoformat(),
            "brain_state": {
                "tech_attention": abs(tech_direction) * weights['tech'],
                "pattern_attention": abs(pattern_direction) * weights['pattern'],
                "lstm_attention": abs(lstm_direction) * weights['lstm'],
                "htf_attention": abs(htf_direction) * weights['htf'],
                "htf_direction": htf_direction,  # -1, 0, or 1 for filter logic
                "onchain_attention": abs(onchain_direction) * weights['onchain'],
                "rl_action": int(rl_action) if self.rl_agent and 'rl_action' in locals() else -1
            }
        }
        self._save_to_dashboard(snapshot)

        if ai_decision == "NEUTRAL": return None

        # Sinyal Paketi (SMART SL/TP)
        price = float(last_row['close'])
        atr = float(last_row['atr'])
        
        # Akıllı Seviyeleri Hesapla
        smart_levels = self.risk_manager.calculate_smart_levels(
            entry_price=price,
            side=ai_decision,
            swing_low=fib_data.get('swing_low', 0),
            swing_high=fib_data.get('swing_high', 0),
            whale_support=whale_support if whale_support else 0,
            whale_resistance=whale_resistance if whale_resistance else 0,
            magnet_price=magnet_price,
            atr=atr
        )
        
        kelly_size = self.risk_manager.calculate_kelly_size(ai_confidence)
        
        signal = {
            "symbol": symbol,
            "side": ai_decision,
            "entry_price": price,
            "sl_price": smart_levels['sl'],
            "tp_price": smart_levels['tp'],
            "confidence": ai_confidence,
            "kelly_size": kelly_size,
            "reason": reason,
            "source": "RL Agent" if self.rl_agent else "Rule-Based",
            "pattern": chart_patterns[-1].get('pattern') if chart_patterns else (candlestick_patterns[-1].get('pattern') if candlestick_patterns else "None"),
            "quality": signal_quality,
            "setup_type": smart_levels['setup_type']
        }
        
        if SignalValidator.validate_outgoing_signal(signal):
            return signal, snapshot  # Return both signal and snapshot for Precision Filter
        else:
            return None, snapshot  # Still return snapshot for dashboard even if no signal

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
