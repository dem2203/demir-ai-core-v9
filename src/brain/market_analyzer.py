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
from src.data_ingestion.trade_flow_analyzer import TradeFlowAnalyzer  # Order Flow
from src.brain.correlation_engine import CorrelationEngine
from src.brain.state_builder import StateVectorBuilder  # PHASE 7: True AI

# PHASE 8: AI Superpowers
# PHASE 8: AI Superpowers
from src.brain.onchain_intel import OnChainIntelligence
from src.brain.onchain_analyzer import OnChainAnalyzer # PHASE 22: True On-Chain
from src.brain.liquidation_hunter import LiquidationHunter
from src.brain.pattern_engine import PatternRecognition
from src.brain.adaptive_intel import AdaptiveIntelligence
from src.brain.technical_analyzer import TechnicalAnalyzer  # PHASE 9: Advanced TA
from src.brain.vision_analyst import VisionAnalyst  # PHASE 12: Visual Cortex
from src.brain.sentiment_analyzer import SentimentAnalyzer  # PHASE 13: Sentiment
from src.data_ingestion.macro_connector import MacroConnector  # PHASE 17: Macro IQ
from src.brain.mtf_transformer import MultiTimeframeTransformer  # Multi-TF Analysis
from src.brain.early_warning import EarlyWarningSystem  # Proactive Alerts

# PHASE 23: Advanced Signal Features
from src.brain.smc_analyzer import SMCAnalyzer  # Smart Money Concepts
from src.brain.mtf_analyzer import MTFAnalyzer  # Multi-Timeframe Confluence
from src.brain.volume_profile import VolumeProfileAnalyzer  # Volume Profile
from src.brain.smart_sltp import SmartSLTPCalculator  # Intelligent SL/TP

# PHASE 6: Reinforcement Learning Agent (Pekiştirmeli Öğrenme Ajanı)
from src.brain.rl_agent.ppo_agent import RLAgent

# PHASE 27: Signal Quality Filter
from src.core.signal_filter import SignalQualityFilter

# PHASE 28: Medium-Term AI Improvements
from src.brain.ensemble_model import EnsembleModel
from src.core.position_sizer import PositionSizer

logger = logging.getLogger("MARKET_ANALYZER_PRO")

class MarketAnalyzer:
    """
    DEMIR AI V20.0 - INSTITUTIONAL STRATEGIST
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
        self.trade_flow = TradeFlowAnalyzer()  # Real-time whale detection
        self.correlation_engine = CorrelationEngine()
        
        # PHASE 7: True AI Decision Engine
        self.state_builder = StateVectorBuilder()
        
        # PHASE 8: AI Superpowers
        self.onchain_intel = OnChainIntelligence() # Binance Proxy
        self.onchain_analyzer = OnChainAnalyzer()  # PHASE 22: True On-Chain
        self.liquidation_hunter = LiquidationHunter()
        self.pattern_engine = PatternRecognition()
        self.adaptive_intel = AdaptiveIntelligence()
        
        # PHASE 9: Advanced Technical Analysis
        self.technical_analyzer = TechnicalAnalyzer()
        
        # PHASE 12: Visual Cortex (AI Vision)
        self.vision = VisionAnalyst()
        
        # PHASE 13: Sentiment Analysis
        self.sentiment = SentimentAnalyzer()
        
        # PHASE 23: Advanced Signal Features
        self.smc_analyzer = SMCAnalyzer()
        self.mtf_analyzer = MTFAnalyzer()
        self.volume_profile = VolumeProfileAnalyzer()
        self.smart_sltp = SmartSLTPCalculator()
        
        # PHASE 27: Signal Quality Filter
        self.signal_filter = SignalQualityFilter(
            min_confidence=60.0,
            min_mtf_confluence=50.0,
            min_risk_reward=1.5
        )
        
        # PHASE 28: Ensemble Model (RL + LSTM Voting)
        self.ensemble_model = EnsembleModel(
            rl_base_weight=0.55,
            lstm_base_weight=0.45,
            enable_dynamic_weights=True
        )
        
        # PHASE 28: Risk-Adjusted Position Sizing
        self.position_sizer = PositionSizer(
            max_position_percent=25.0,
            kelly_fraction=0.25,
            max_risk_per_trade=2.0
        )
        
        # PHASE 6: RL Agent (Self-Learning Trader) - v3: 2 YEARS DATA + 500K TIMESTEPS!
        # Multi-coin support: each coin has its own trained model
        self.rl_agents = {}  # Symbol -> RLAgent
        self.rl_model_map = {
            'BTC/USDT': 'ppo_btc_v5',  # v5: 5 years, 500K steps, Sharpe 0.13
            'ETH/USDT': 'ppo_eth_v5',  # v5: 5 years, 500K steps, Sharpe 0.13
            'LTC/USDT': 'ppo_ltc_v5',  # v5: 5 years, 500K steps, Sharpe 0.07
            'SOL/USDT': 'ppo_sol_v5'   # v5: 5 years, 500K steps, Sharpe 0.22 (BEST!)
        }
        
        self.regime_classifier = RegimeClassifier()
        
        # Başlangıç Yüklemeleri
        self.load_rl_agent()
    
    def get_rl_agent_for_symbol(self, symbol: str) -> RLAgent:
        """
        Get the RL agent for a specific symbol.
        Loads model on first use, caches for reuse.
        (Her coin için kendi RL modelini döner, ilk kullanımda yükler)
        """
        if symbol not in self.rl_agents:
            model_name = self.rl_model_map.get(symbol, 'ppo_btc_v2')  # Fallback to BTC
            agent = RLAgent()
            loaded = agent.load(model_name)
            if loaded:
                logger.info(f"🧠 Loaded RL model for {symbol}: {model_name}")
            else:
                logger.warning(f"⚠️ Could not load RL model {model_name}, using uninitialized agent")
            self.rl_agents[symbol] = agent
        return self.rl_agents[symbol]

    # ... [Rest of class methods unchanged until analyze_market] ... 
    # NOTE: I will only replace the analyze_market part in a separate call if needed or include the import + init here.
    # Since analyze_market is further down, I will apply this change to imports and init first.
    
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
                logger.debug(f"TimeNet prediction failed: {e}")
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

    async def ensure_active_brain(self):
        """
        Check if AI models exist. If not, trigger 'Cold Start' training.
        """
        from src.config.settings import Config
        from src.brain.trainer import AITrainer
        
        logger.info("🧠 Checking Brain Health (Model Verification)...")
        
        trainer = None
        
        for symbol in Config.TARGET_COINS:
            if not self.load_lstm_for_symbol(symbol):
                logger.warning(f"⚠️ Brain is cold for {symbol} (No Model). Starting accelerated learning...")
                
                if not trainer: trainer = AITrainer()
                
                success = await trainer.train_model_for_symbol(symbol)
                
                if success:
                    logger.info(f"✅ Brain activation complete for {symbol}.")
                    # Reload the newly trained model
                    self.load_lstm_for_symbol(symbol)
                else:
                    logger.error(f"❌ Brain activation failed for {symbol}.")
            else:
                logger.info(f"🧠 Brain is active for {symbol}.")

    async def analyze_market(self, symbol: str, raw_data_1h: List[Dict]) -> Optional[Dict]:
        # NOT: raw_data_1h parametresi engine'den geliyor ama biz burada kendi verimizi çekeceğiz
        # Engine'i güncellemek yerine burada override ediyoruz.
        
        # 1. MULTI-TIMEFRAME VERİ ÇEK
        mtf_data = await self.fetch_multi_timeframe_data(symbol)
        if not mtf_data or '1h' not in mtf_data:
            return None  # Cannot analyze without 1h data
            
        df_1h = mtf_data['1h']  # Ana analiz 1H üzerinden döner
        last_row = df_1h.iloc[-1]

        # 2. MAKRO VERİ & FÜZYON (using safe helper function)
        from src.brain.macro_helpers import fetch_and_merge_macro
        df = await fetch_and_merge_macro(self.macro, df_1h)
        
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
        
        # Check for macro columns in the merged df
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
            # A. Binance Proxy (Volume/Orderbook based)
            onchain_proxy = await self.onchain_intel.get_full_onchain_analysis(clean_symbol)
            proxy_score = onchain_proxy.get('composite_score', 0)
            
            # B. True On-Chain (Phase 22 - Etherscan/Blockchain.com)
            # Returns bias -1.0 to 1.0 -> Normalize to -100 to 100
            onchain_true = await self.onchain_analyzer.get_whale_sentiment(clean_symbol)
            true_score = onchain_true.get('score', 0) * 100 
            
            # C. Fusion (60% Proxy, 40% True Chain)
            # Proxy is faster/real-time, True Chain is slower but more fundamental
            final_onchain_score = (proxy_score * 0.6) + (true_score * 0.4)
            
            if final_onchain_score >= 25:
                onchain_signal = "STRONG_BUY"
            elif final_onchain_score >= 10:
                onchain_signal = "BUY"
            elif final_onchain_score <= -25:
                onchain_signal = "STRONG_SELL"
            elif final_onchain_score <= -10:
                onchain_signal = "SELL"
            else:
                onchain_signal = "NEUTRAL"
            
            # Combine for downstream usage
            onchain_data = {
                'proxy': onchain_proxy,
                'true_chain': onchain_true,
                'signal': onchain_signal,
                'composite_score': final_onchain_score
            }
            
            onchain_score = final_onchain_score # FIX: Define variable for downstream use
            logger.info(f"🐋 On-Chain Fusion: {onchain_signal} (Final: {final_onchain_score:.1f} | Proxy: {proxy_score} | True: {true_score})")
            
        except Exception as e:
            logger.warning(f"On-chain analysis failed: {e}")
            onchain_data = {'signal': 'NEUTRAL', 'composite_score': 0}
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
        
        # Use coin-specific RL agent (v2 models - 200K timesteps trained!)
        rl_agent = self.get_rl_agent_for_symbol(symbol)
        if rl_agent and rl_agent.is_trained:
            try:
                # RL agent predicts using the unified state vector
                # Uses RLAgent.predict() wrapper for proper handling
                action, rl_confidence = rl_agent.predict(state_vector, deterministic=True)
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
        
        # 1. RL AGENT (TRUE AI) - ÖNCELİKLİ - Using v2 models (200K trained)
        rl_agent = self.get_rl_agent_for_symbol(symbol)
        if rl_agent and rl_agent.is_trained:
            try:
                # RL agent predicts using wrapper for proper handling
                action, rl_confidence = rl_agent.predict(state_vector, deterministic=True)
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
        # 6. KARAR MEKANİZMASI (DECISION ENGINE)
        # RL Agent veya Kural Tabanlı Karar
        
        # PHASE 12: Visual Analysis
        logger.info(f"👁️ Requesting Visual Analysis for {symbol}...")
        visual_analysis = self.vision.analyze_chart(symbol, df)
        logger.info(f"👁️ VISUAL OPINION: {visual_analysis['trend']} | Score: {visual_analysis['visual_score']} | Pattern: {visual_analysis['pattern']}")
        
        # PHASE 13: Sentiment Analysis
        sentiment_data = self.sentiment.get_sentiment(symbol.split('/')[0])  # Extract base symbol (BTC from BTC/USDT)
        logger.info(f"📊 SENTIMENT: {sentiment_data['sentiment']} | Score: {sentiment_data['composite_score']} | F&G: {sentiment_data['fear_greed_index']}")

        # PHASE 17: Macro IQ
        try:
            macro_data = self.macro.fetch_data() if hasattr(self, 'macro') else {}
        except Exception:
            macro_data = {}

        # ======================================
        # PHASE 23: ADVANCED SIGNAL FEATURES
        # SMC, MTF Confluence, Volume Profile, Smart SL/TP
        # ======================================
        try:
            # SMC Analysis (Order Blocks, FVG, Liquidity)
            smc_data = self.smc_analyzer.analyze(df)
            logger.info(f"🎯 SMC: {smc_data.get('smc_bias', 'N/A')} | OBs: {len(smc_data.get('order_blocks', []))} | FVGs: {len(smc_data.get('fvgs', []))}")
        except Exception as e:
            logger.warning(f"SMC analysis failed: {e}")
            smc_data = {}
        
        try:
            # MTF Confluence (1H/4H/1D)
            mtf_confluence = self.mtf_analyzer.analyze(symbol)
            logger.info(f"📊 MTF: {mtf_confluence.get('confluence_type', 'N/A')} ({mtf_confluence.get('confluence_score', 0)}%)")
        except Exception as e:
            logger.warning(f"MTF analysis failed: {e}")
            mtf_confluence = {}
        
        try:
            # Volume Profile (VPOC, HVN, LVN)
            vp_data = self.volume_profile.analyze(df)
            logger.info(f"📈 VP: VPOC=${vp_data.get('vpoc', 0):,.0f} | Position: {vp_data.get('price_position', 'N/A')}")
        except Exception as e:
            logger.warning(f"Volume Profile analysis failed: {e}")
            vp_data = {}
        
        try:
            # Smart SL/TP (based on SMC + MTF + VP)
            price = float(last_row['close'])
            atr = float(last_row.get('atr', price * 0.02))
            direction = 'LONG' if ai_decision == 'BUY' else 'SHORT' if ai_decision == 'SELL' else 'NONE'
            
            smart_sltp = self.smart_sltp.calculate(
                direction=direction,
                entry_price=price,
                smc_data=smc_data,
                mtf_data=mtf_confluence,
                vp_data=vp_data,
                atr=atr
            )
            logger.info(f"🎯 SL/TP: SL=${smart_sltp.get('stop_loss', 0):,.0f} | TP1=${smart_sltp.get('take_profit_1', 0):,.0f} | Quality: {smart_sltp.get('quality', 'N/A')}")
        except Exception as e:
            logger.warning(f"Smart SL/TP calculation failed: {e}")
            smart_sltp = {}

        snapshot = {
            "symbol": symbol,
            "price": float(last_row['close']),
            "dxy": float(macro_data.get('dxy', 0)), 
            "vix": float(macro_data.get('vix', 0)), 
            "macro_score": macro_data.get('macro_score', 0),
            "interest_rate": macro_data.get('interest_rate', 0),
            "macro_debug": macro_data.get('debug_errors', 'OK'),
            "funding_rate": funding_rate * 100,
            "ai_decision": ai_decision,
            "ai_confidence": ai_confidence,
            "reason": reason,
            "visual_analysis": visual_analysis,  # Export Visual Data
            "sentiment_data": sentiment_data,  # Export Sentiment Data
            "brain_state": {
                "tech_attention": abs(tech_direction) * weights['tech'],
                "pattern_attention": abs(pattern_direction) * weights['pattern'],
                "lstm_attention": abs(lstm_direction) * weights['lstm'],
                "htf_attention": abs(htf_direction) * weights['htf'],
                "htf_direction": htf_direction,
                "onchain_attention": abs(onchain_direction) * weights['onchain'],
                "rl_action": int(rl_action) if self.rl_agent and 'rl_action' in locals() else -1,
                # New Visual Metrics
                "visual_score": visual_analysis['visual_score'],
                "visual_trend": visual_analysis['trend'],
                # New Sentiment Metrics
                "sentiment_score": sentiment_data['composite_score'],
                "sentiment": sentiment_data['sentiment'],
                "fear_greed": sentiment_data['fear_greed_index']
            },
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
            # PHASE 23: Advanced Signal Features
            "smc": smc_data,
            "mtf": mtf_confluence,
            "volume_profile": vp_data,
            "smart_sltp": smart_sltp,

        }
        
        # ======================================
        # PROACTIVE EARLY WARNING SYSTEM
        # Detects patterns, whale activity, breakouts BEFORE they complete
        # ======================================
        try:
            early_warnings = EarlyWarningSystem.analyze_for_early_warnings(
                symbol, 
                snapshot,
                visual_analysis
            )
            snapshot['early_warnings'] = early_warnings
            if early_warnings:
                logger.info(f"⚡ {len(early_warnings)} Early Warnings for {symbol}")
                for w in early_warnings[:2]:  # Log top 2
                    logger.info(f"   → [{w.get('priority')}] {w.get('title')}")
        except Exception as e:
            logger.error(f"Early Warning generation failed: {e}")
            snapshot['early_warnings'] = []
        
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
