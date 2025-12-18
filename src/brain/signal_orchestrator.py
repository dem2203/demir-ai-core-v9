# -*- coding: utf-8 -*-
"""
DEMIR AI - Signal Orchestrator
Tüm veri kaynaklarını ve modelleri birleştirip güçlü sinyal üretir.

PHASE 49: Advanced Signal Fusion
- Tüm modüllerden sinyal toplama
- Dinamik ağırlık hesaplama
- Consensus mechanism
- Final güçlü sinyal üretimi
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Fix event loop conflicts
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger("SIGNAL_ORCHESTRATOR")


@dataclass
class ModuleSignal:
    """Bir modülden gelen sinyal"""
    module_name: str
    direction: str  # LONG / SHORT / NEUTRAL
    confidence: float  # 0-100
    weight: float  # 0-1 (ağırlık)
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FinalSignal:
    """Birleştirilmiş final sinyal"""
    direction: str  # LONG / SHORT / WAIT
    confidence: float  # 0-100
    strength: str  # STRONG / MODERATE / WEAK
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    contributing_modules: List[str]
    consensus_ratio: float  # Kaç modül aynı fikirde
    timestamp: datetime = field(default_factory=datetime.now)


class SignalOrchestrator:
    """
    Merkezi Sinyal Orkestratörü
    
    Tüm modülleri koordine eder:
    1. Markov Predictor
    2. LSTM Trend (trained model)
    3. Research Agent
    4. SMC Analyzer
    5. Whale Intelligence
    6. Liquidation Hunter
    7. Predictive Analyzer
    8. News Sentiment (YENİ)
    9. CME Gap Tracker (YENİ)
    10. Options Flow (YENİ)
    11. OnChain Intel (YENİ)
    
    Consensus kuralları:
    - Minimum 5/11 modül aynı yönde olmalı
    - Güven skoru > 65% olmalı
    - Risk/Reward > 2:1 olmalı
    """
    
    # Modül ağırlıkları (performansa göre ayarlanabilir)
    DEFAULT_WEIGHTS = {
        'MarkovPredictor': 0.06,
        'LSTMTrend': 0.08,
        'ResearchAgent': 0.07,
        'SMCAnalyzer': 0.07,
        'WhaleIntelligence': 0.07,
        'LiquidationHunter': 0.05,
        'PredictiveAnalyzer': 0.05,
        'NewsSentiment': 0.06,
        'CMEGapTracker': 0.07,
        'OptionsFlow': 0.05,
        'OnChainIntel': 0.05,
        'TradingViewTA': 0.10,  # Proven data source
        # PHASE 61-64: Sentiment & Multi-Source
        'TwitterSentiment': 0.04,  # PHASE 61
        'OrderBookDepth': 0.05,    # PHASE 62
        'EnsembleModel': 0.03,     # PHASE 63
        'MultiTimeframe': 0.03,    # PHASE 64
        'GoogleTrends': 0.02,      # PHASE 66 - Retail sentiment
        # PHASE 71-75: SUDDEN MOVEMENT DETECTION 🚨
        'BollingerSqueeze': 0.05,     # PHASE 71 - Volatility compression
        'LiquidationCascade': 0.06,   # PHASE 72 - Squeeze risk
        'VolumeSpike': 0.05,          # PHASE 73 - Big player activity
        'TakerFlowDelta': 0.05,       # PHASE 74 - Aggressive buy/sell
        'ExchangeDivergence': 0.04,   # PHASE 75 - Coinbase premium
    }
    
    # Minimum sinyal gereksinimleri
    MIN_CONSENSUS_RATIO = 0.5  # En az %50 aynı fikirde (11 modül için)
    MIN_CONFIDENCE = 60  # Minimum güven
    MIN_RISK_REWARD = 2.0  # Minimum R:R
    
    def __init__(self):
        self.module_signals: List[ModuleSignal] = []
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.last_signal: Optional[FinalSignal] = None
        self.signal_history: List[FinalSignal] = []
    
    async def collect_all_signals(self, symbol: str = 'BTCUSDT', current_price: float = 0) -> List[ModuleSignal]:
        """Tüm modüllerden sinyal topla."""
        self.module_signals = []
        import requests
        
        # 1. Markov Predictor - TRAINED on historical data
        try:
            from src.brain.markov_predictor import MarkovPredictor
            import pandas as pd
            
            markov = MarkovPredictor()
            
            # Fetch 100h of data for training
            resp = requests.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=100", timeout=5)
            if resp.status_code == 200:
                klines = resp.json()
                
                # Train Markov on historical data
                df = pd.DataFrame({
                    'close': [float(k[4]) for k in klines]
                })
                markov.train(df, interval_hours=1)
                
                # Get prediction with trained model
                pct_change = ((float(klines[-1][4]) / float(klines[-2][4])) - 1) * 100
                pred = markov.predict_1_2_hours(pct_change)
                
                # Boost confidence for non-WAIT signals
                raw_strength = pred['signal_strength']
                if pred['combined_signal'] != 'WAIT':
                    boosted_strength = max(50, raw_strength + 20)  # Min 50% for actionable signals
                else:
                    boosted_strength = max(40, raw_strength + 10)  # Min 40% for all
                
                self.module_signals.append(ModuleSignal(
                    module_name='MarkovPredictor',
                    direction=pred['combined_signal'].replace('WAIT', 'NEUTRAL'),
                    confidence=boosted_strength,
                    weight=self.weights['MarkovPredictor'],
                    reasoning=f"1h: {pred['1_hour']['direction']}, 2h: {pred['2_hour']['direction']} (trained)"
                ))
        except Exception as e:
            logger.warning(f"Markov signal failed: {e}")
        
        # 2. LSTM Trend - EMA Crossover (replaces broken TF model)
        try:
            import pandas as pd
            
            resp = requests.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=50", timeout=5)
            if resp.status_code == 200:
                klines = resp.json()
                closes = [float(k[4]) for k in klines]
                
                # Calculate EMA9 and EMA21 crossover
                def ema(data, period):
                    multiplier = 2 / (period + 1)
                    ema_val = data[0]
                    for price in data[1:]:
                        ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
                    return ema_val
                
                ema9 = ema(closes[-20:], 9)
                ema21 = ema(closes[-30:], 21)
                ema50 = ema(closes, 50) if len(closes) >= 50 else ema(closes, len(closes))
                
                current = closes[-1]
                prev = closes[-2]
                momentum = ((current / prev) - 1) * 100
                
                # Trend detection via EMA alignment
                if ema9 > ema21 > ema50:
                    direction = 'LONG'
                    confidence = 60
                    reason = 'EMA9 > EMA21 > EMA50 (Strong uptrend)'
                elif ema9 > ema21:
                    direction = 'LONG'
                    confidence = 55
                    reason = 'EMA9 > EMA21 (Uptrend)'
                elif ema9 < ema21 < ema50:
                    direction = 'SHORT'
                    confidence = 60
                    reason = 'EMA9 < EMA21 < EMA50 (Strong downtrend)'
                elif ema9 < ema21:
                    direction = 'SHORT'
                    confidence = 55
                    reason = 'EMA9 < EMA21 (Downtrend)'
                else:
                    direction = 'NEUTRAL'
                    confidence = 45
                    reason = 'EMAs tangled (No clear trend)'
                
                # Momentum boost
                if abs(momentum) > 0.5:
                    confidence += 5
                
                self.module_signals.append(ModuleSignal(
                    module_name='LSTMTrend',
                    direction=direction,
                    confidence=confidence,
                    weight=self.weights['LSTMTrend'],
                    reasoning=reason
                ))
        except Exception as e:
            logger.warning(f"LSTM signal failed: {e}")
        
        # 3. Research Agent
        try:
            from src.brain.research_agent import ResearchAgent
            agent = ResearchAgent()
            
            research = await agent.research_coin(symbol)
            
            dir_map = {'BULLISH': 'LONG', 'BEARISH': 'SHORT', 'NEUTRAL': 'NEUTRAL'}
            self.module_signals.append(ModuleSignal(
                module_name='ResearchAgent',
                direction=dir_map.get(research.overall_bias, 'NEUTRAL'),
                confidence=research.overall_confidence,
                weight=self.weights['ResearchAgent'],
                reasoning=f"{len(research.findings)} bulgu analiz edildi"
            ))
        except Exception as e:
            logger.warning(f"Research signal failed: {e}")
        
        # 4. Whale Intelligence (YENİ - Binance API)
        try:
            from src.brain.whale_intelligence import WhaleIntelligence
            whale = WhaleIntelligence()
            whale_signal = whale.get_signal_for_orchestrator(symbol)
            
            if whale_signal.get('confidence', 0) > 0:
                self.module_signals.append(ModuleSignal(
                    module_name='WhaleIntelligence',
                    direction=whale_signal['direction'],
                    confidence=whale_signal.get('confidence', 50),
                    weight=self.weights['WhaleIntelligence'],
                    reasoning=whale_signal.get('reason', 'Whale analizi')
                ))
        except Exception as e:
            logger.warning(f"Whale signal failed: {e}")
        
        # 5. SMC ANALYZER (Smart Money Concepts) - Always add
        try:
            from src.brain.smc_analyzer import SMCAnalyzer
            import pandas as pd
            
            smc = SMCAnalyzer()
            
            # Fetch OHLCV data from Binance
            resp = requests.get(
                f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=100",
                timeout=5
            )
            if resp.status_code == 200:
                klines = resp.json()
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                result = smc.analyze(df)
                smc_signal = result.get('smc_signal', {})
                
                dir_map = {'BULLISH': 'LONG', 'BEARISH': 'SHORT', 'NEUTRAL': 'NEUTRAL'}
                self.module_signals.append(ModuleSignal(
                    module_name='SMCAnalyzer',
                    direction=dir_map.get(smc_signal.get('direction', 'NEUTRAL'), 'NEUTRAL'),
                    confidence=smc_signal.get('strength', 40),
                    weight=self.weights['SMCAnalyzer'],
                    reasoning=smc_signal.get('reason', 'SMC OB/FVG analizi')[:50]
                ))
        except Exception as e:
            logger.warning(f"SMC signal failed: {e}")
        
        # 6. Predictive Analyzer - Simplified sync version
        try:
            # Use simple momentum prediction instead of async PredictiveAnalyzer
            resp = requests.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=4h&limit=6", timeout=5)
            if resp.status_code == 200:
                klines = resp.json()
                closes = [float(k[4]) for k in klines]
                
                # Calculate momentum
                short_ma = sum(closes[-3:]) / 3
                long_ma = sum(closes) / 6
                momentum = ((short_ma / long_ma) - 1) * 100
                
                # BOOSTED thresholds - lower threshold for signal
                if momentum > 0.1:
                    direction = 'LONG'
                    confidence = min(65, 50 + momentum * 15)  # Base 50, boost by momentum
                elif momentum < -0.1:
                    direction = 'SHORT'
                    confidence = min(65, 50 + abs(momentum) * 15)
                else:
                    direction = 'NEUTRAL'
                    confidence = 45  # Higher neutral confidence
                
                self.module_signals.append(ModuleSignal(
                    module_name='PredictiveAnalyzer',
                    direction=direction,
                    confidence=confidence,
                    weight=self.weights['PredictiveAnalyzer'],
                    reasoning=f"4h Momentum: {momentum:.2f}%"
                ))
        except Exception as e:
            logger.warning(f"Predictive signal failed: {e}")
        
        # 7. Liquidation Hunter - Sync Binance API based
        try:
            # Use Binance Futures Open Interest change as proxy for liquidation pressure
            resp = requests.get(
                f"https://fapi.binance.com/fapi/v1/openInterest",
                params={'symbol': symbol},
                timeout=5
            )
            
            # Also get OI stats for change
            oi_stats_resp = requests.get(
                f"https://fapi.binance.com/futures/data/openInterestHist",
                params={'symbol': symbol, 'period': '5m', 'limit': 12},
                timeout=5
            )
            
            if resp.status_code == 200 and oi_stats_resp.status_code == 200:
                current_oi = float(resp.json().get('openInterest', 0))
                oi_history = oi_stats_resp.json()
                
                if len(oi_history) >= 2:
                    prev_oi = float(oi_history[0].get('sumOpenInterest', current_oi))
                    oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100 if prev_oi > 0 else 0
                    
                    # OI decrease = liquidations happening
                    if oi_change_pct < -2:
                        direction = 'NEUTRAL'  # Major liquidations, wait
                        confidence = 50
                        reasoning = f"OI düştü ({oi_change_pct:.1f}%) - Likidasyonlar devam ediyor"
                    elif oi_change_pct > 3:
                        direction = 'LONG'  # Fresh positions opening, trend continuation
                        confidence = 55
                        reasoning = f"OI arttı ({oi_change_pct:.1f}%) - Yeni pozisyonlar açılıyor"
                    else:
                        direction = 'NEUTRAL'
                        confidence = 45
                        reasoning = f"OI stabil ({oi_change_pct:.1f}%)"
                    
                    self.module_signals.append(ModuleSignal(
                        module_name='LiquidationHunter',
                        direction=direction,
                        confidence=confidence,
                        weight=self.weights['LiquidationHunter'],
                        reasoning=reasoning
                    ))
        except Exception as e:
            logger.warning(f"Liquidation signal failed: {e}")
        
        # 7. NEWS SENTIMENT - Always add signal
        try:
            from src.brain.news_scraper import CryptoNewsScraper
            scraper = CryptoNewsScraper()
            scraper.fetch_all_news(max_age_hours=4)
            sentiment = scraper.get_market_sentiment()
            
            dir_map = {'BULLISH': 'LONG', 'BEARISH': 'SHORT', 'NEUTRAL': 'NEUTRAL'}
            news_conf = max(45, sentiment.get('confidence', 45))  # Min 45% confidence
            self.module_signals.append(ModuleSignal(
                module_name='NewsSentiment',
                direction=dir_map.get(sentiment['overall'], 'NEUTRAL'),
                confidence=news_conf,
                weight=self.weights['NewsSentiment'],
                reasoning=f"{sentiment['bullish_count']} bullish, {sentiment['bearish_count']} bearish haber"
            ))
        except Exception as e:
            logger.warning(f"News signal failed: {e}")
        
        # 8. CME GAP TRACKER (YENİ)
        try:
            from src.brain.cme_gap_tracker import CMEGapTracker
            tracker = CMEGapTracker()
            gap_signal = tracker.get_signal_bias()
            
            if gap_signal.get('has_gap') and gap_signal.get('signal') != 'NEUTRAL':
                self.module_signals.append(ModuleSignal(
                    module_name='CMEGapTracker',
                    direction=gap_signal['signal'],
                    confidence=gap_signal.get('confidence', 50),
                    weight=self.weights['CMEGapTracker'],
                    reasoning=gap_signal.get('reason', 'CME Gap analizi')
                ))
        except Exception as e:
            logger.warning(f"CME Gap signal failed: {e}")
        
        # 9. OPTIONS FLOW - Always add signal
        try:
            from src.brain.options_flow import OptionsFlowAnalyzer
            analyzer = OptionsFlowAnalyzer()
            options = analyzer.get_signal_for_orchestrator()
            
            self.module_signals.append(ModuleSignal(
                module_name='OptionsFlow',
                direction=options.get('direction', 'NEUTRAL'),
                confidence=options.get('confidence', 40),
                weight=self.weights['OptionsFlow'],
                reasoning=options.get('reason', 'Options flow analizi')
            ))
        except Exception as e:
            logger.warning(f"Options signal failed: {e}")
        
        # 10. ON-CHAIN INTEL - Sync version (avoid async issues)
        try:
            from src.brain.whale_intelligence import WhaleIntelligence
            onchain = WhaleIntelligence()  # Use Binance API instead
            analysis = onchain.get_full_whale_analysis(symbol)
            
            if analysis.get('available'):
                self.module_signals.append(ModuleSignal(
                    module_name='OnChainIntel',
                    direction=analysis.get('whale_bias', 'NEUTRAL'),
                    confidence=analysis.get('confidence', 40),
                    weight=self.weights['OnChainIntel'],
                    reasoning=f"L/S: {analysis.get('long_short_ratio', 1):.2f}, Funding: {analysis.get('funding_rate_pct', 0):.3f}%"
                ))
        except Exception as e:
            logger.warning(f"OnChain signal failed: {e}")
        
        # 11. TRADINGVIEW TECHNICAL ANALYSIS (YENİ - En güvenilir kaynak)
        try:
            from src.brain.tv_playwright import TradingViewPlaywright
            tv = TradingViewPlaywright()
            tv_signal = tv.get_signal_for_orchestrator(symbol)
            
            if tv_signal.get('confidence', 0) > 0:
                self.module_signals.append(ModuleSignal(
                    module_name='TradingViewTA',
                    direction=tv_signal['direction'],
                    confidence=tv_signal.get('confidence', 50),
                    weight=self.weights['TradingViewTA'],
                    reasoning=tv_signal.get('reason', 'TradingView Technical Analysis')
                ))
        except Exception as e:
            logger.warning(f"TradingView signal failed: {e}")
        
        # 12. TWITTER SENTIMENT (PHASE 61) - Influencer mood analysis
        try:
            from src.brain.twitter_sentiment import TwitterSentimentScraper
            twitter = TwitterSentimentScraper()
            sentiment = twitter.get_influencer_sentiment(hours=24)
            
            if sentiment.get('tweet_count', 0) > 0:
                score = sentiment.get('score', 50)  # 0-100, 50=neutral
                
                if score > 60:
                    direction = 'LONG'
                    confidence = min(65, 45 + (score - 50) * 0.4)
                elif score < 40:
                    direction = 'SHORT'
                    confidence = min(65, 45 + (50 - score) * 0.4)
                else:
                    direction = 'NEUTRAL'
                    confidence = 45
                
                self.module_signals.append(ModuleSignal(
                    module_name='TwitterSentiment',
                    direction=direction,
                    confidence=confidence,
                    weight=self.weights['TwitterSentiment'],
                    reasoning=f"X Influencer: {sentiment.get('sentiment', 'NEUTRAL')} ({sentiment.get('bullish_count', 0)}↑/{sentiment.get('bearish_count', 0)}↓)"
                ))
        except Exception as e:
            logger.warning(f"Twitter signal failed: {e}")
        
        # 13. ORDER BOOK DEPTH (PHASE 62) - Whale walls detection
        try:
            from src.data_ingestion.orderbook_analyzer import OrderBookAnalyzer
            import asyncio
            
            ob = OrderBookAnalyzer()
            
            # Run async in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            
            # Fetch and analyze orderbook
            orderbook = loop.run_until_complete(ob.fetch_order_book(symbol))
            if orderbook:
                analysis = loop.run_until_complete(ob.detect_whale_walls(orderbook, current_price))
                
                bid_liq = analysis.get('total_bid_liquidity', 0)
                ask_liq = analysis.get('total_ask_liquidity', 0)
                
                if bid_liq > ask_liq * 1.5:
                    direction = 'LONG'
                    confidence = min(65, 50 + (bid_liq / max(ask_liq, 1) - 1) * 10)
                elif ask_liq > bid_liq * 1.5:
                    direction = 'SHORT'
                    confidence = min(65, 50 + (ask_liq / max(bid_liq, 1) - 1) * 10)
                else:
                    direction = 'NEUTRAL'
                    confidence = 45
                
                self.module_signals.append(ModuleSignal(
                    module_name='OrderBookDepth',
                    direction=direction,
                    confidence=confidence,
                    weight=self.weights['OrderBookDepth'],
                    reasoning=f"Bid: ${bid_liq/1e6:.1f}M, Ask: ${ask_liq/1e6:.1f}M"
                ))
            loop.run_until_complete(ob.close())
        except Exception as e:
            logger.warning(f"OrderBook signal failed: {e}")
        
        # 14. ENSEMBLE MODEL (PHASE 63) - RL+LSTM voting
        try:
            from src.brain.ensemble_model import EnsembleModel
            ensemble = EnsembleModel()
            
            # Get RL and LSTM predictions from existing signals
            rl_signal = next((s for s in self.module_signals if s.module_name == 'MarkovPredictor'), None)
            lstm_signal = next((s for s in self.module_signals if s.module_name == 'LSTMTrend'), None)
            
            if rl_signal and lstm_signal:
                rl_action = 2 if rl_signal.direction == 'LONG' else (0 if rl_signal.direction == 'SHORT' else 1)
                lstm_pred = 0.8 if lstm_signal.direction == 'LONG' else (0.2 if lstm_signal.direction == 'SHORT' else 0.5)
                
                signal, conf, reason = ensemble.predict(
                    rl_action=rl_action,
                    rl_confidence=rl_signal.confidence,
                    lstm_prediction=lstm_pred,
                    lstm_confidence=lstm_signal.confidence
                )
                
                self.module_signals.append(ModuleSignal(
                    module_name='EnsembleModel',
                    direction=signal,
                    confidence=conf,
                    weight=self.weights['EnsembleModel'],
                    reasoning=reason[:50]
                ))
        except Exception as e:
            logger.warning(f"Ensemble signal failed: {e}")
        
        # 15. MULTI-TIMEFRAME SYNC (PHASE 64) - 1h + 4h + 1d alignment
        try:
            # Fetch EMA for 3 timeframes
            def get_ema_direction(tf: str) -> str:
                resp = requests.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit=30", timeout=5)
                if resp.status_code != 200:
                    return 'NEUTRAL'
                closes = [float(k[4]) for k in resp.json()]
                ema9 = sum(closes[-9:]) / 9
                ema21 = sum(closes[-21:]) / 21
                return 'LONG' if ema9 > ema21 else 'SHORT'
            
            tf_1h = get_ema_direction('1h')
            tf_4h = get_ema_direction('4h')
            tf_1d = get_ema_direction('1d')
            
            # Check alignment
            directions = [tf_1h, tf_4h, tf_1d]
            long_count = directions.count('LONG')
            short_count = directions.count('SHORT')
            
            if long_count == 3:
                direction = 'LONG'
                confidence = 70  # Perfect alignment
            elif short_count == 3:
                direction = 'SHORT'
                confidence = 70
            elif long_count >= 2:
                direction = 'LONG'
                confidence = 55
            elif short_count >= 2:
                direction = 'SHORT'
                confidence = 55
            else:
                direction = 'NEUTRAL'
                confidence = 40
            
            self.module_signals.append(ModuleSignal(
                module_name='MultiTimeframe',
                direction=direction,
                confidence=confidence,
                weight=self.weights['MultiTimeframe'],
                reasoning=f"1h:{tf_1h} 4h:{tf_4h} 1d:{tf_1d}"
            ))
        except Exception as e:
            logger.warning(f"MultiTimeframe signal failed: {e}")
        
        # 16. GOOGLE TRENDS (PHASE 66) - Retail FOMO detection
        try:
            from src.brain.google_trends import GoogleTrendsScraper
            trends = GoogleTrendsScraper()
            result = trends.get_bitcoin_interest(days=7)
            
            if result.get('available'):
                dir_map = {'BULLISH': 'LONG', 'BEARISH': 'SHORT', 'NEUTRAL': 'NEUTRAL'}
                self.module_signals.append(ModuleSignal(
                    module_name='GoogleTrends',
                    direction=dir_map.get(result['sentiment'], 'NEUTRAL'),
                    confidence=result.get('confidence', 40),
                    weight=self.weights['GoogleTrends'],
                    reasoning=f"BTC Search: {result.get('current_interest', 0)} ({result.get('change_pct', 0):+.1f}%)"
                ))
        except Exception as e:
            logger.warning(f"Google Trends signal failed: {e}")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 71-75: SUDDEN MOVEMENT DETECTION SYSTEM 🚨
        # ═══════════════════════════════════════════════════════════════════
        
        # 17. BOLLINGER SQUEEZE (PHASE 71) - Volatility compression
        try:
            from src.brain.bollinger_squeeze import BollingerSqueezeDetector
            squeeze = BollingerSqueezeDetector()
            result = squeeze.detect_squeeze(symbol, '15m')
            
            if result.get('available'):
                if result.get('squeeze_active') or result.get('breakout_imminent'):
                    self.module_signals.append(ModuleSignal(
                        module_name='BollingerSqueeze',
                        direction=result['direction'],
                        confidence=result.get('confidence', 50),
                        weight=self.weights['BollingerSqueeze'],
                        reasoning=f"Squeeze: {result.get('bandwidth_pct', 0):.1f}% {'🔥BREAKOUT!' if result.get('breakout_imminent') else ''}"
                    ))
        except Exception as e:
            logger.warning(f"Bollinger Squeeze signal failed: {e}")
        
        # 18. LIQUIDATION CASCADE (PHASE 72) - Long/Short squeeze risk
        try:
            from src.brain.liquidation_cascade import LiquidationCascadePredictor
            cascade = LiquidationCascadePredictor()
            result = cascade.predict_cascade(symbol)
            
            if result.get('available') and result.get('cascade_risk') in ['HIGH', 'MEDIUM']:
                self.module_signals.append(ModuleSignal(
                    module_name='LiquidationCascade',
                    direction=result['direction'],
                    confidence=result.get('confidence', 50),
                    weight=self.weights['LiquidationCascade'],
                    reasoning=f"{result.get('squeeze_type', '')} Risk:{result.get('risk_score', 0)} F:{result.get('funding_rate_pct', 0):.2f}%"
                ))
        except Exception as e:
            logger.warning(f"Liquidation Cascade signal failed: {e}")
        
        # 19. VOLUME SPIKE (PHASE 73) - Big player activity
        try:
            from src.brain.volume_spike import VolumeSpikeDetector
            spike = VolumeSpikeDetector()
            result = spike.detect_spike(symbol, '15m')
            
            if result.get('available') and result.get('spike_detected'):
                self.module_signals.append(ModuleSignal(
                    module_name='VolumeSpike',
                    direction=result['direction'],
                    confidence=result.get('confidence', 55),
                    weight=self.weights['VolumeSpike'],
                    reasoning=f"Volume: {result.get('spike_strength', 1):.1f}x normal ({result.get('candle_type', '')})"
                ))
        except Exception as e:
            logger.warning(f"Volume Spike signal failed: {e}")
        
        # 20. TAKER FLOW DELTA (PHASE 74) - Aggressive buy/sell imbalance
        try:
            from src.brain.taker_flow import TakerFlowDelta
            taker = TakerFlowDelta()
            result = taker.analyze_flow(symbol, minutes=15)
            
            if result.get('available') and result.get('imbalance') != 'NONE':
                self.module_signals.append(ModuleSignal(
                    module_name='TakerFlowDelta',
                    direction=result['direction'],
                    confidence=result.get('confidence', 50),
                    weight=self.weights['TakerFlowDelta'],
                    reasoning=f"Taker B/S: {result.get('ratio', 1):.2f} ({result.get('imbalance', '')})"
                ))
        except Exception as e:
            logger.warning(f"Taker Flow signal failed: {e}")
        
        # 21. EXCHANGE DIVERGENCE (PHASE 75) - Coinbase/Kraken premium
        try:
            from src.brain.exchange_divergence import ExchangeDivergenceDetector
            diverge = ExchangeDivergenceDetector()
            result = diverge.detect_divergence(symbol)
            
            if result.get('available') and result.get('divergence_type') != 'ALIGNED':
                self.module_signals.append(ModuleSignal(
                    module_name='ExchangeDivergence',
                    direction=result['direction'],
                    confidence=result.get('confidence', 50),
                    weight=self.weights['ExchangeDivergence'],
                    reasoning=f"{result.get('divergence_type', '')}: {result.get('premium_pct', 0):+.2f}%"
                ))
        except Exception as e:
            logger.warning(f"Exchange Divergence signal failed: {e}")
        
        # PHASE 67: Correlation Filter Warning
        long_count = sum(1 for s in self.module_signals if s.direction == 'LONG')
        short_count = sum(1 for s in self.module_signals if s.direction == 'SHORT')
        total = len(self.module_signals)
        
        if long_count >= 10:
            logger.warning(f"⚠️ CROWDED TRADE WARNING: {long_count}/{total} modules are LONG!")
        elif short_count >= 10:
            logger.warning(f"⚠️ CROWDED TRADE WARNING: {short_count}/{total} modules are SHORT!")
        
        logger.info(f"Collected {len(self.module_signals)} signals from {len(self.weights)} modules")
        return self.module_signals
    
    def calculate_consensus(self) -> Tuple[str, float, float]:
        """Konsensüs hesapla."""
        if not self.module_signals:
            return 'WAIT', 0, 0
        
        long_score = 0
        short_score = 0
        neutral_score = 0
        total_weight = 0
        
        for sig in self.module_signals:
            total_weight += sig.weight
            weighted = sig.weight * (sig.confidence / 100)
            
            if sig.direction == 'LONG':
                long_score += weighted
            elif sig.direction == 'SHORT':
                short_score += weighted
            else:
                neutral_score += weighted
        
        if total_weight == 0:
            return 'WAIT', 0, 0
        
        # Normalize
        long_pct = long_score / total_weight
        short_pct = short_score / total_weight
        
        # Determine winner
        if long_pct > short_pct and long_pct > 0.4:
            consensus_dir = 'LONG'
            consensus_ratio = long_pct
        elif short_pct > long_pct and short_pct > 0.4:
            consensus_dir = 'SHORT'
            consensus_ratio = short_pct
        else:
            consensus_dir = 'WAIT'
            consensus_ratio = max(long_pct, short_pct)
        
        # Ortalama güven
        avg_confidence = sum(s.confidence * s.weight for s in self.module_signals) / total_weight
        
        return consensus_dir, consensus_ratio * 100, avg_confidence
    
    def generate_final_signal(self, current_price: float) -> Optional[FinalSignal]:
        """Final sinyal üret."""
        direction, consensus_ratio, avg_confidence = self.calculate_consensus()
        
        # Minimum gereksinimleri kontrol et
        if direction == 'WAIT':
            logger.info("No clear consensus - WAIT signal")
            return None
        
        if consensus_ratio < self.MIN_CONSENSUS_RATIO * 100:
            logger.info(f"Consensus too low: {consensus_ratio:.0f}% < {self.MIN_CONSENSUS_RATIO*100}%")
            return None
        
        if avg_confidence < self.MIN_CONFIDENCE:
            logger.info(f"Confidence too low: {avg_confidence:.0f}% < {self.MIN_CONFIDENCE}%")
            return None
        
        # Entry, SL, TP hesapla
        if direction == 'LONG':
            entry = current_price
            stop_loss = current_price * 0.98  # %2 SL
            take_profit = current_price * 1.04  # %4 TP
        else:  # SHORT
            entry = current_price
            stop_loss = current_price * 1.02  # %2 SL
            take_profit = current_price * 0.96  # %4 TP
        
        risk_reward = abs(take_profit - entry) / abs(entry - stop_loss) if entry != stop_loss else 0
        
        # R:R kontrolü
        if risk_reward < self.MIN_RISK_REWARD:
            logger.info(f"R:R too low: {risk_reward:.1f} < {self.MIN_RISK_REWARD}")
            return None
        
        # Sinyal gücü
        if avg_confidence > 80 and consensus_ratio > 80:
            strength = 'STRONG'
        elif avg_confidence > 65 and consensus_ratio > 65:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'
        
        # Katkıda bulunan modüller
        contributing = [s.module_name for s in self.module_signals if s.direction == direction]
        
        signal = FinalSignal(
            direction=direction,
            confidence=avg_confidence,
            strength=strength,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            contributing_modules=contributing,
            consensus_ratio=consensus_ratio
        )
        
        self.last_signal = signal
        self.signal_history.append(signal)
        
        logger.info(f"🎯 FINAL SIGNAL: {direction} | Confidence: {avg_confidence:.0f}% | R:R: {risk_reward:.1f}")
        
        return signal
    
    async def orchestrate(self, symbol: str = 'BTCUSDT') -> Optional[FinalSignal]:
        """
        Ana orkestrasyon fonksiyonu.
        Tüm süreci yönetir.
        """
        # Güncel fiyatı al
        try:
            import requests
            resp = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5)
            current_price = float(resp.json()['price']) if resp.status_code == 200 else 0
        except:
            current_price = 0
        
        if current_price == 0:
            logger.error("Cannot get current price")
            return None
        
        # Tüm sinyalleri topla
        await self.collect_all_signals(symbol, current_price)
        
        # Final sinyal üret
        return self.generate_final_signal(current_price)
    
    def get_signal_breakdown(self) -> Dict:
        """Sinyal detay raporu."""
        return {
            'signals_collected': len(self.module_signals),
            'modules': [
                {
                    'name': s.module_name,
                    'direction': s.direction,
                    'confidence': s.confidence,
                    'weight': s.weight,
                    'reasoning': s.reasoning
                }
                for s in self.module_signals
            ],
            'consensus': self.calculate_consensus(),
            'last_signal': {
                'direction': self.last_signal.direction if self.last_signal else None,
                'confidence': self.last_signal.confidence if self.last_signal else None,
                'strength': self.last_signal.strength if self.last_signal else None,
            } if self.last_signal else None
        }


# Convenience functions
def get_orchestrated_signal(symbol: str = 'BTCUSDT') -> Optional[Dict]:
    """Senkron orkestre sinyal."""
    orchestrator = SignalOrchestrator()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    signal = loop.run_until_complete(orchestrator.orchestrate(symbol))
    loop.close()
    
    if signal:
        return {
            'direction': signal.direction,
            'confidence': signal.confidence,
            'strength': signal.strength,
            'entry': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'risk_reward': signal.risk_reward,
            'modules': signal.contributing_modules,
            'consensus': signal.consensus_ratio
        }
    return None
