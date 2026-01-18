import logging
import asyncio
from typing import Dict, List, Optional
from src.brain.macro import MacroBrain
from src.brain.technical_analyzer import TechnicalAnalyzer
from src.brain.price_action_detector import PriceActionDetector
from src.brain.market_microstructure import MarketMicrostructure
from src.brain.claude_strategist import ClaudeStrategist
from src.brain.news_sentiment import NewsSentimentAnalyzer
from src.brain.deepseek_validator import DeepSeekValidator
from src.brain.gemini_vision import GeminiVisionAnalyzer  # NEW: Visual chart analysis

# Optional: TradingView screenshot (requires Playwright)
try:
    from src.utils.chart_capture import TradingViewCapture
    SCREENSHOT_AVAILABLE = True
except ImportError:
    SCREENSHOT_AVAILABLE = False
    TradingViewCapture = None

from src.infrastructure.binance_api import BinanceAPI

logger = logging.getLogger("AI_CORTEX")

class AIVote:
    """Individual AI's vote"""
    def __init__(self, name: str, vote: str, confidence: int, reasoning: str):
        self.name = name
        self.vote = vote  # BULLISH, BEARISH, NEUTRAL
        self.confidence = confidence  # 1-10
        self.reasoning = reasoning

class DirectorDecision:
    """
    The final output from the AI Cortex.
    """
    def __init__(self, symbol: str, position: str, reasoning: str, confidence: int, 
                 risk_level: str, entry_conditions: str, votes: list, 
                 stop_loss: float = None, take_profit: float = None, position_size: float = None):
        self.symbol = symbol
        self.position = position  # LONG, SHORT, CASH
        self.reasoning = reasoning
        self.confidence = confidence  # 1-10
        self.risk_level = risk_level  # HIGH, MEDIUM, LOW
        self.entry_conditions = entry_conditions
        self.votes = votes  # List of AIVote objects
        self.stop_loss = stop_loss  # NEW: ATR-based stop
        self.take_profit = take_profit  # NEW: ATR-based target
        self.position_size = position_size  # NEW: Kelly-sized position
        
    def get_consensus_report(self) -> str:
        """Generate human-readable consensus report"""
        lines = ["🗳️ YAPAY ZEKA OY SONUÇLARI:"]
        lines.append("━━━━━━━━━━━━━━━━━━")
        
        for vote in self.votes:
            # Translate vote to Turkish
            vote_tr = {"BULLISH": "YÜKSELİŞ", "BEARISH": "DÜŞÜŞ", "NEUTRAL": "NÖTR", "MIXED": "KARIŞIK"}.get(vote.vote, vote.vote)
            emoji = "🟢" if vote.vote == "BULLISH" else "🔴" if vote.vote == "BEARISH" else "⚪"
            lines.append(f"{emoji} {vote.name}: {vote_tr} ({vote.confidence}/10)")
        
        # Count votes
        bullish = sum(1 for v in self.votes if v.vote == "BULLISH")
        bearish = sum(1 for v in self.votes if v.vote == "BEARISH")
        neutral = sum(1 for v in self.votes if v.vote == "NEUTRAL")
        
        lines.append(f"\n📊 Konsensus: {bullish} YÜKSELİŞ | {bearish} DÜŞÜŞ | {neutral} NÖTR")
        return "\n".join(lines)

class AICortex:
    """
    PROFESSIONAL TRADING SYSTEM WITH REAL MARKET MICROSTRUCTURE
    - Order Book Imbalance
    - Funding Rates
    - Volume Profile
    - CVD (Cumulative Volume Delta)
    - Price Action
    - AI Consensus (Claude, GPT-4, DeepSeek)
    """
    def __init__(self, binance_api: BinanceAPI):
        self.binance = binance_api
        
        # AI Modules
        self.macro = MacroBrain()
        self.technical = TechnicalAnalyzer()
        self.price_action = PriceActionDetector()
        self.market_micro = MarketMicrostructure(binance_api)
        self.claude = ClaudeStrategist()
        self.news = NewsSentimentAnalyzer()
        self.validator = DeepSeekValidator()
        
        # State tracking for RL feedback
        self.current_market_state = "LOW_VOL_RANGE"  # Default state
        
        # SOCIAL SENTIMENT (Grok - NEW)
        try:
            from src.brain.grok_sentiment import GrokSentimentAnalyzer
            self.grok = GrokSentimentAnalyzer()
            logger.info("🚀 Grok Social Sentiment initialized")
        except Exception as e:
            logger.warning(f"⚠️ Grok unavailable: {e}")
            self.grok = None
        
        # VISUAL ANALYSIS (New - Primary)
        self.gemini_vision = GeminiVisionAnalyzer()
        self.chart_capture = TradingViewCapture() if SCREENSHOT_AVAILABLE else None
        
        # Consensus requirements
        self.MIN_CONSENSUS = 2  # At least 2/3 AIs must agree
        
        # Performance tracking for self-learning
        from src.utils.signal_tracker import SignalPerformanceTracker
        from src.utils.adaptive_weights import AdaptiveModuleWeightManager
        self.tracker = SignalPerformanceTracker()
        self.adaptive_weights = AdaptiveModuleWeightManager(self.tracker)
        
        # ML SYSTEMS (NEW)
        try:
            from src.ml.lstm_regime import LSTMRegimeDetector
            from src.ml.volatility_predictor import PredictiveVolatilityScaler
            self.lstm_regime = LSTMRegimeDetector()
            self.vol_predictor = PredictiveVolatilityScaler()
            logger.info("🤖 ML Systems initialized (LSTM + Volatility Predictor)")
        except Exception as e:
            logger.warning(f"⚠️ ML Systems unavailable: {e}")
            self.lstm_regime = None
            self.vol_predictor = None
        
        # PRE-PUMP DETECTION (NEW)
        try:
            from src.brain.pre_spike_detector import PreSpikeDetector
            self.pre_spike = PreSpikeDetector()
            logger.info("🔮 Pre-Spike Detector initialized")
        except Exception as e:
            logger.warning(f"⚠️ Pre-Spike Detector unavailable: {e}")
            self.pre_spike = None
        # LIQUIDATION TRACKER (NEW)
        try:
            from src.brain.liquidation_tracker import LiquidationTracker
            self.liq_tracker = LiquidationTracker()
            logger.info("🩸 Liquidation Heatmap Engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Liquidation Tracker unavailable: {e}")
            self.liq_tracker = None
            
        # ON-CHAIN WHALE ANALYZER (NEW)
        try:
            from src.brain.on_chain_analyzer import OnChainAnalyzer
            self.whale_hunter = OnChainAnalyzer(self.grok)  # Uses Grok for analysis
            logger.info("🐋 Whale Hunter initialized (On-Chain Tracking)")
        except Exception as e:
            logger.warning(f"⚠️ Whale Hunter unavailable: {e}")
            self.whale_hunter = None
            
        # RL AGENT (Self-Learning) - NEW
        try:
            from src.ml.rl_agent import RLAgent
            self.rl_agent = RLAgent()
            logger.info("🧠 Self-Learning RL Agent initialized")
        except Exception as e:
            logger.warning(f"⚠️ RL Agent unavailable: {e}")
            self.rl_agent = None
            
    async def think(self, symbol: str) -> DirectorDecision:
        """
        PROFESSIONAL AI decision loop with MARKET MICROSTRUCTURE
        """
        logger.info(f"🧠 AI Cortex: Starting professional analysis for {symbol}...")
        
        try:
            # 1. Gather ALL Data in Parallel
            logger.info("📡 Gathering data from all sources (microstructure, sentiment, whale)...")
            data = await self._gather_all_data(symbol)
            
            # RL OPTIMIZATION (Dynamic Weights)
            # Determine state: Volatility + Trend
            current_vol = data['microstructure']['volume_profile'].get('volatility', 0.01) # fallback
            trend_strength = abs(data['chart'].get('adx', 20)) 
            
            # Ask RL Agent for best weights
            rl_weights = {}
            current_market_state = "LOW_VOL_RANGE"  # Default
            if self.rl_agent:
                state = self.rl_agent.get_state(current_vol, trend_strength)
                current_market_state = state  # Store for later learning
                rl_weights = self.rl_agent.get_optimized_weights(state)
                logger.info(f"🧠 RL Strategy ({state}): Adjusted voting weights")
            
            # Store state for this symbol (for RL feedback later)
            self.current_market_state = current_market_state
            
            # get whale data (if available) use get to avoid error
            whale_data = data.get('whale', {})
            liq_data = data.get('liquidation', {})
            
            # 2. Collect votes from all analyzers
            votes = await self._collect_votes_professional(
                data['macro'], 
                data['chart'], 
                data['news'], 
                data['price_action'], 
                data['microstructure']['order_book'],
                data['microstructure']['funding_rate'],
                data['microstructure']['volume_profile'],
                data['microstructure']['cvd'],
                data,  # Full data for Gemini
                whale_data, # New whale data
                liq_data, # New liquidation data
                rl_weights # Dynamic weights from RL
            )
            
            # 3. Claude Strategic Reasoning (WITH FEEDBACK)
            logger.info("🧠 Claude analyzing all inputs...")
            performance_feedback = self.tracker.get_ai_feedback_prompt()
            strategy = await self.claude.formulate_strategy(
                data['macro'], 
                data['chart'], 
                data['news'], 
                performance_feedback=performance_feedback
            )
            claude_vote = self._extract_claude_vote(strategy)
            votes.append(claude_vote)

            # Calculate initial consensus
            consensus = self._calculate_consensus_weighted(votes)
            
            # ===== HYBRID CONFLICT RESOLUTION (NEW) =====
            # Check for high disagreement (e.g., split vote 60/40)
            total_score = consensus['bullish'] + consensus['bearish']
            conflict_detected = False
            resolution_result = None
            
            if total_score > 0:
                bull_ratio = consensus['bullish'] / total_score
                # If ratio is between 0.4 and 0.6 (40% - 60%), we have a conflict
                if 0.4 <= bull_ratio <= 0.6:
                    logger.warning(f"⚠️ AI CONFLICT DETECTED! Bull Ratio: {bull_ratio:.2f}. Triggering Supreme Court...")
                    conflict_detected = True
                    
                    # Prepare summaries for arbitration
                    votes_summary = "\n".join([f"- {v.name}: {v.vote} ({v.confidence})" for v in votes])
                    chart_summary = f"Trend: {data['chart'].get('trend')}\nAnalysis: {data['chart'].get('analysis')[:200]}"
                    
                    # Call DeepSeek Arbitration
                    arbitration = await self.validator.resolve_conflict(votes_summary, chart_summary)
                    resolution_result = arbitration
                    
                    logger.info(f"⚖️ SUPREME COURT VERDICT: {arbitration.get('verdict')} (Conf: {arbitration.get('confidence')})")
                    logger.info(f"Reasoning: {arbitration.get('reasoning')}")
                    
                    # Apply verdict: Add huge weight to the winner
                    verdict = arbitration.get('verdict')
                    if verdict == "BULLISH":
                        consensus['bullish'] += 20  # Overpower disagreement
                    elif verdict == "BEARISH":
                        consensus['bearish'] += 20
                    else:
                        # Neutral verdict kills the trade
                        consensus['bullish'] = 0
                        consensus['bearish'] = 0

            # 4. DeepSeek Cross-Validation (Standard)
            logger.info("🔍 DeepSeek validating decisions...")
            validation = await self.validator.validate(votes, data['chart'], data['macro'])
            
            # 5. Finalize Decision
            decision = await self._build_final_decision(symbol, consensus, strategy, votes, validation, data)
            
            # Log consensus + RISK MANAGEMENT
            logger.info(f"\n{decision.get_consensus_report()}")
            logger.info(f"✅ Final Decision: {decision.position} (Confidence: {decision.confidence}/10)")
            if decision.stop_loss and decision.take_profit:
                logger.info(f"🎯 Risk Management: SL=${decision.stop_loss:.2f} | TP=${decision.take_profit:.2f}")
            if decision.position_size:
                logger.info(f"💰 Position Size: {decision.position_size:.4f} {symbol[:3]}")
            logger.info("")
            
            return decision
            
        except Exception as e:
            logger.error(f"AI Cortex error: {e}")
            import traceback
            traceback.print_exc()
            return DirectorDecision(
                symbol=symbol,
                position="CASH",
                reasoning=f"Error in AI processing: {str(e)}",
                confidence=0,
                risk_level="HIGH",
                entry_conditions="System error - stay cash",
                votes=[]
            )

    async def _gather_all_data(self, symbol: str) -> dict:
        """Gather all market data in parallel (Multi-Timeframe)"""
        # Start independent tasks
        macro_task = self.macro.analyze_world()
        news_task = self.news.analyze_sentiment()
        
        # Fetch price data (Multi-Timeframe)
        # 1h = Primary Trading Timeframe
        # 4h = Intermediate Trend
        # 1d = Major Trend
        df_1h_task = self.binance.fetch_candles(symbol, timeframe="1h", limit=200)
        df_4h_task = self.binance.fetch_candles(symbol, timeframe="4h", limit=100)
        df_1d_task = self.binance.fetch_candles(symbol, timeframe="1d", limit=100)
        
        current_price_task = self.binance.get_current_price(symbol)
        
        # Wait for price data first (needed for indicators)
        df_1h, df_4h, df_1d, current_price = await asyncio.gather(
            df_1h_task, df_4h_task, df_1d_task, current_price_task
        )
        
        # Start dependent tasks with 1h data (primary)
        chart_task = self._analyze_chart_professional(symbol, df_1h, df_4h, df_1d)
        pa_detector_task = asyncio.create_task(self.price_action.analyze_price_action(df_1h, symbol))
        
        # Market Microstructure (Orderbook, Funding, etc.)
        orderbook_task = self.market_micro.analyze_orderbook_imbalance(symbol)
        funding_task = self.market_micro.analyze_funding_rate(symbol)
        volume_task = self.market_micro.analyze_volume_profile(df_1h)
        cvd_task = self.market_micro.analyze_cvd(df_1h)
        
        # Whale Analysis (NEW)
        whale_task = self.whale_hunter.analyze_whale_activity(symbol) if self.whale_hunter else asyncio.sleep(0, result={})
        
        # Liquidation Analysis (NEW)
        liq_task = asyncio.to_thread(self.liq_tracker.analyze_liquidation_zones, df_1h, current_price) if self.liq_tracker else asyncio.sleep(0, result={})
        
        # Wait for all
        macro_data, news_data, chart_analysis, pa_data, ob_data, fund_data, vol_data, cvd_data, whale_data, liq_data = await asyncio.gather(
            macro_task, news_task, chart_task, pa_detector_task,
            orderbook_task, funding_task, volume_task, cvd_task, whale_task, liq_task
        )
        
        # Aggregate microstructure
        microstructure = {
            "order_book": ob_data,
            "funding_rate": fund_data,
            "volume_profile": vol_data,
            "cvd": cvd_data
        }
        
        return {
            "df": df_1h,
            "current_price": current_price,
            "macro": macro_data,
            "news": news_data,
            "chart": chart_analysis, # Contains HTF info now
            "price_action": pa_data,
            "microstructure": microstructure,
            "whale": whale_data,
            "liquidation": liq_data
        }

    async def _analyze_chart_professional(self, symbol: str, df_1h, df_4h, df_1d) -> dict:
        """
        Professional chart analysis using Multi-Timeframe technical indicators
        """
        try:
            if df_1h.empty:
                return {"trend": "UNKNOWN", "analysis": "No data available"}
            
            # Analyze Primary Timeframe (1h)
            analysis_1h = self.technical.analyze(df_1h, symbol)
            
            # Simple Trend Checks for HTF
            # 4h Trend
            ema200_4h = df_4h['close'].ewm(span=200).mean().iloc[-1]
            price_4h = df_4h['close'].iloc[-1]
            trend_4h = "BULLISH" if price_4h > ema200_4h else "BEARISH"
            
            # 1d Trend
            ema200_1d = df_1d['close'].ewm(span=200).mean().iloc[-1]
            price_1d = df_1d['close'].iloc[-1]
            trend_1d = "BULLISH" if price_1d > ema200_1d else "BEARISH"
            
            # Enrich Analysis with Confluence
            analysis_1h['trend_4h'] = trend_4h
            analysis_1h['trend_1d'] = trend_1d
            
            # Append context to analysis text
            analysis_1h['analysis'] += f"\n📊 Multi-Timeframe: 4h={trend_4h} | 1d={trend_1d}"
            
            return analysis_1h
            
        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
            return {"trend": "ERROR", "analysis": str(e)}
    
    async def _analyze_with_gemini_vision(self, data: dict) -> dict:
        """
        Orchestrate Gemini Vision analysis - Primary visual signal WITH CONTEXT
        
        Strategy:
        1. Always: Analyze candlestick chart from DataFrame
        2. Optional: Try TradingView screenshot (fallback if fails)
        3. Combine insights if both available
        4. **NEW: Pass macro + HTF context for smarter analysis**
        """
        try:
            df = data.get('df')
            symbol = data.get('chart', {}).get('symbol', 'BTCUSDT')
            macro_context = data.get('macro')
            htf_context = data.get('chart')  # Contains trend_1d, trend_4h
            
            # PRIMARY: Visual analysis from candlestick data WITH CONTEXT
            visual_result = await self.gemini_vision.analyze_chart_visual(
                df, symbol, macro_context, htf_context
            )
            
            # OPTIONAL: Try screenshot analysis (non-blocking)
            screenshot_result = None
            if self.chart_capture:  # Only if Playwright available
                try:
                    screenshot_path = await asyncio.wait_for(
                        self.chart_capture.capture_chart(symbol.replace('/', ''), timeframe="15"),
                        timeout=10.0
                    )
                    
                    if screenshot_path:
                        # Analyze screenshot with Gemini
                        with open(screenshot_path, 'rb') as f:
                            import base64
                            img_b64 = base64.b64encode(f.read()).decode()
                            
                        # Quick screenshot analysis
                        prompt = "Analyze this TradingView chart. BULLISH, BEARISH, or NEUTRAL? 1 sentence."
                        screenshot_result = await self.gemini_vision._call_gemini_vision(img_b64, prompt)
                        
                except Exception as e:
                    logger.warning(f"Screenshot analysis skipped: {e}")
            
            # Combine results if both available
            if screenshot_result and screenshot_result.get('verdict') == visual_result.get('verdict'):
                # Both agree - boost confidence
                visual_result['confidence'] = min(visual_result['confidence'] + 1, 10)
                visual_result['reasoning'] += " (Screenshot confirmed)"
                logger.info("📸 Screenshot CONFIRMED visual analysis")
            
            return visual_result
            
        except Exception as e:
            logger.error(f"Gemini Vision failed: {e}")
            return {'verdict': 'ERROR', 'confidence': 0, 'reasoning': str(e), 'patterns': []}
    
    def _extract_claude_vote(self, strategy) -> AIVote:
        """Extract Claude's vote from strategy"""
        position = strategy.get('position', 'CASH')
        
        if position == 'LONG':
            vote = "BULLISH"
        elif position == 'SHORT':
            vote = "BEARISH"
        else:
            vote = "NEUTRAL"
            
        confidence = strategy.get('confidence', 7) if 'confidence' in strategy else 7
        
        return AIVote(
            "Claude Strategist",  # Turkish name
            vote,
            confidence,
            strategy.get('reasoning', 'Strategic analysis')[:100]
        )
    
    
    def _calculate_consensus_weighted(self, votes: List[AIVote]) -> Dict[str, int]:
        """Calculate weighted consensus score"""
        bullish_score = 0
        bearish_score = 0
        total_weight = 0
        
        for vote in votes:
            weight = vote.confidence
            if vote.vote == "BULLISH":
                bullish_score += weight
            elif vote.vote == "BEARISH":
                bearish_score += weight
            total_weight += weight
            
        return {
            "bullish": bullish_score,
            "bearish": bearish_score,
            "total_weight": total_weight
        }

    async def _build_final_decision(self, symbol: str, consensus: dict, strategy: dict, votes: list, validation: dict, data: dict) -> DirectorDecision:
        """Construct the final DirectorDecision object with PROFESSIONAL FILTERS"""
        current_price = data['current_price']
        chart = data['chart']
        macro = data['macro']
        
        # Determine raw signals
        bullish_score = consensus['bullish']
        bearish_score = consensus['bearish']
        total_weight = consensus['total_weight']
        
        # Apply Validation Adjustment
        confidence_adjustment = validation.get('confidence_adjustment', 0)
        
        # Final Scoring
        if bullish_score > bearish_score + 10:
            position = "LONG"
            raw_confidence = (bullish_score / max(total_weight, 1)) * 10
        elif bearish_score > bullish_score + 10:
            position = "SHORT"
            raw_confidence = (bearish_score / max(total_weight, 1)) * 10
        else:
            position = "CASH"
            raw_confidence = 5
        
        # === LEVEL 2 PROFESSIONAL FILTERS ===
        
        # FILTER #1: HTF Protection (Daily trend veto)
        trend_1d = chart.get('trend_1d', 'UNKNOWN')
        if position == "LONG" and trend_1d == "BEARISH" and raw_confidence < 8:
            logger.warning(f"🚫 HTF VETO: Daily BEARISH blocks LONG (confidence {raw_confidence:.1f} < 8)")
            position = "CASH"
            confidence_adjustment -= 2
        elif position == "SHORT" and trend_1d == "BULLISH" and raw_confidence < 8:
            logger.warning(f"🚫 HTF VETO: Daily BULLISH blocks SHORT (confidence {raw_confidence:.1f} < 8)")
            position = "CASH"
            confidence_adjustment -= 2
        
        # FILTER #2: Macro Regime Filter (RISK-OFF penalizes LONG)
        macro_regime = macro.get('regime', 'NEUTRAL')
        macro_score = macro.get('score', 0)
        if position == "LONG" and macro_regime == "RISK_OFF":
            logger.warning(f"⚠️ RISK-OFF: Reducing LONG confidence (macro score: {macro_score})")
            confidence_adjustment -= 2
        
        # FILTER #3: Dynamic Confidence Scaling (volatility-based)
        df = data.get('df')
        if df is not None and not df.empty:
            # Calculate ATR ratio for volatility
            atr = df['high'].rolling(14).mean() - df['low'].rolling(14).mean()
            current_atr = atr.iloc[-1]
            avg_atr = atr.mean()
            atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            if atr_ratio > 1.5:  # High volatility
                logger.info(f"⚡ HIGH VOLATILITY (ATR ratio: {atr_ratio:.2f}) - Raising bar")
                confidence_adjustment -= 1  # More strict
            elif atr_ratio < 0.7:  # Low volatility
                logger.info(f"😴 LOW VOLATILITY (ATR ratio: {atr_ratio:.2f}) - Easier signals")
                confidence_adjustment += 0  # Keep normal (already at 6)
        
        final_confidence = min(max(int(raw_confidence + confidence_adjustment), 1), 10)
        
        # AGGRESSIVE MODE: Ensure we take trades even if DeepSeek complains slightly
        if final_confidence < 5 and raw_confidence > 7:
             logger.info("⚠️ Overriding Validator: Raw confidence is strong enough!")
             final_confidence = 6
        
        # Risk Management & Reasoning
        entry_conditions = strategy if position != "CASH" else {}
        
        # Build detailed reasoning
        reasoning = self._build_reasoning_with_votes(
            data['macro'], data['chart'], data['news'], strategy, votes, validation
        )
        
        decision = DirectorDecision(
            symbol=symbol,
            position=position,
            confidence=final_confidence,
            votes=votes,
            reasoning=reasoning,
            entry_conditions=entry_conditions,
            risk_level=strategy.get('risk_level', 'MEDIUM')
        )
        
        # Populate professional trade details if valid signal
        if decision.position != "CASH":
            await self._populate_trade_details(decision, symbol, current_price, final_confidence)
            
        return decision

    async def _populate_trade_details(self, decision: DirectorDecision, symbol: str, current_price: float, confidence: int):
        """Populate stop loss, take profit, and position size"""
        try:
            # Get Stops from ATR
            from src.risk.position_sizer import ATRStopCalculator, KellyPositionSizer
            atr_calc = ATRStopCalculator()
            
            # Quick ATR calc (fetch new for accuracy or use existing df if possible, here fetch small)
            df = await self.binance.fetch_candles(symbol, limit=50) 
            atr = atr_calc.calculate_atr(df)
            
            stops = atr_calc.calculate_stops(
                current_price, atr, decision.position, multiplier=2.0
            )
            
            decision.stop_loss = stops['stop_loss']
            decision.take_profit = stops['take_profit']
            
            # Get Position Size from Kelly
            kelly = KellyPositionSizer()
            account_balance = await self.binance.get_balance()
            if account_balance < 10: account_balance = 1000 # Fallback
            
            perf_stats = self.tracker.get_performance_stats()
            win_rate = perf_stats.get('win_rate', 50) / 100
            
            # FIX 1.7: Use historical data
            if 'avg_win_pct' in perf_stats:
                avg_win = perf_stats['avg_win_pct']
                avg_loss = perf_stats['avg_loss_pct']
            else:
                avg_win = 0.03
                avg_loss = 0.015
            
            sizing = kelly.calculate_position_size(
                account_balance, win_rate, avg_win, avg_loss, confidence
            )
            
            decision.position_size = sizing['position_value'] / current_price
            
            # Update entry conditions for UI
            decision.entry_conditions.update({
                "entry_price": current_price,
                "stop_loss": decision.stop_loss,
                "target_1": decision.take_profit,
                "risk_reward": f"1:{stops['risk_reward_ratio']}",
                "conviction": confidence
            })
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            import traceback
            traceback.print_exc()
    
    def _build_reasoning_with_votes(self, macro, chart, news, strategy, votes, validation) -> str:
        """Create human-readable reasoning with vote details and validation"""
        parts = []
        
        # Show all votes
        parts.append("🗳️ AI VOTING:")
        for vote in votes:
            emoji = "🟢" if vote.vote == "BULLISH" else "🔴" if vote.vote == "BEARISH" else "⚪"
            # User-facing output (Turkish)
            
            vote_tr = {"BULLISH": "BULLISH", "BEARISH": "BEARISH", "NEUTRAL": "NEUTRAL", "MIXED": "MIXED"}.get(vote.vote, vote.vote)
            parts.append(f"{emoji} {vote.name}: {vote_tr} ({vote.confidence}/10)")
        
        # DeepSeek validation
        if validation.get('confidence_adjustment') != 0:
            parts.append(f"\n🔍 DEEPSEEK VALIDATION:")
            parts.append(f"  Confidence Adj: {validation.get('confidence_adjustment'):+d}")
            if validation.get('concerns'):
                parts.append(f"  {validation.get('concerns')[:150]}")
        
        parts.append("\n📊 DETAILED ANALYSIS:")
        
        # Macro
        parts.append(f"🌍 MACRO: {macro.get('regime', 'UNKNOWN')}")
        if macro.get('reasoning'):
            parts.append("  " + " | ".join(macro['reasoning'][:2]))
        
        # Chart (NOW WITH PROFESSIONAL TA!)
        parts.append(f"\n📈 TECHNICAL: {chart.get('trend', 'NONE')}")
        if chart.get('analysis'):
            parts.append(f"  {chart['analysis'][:200]}")
        
        # News
        parts.append(f"\n📰 NEWS (GPT-4): {news.get('sentiment', 'NONE')}")
        
        # Strategy
        parts.append(f"\n🧠 CLAUDE VERDICT:")
        parts.append(f"  {strategy.get('reasoning', 'NONE')[:200]}")
        
        return "\n".join(parts)
    async def _collect_votes_professional(self, macro, chart, news, price_action, 
                                     orderbook, funding, volume_profile, cvd, data) -> List[AIVote]:
        """
        Collect votes from ALL sources including PROFESSIONAL market signals
        """
        votes = []
        
        # 1. Macro Vote
        macro_score = macro.get('score', 0)
        if macro_score > 20:
            macro_vote = "BULLISH"
        elif macro_score < -20:
            macro_vote = "BEARISH"
        else:
            macro_vote = "NEUTRAL"
            
        macro_conf = min(abs(macro_score) // 10 + 5, 10)
        votes.append(AIVote(
            "Makro Beyin (VIX/DXY)",
            macro_vote,
            macro_conf,
            f"Skor: {macro_score} | {macro.get('regime', 'BİLİNMİYOR')}"
        ))
        
        # ** 2. GEMINI VISION - VISUAL CHART ANALYSIS (PRIMARY - 3 VOTES) **
        # This is now the MOST IMPORTANT signal - what we SEE on the chart
        gemini_analysis = await self._analyze_with_gemini_vision(data)
        if gemini_analysis['verdict'] != 'ERROR':
            # GET ADAPTIVE WEIGHT for Gemini Vision
            gemini_weight = self.adaptive_weights.get_adaptive_weight("Gemini Vision")
            num_gemini_votes = max(1, min(int(gemini_weight), 5))  # 1-5 votes based on performance
            
            base_confidence = gemini_analysis['confidence']
            
            for i in range(num_gemini_votes):  # DYNAMIC weight!
                vote_name = f"👁️ Gemini Vision #{i+1}"
                if i == 0:
                    vote_name = "👁️ Gemini Vision (Chart Pattern)"
                elif i == 1:
                    vote_name = "👁️ Gemini Vision (Volume)"
                elif i == 2:
                    vote_name = "👁️ Gemini Vision (Breakout)"
                
                votes.append(AIVote(
                    vote_name,
                    gemini_analysis['verdict'],
                    base_confidence if i == 0 else max(base_confidence - 1, 1),  # Slightly lower for duplicates
                    gemini_analysis['reasoning'][:100]
                ))
            
            logger.info(f"👁️ VISUAL ANALYSIS: {gemini_analysis['verdict']} ({base_confidence}/10) - {num_gemini_votes}x WEIGHT (adaptive)")
        else:
            # Fallback if vision fails
            votes.append(AIVote("👁️ Gemini Vision", "NEUTRAL", 5, "Vision analysis unavailable"))
        
        # 3. Technical Analysis Vote - PRICE ACTION FOCUS
        chart_trend = chart.get('trend', 'UNKNOWN')
        chart_strength = chart.get('strength', 0.5)
        trend_4h = chart.get('trend_4h', 'UNKNOWN')
        trend_1d = chart.get('trend_1d', 'UNKNOWN')
        
        if chart_trend == 'BULLISH':
            chart_vote = "BULLISH"
            chart_conf = min(int(chart_strength * 12) + 3, 10)
            
            # HTF Confirmation (SOFT, not blocking)
            if trend_1d == 'BULLISH': chart_conf = min(chart_conf + 1, 10)
            # REMOVED: Hard cap at 6 when 1d conflicts - let it through!
            
        elif chart_trend == 'BEARISH':
            chart_vote = "BEARISH"
            chart_conf = min(int(chart_strength * 12) + 3, 10)
            
            # HTF Confirmation (SOFT)
            if trend_1d == 'BEARISH': chart_conf = min(chart_conf + 1, 10)
            # REMOVED: Hard cap blocking logic
            
        else:
            chart_vote = "NEUTRAL"
            chart_conf = 5
            
        votes.append(AIVote(
            f"Teknik Analiz (RSI/MACD) [1d: {trend_1d}]",
            chart_vote,
            chart_conf,
            chart.get('analysis', 'Teknik analiz')[:100]
        ))
        
        # 3. Price Action Vote
        pa_signal = price_action.get('signal', 'NEUTRAL')
        if 'BULLISH' in pa_signal:
            pa_vote = "BULLISH"
            pa_conf = min(price_action.get('strength', 5) + 2, 10)
        elif 'BEARISH' in pa_signal:
            pa_vote = "BEARISH"
            pa_conf = min(price_action.get('strength', 5) + 2, 10)
        else:
            pa_vote = "NEUTRAL"
            pa_conf = 5
        
        indicators_text = " | ".join(price_action.get('indicators', [])[:2])
        votes.append(AIVote(
            "Price Action Detector",
            pa_vote,
            pa_conf,
            indicators_text[:100] if indicators_text else "No strong signals"
        ))
        
        # 4. ORDER BOOK IMBALANCE (NEW!)
        if orderbook.get('signal') != 'ERROR':
            votes.append(AIVote(
                "Order Book Imbalance",
                orderbook['signal'],
                orderbook['strength'],
                orderbook['reason'][:100]
            ))
        else:
            logger.warning(f"📊 Order Book FAILED: {orderbook.get('reason', 'Unknown error')}")
            votes.append(AIVote("Order Book Imbalance", "NEUTRAL", 5, "Data unavailable"))
        
        # 5. FUNDING RATE (NEW!)
        if funding.get('signal') != 'ERROR':
            votes.append(AIVote(
                "Funding Rate",
                funding['signal'],
                funding['strength'],
                funding['reason'][:100]
            ))
        else:
            logger.warning(f"💰 Funding Rate FAILED: {funding.get('reason', 'Unknown error')}")
            votes.append(AIVote("Funding Rate", "NEUTRAL", 5, "Data unavailable"))
        
        # 6. VOLUME PROFILE (NEW!)
        if volume_profile.get('signal') != 'ERROR':
            votes.append(AIVote(
                "Volume Profile (POC)",
                volume_profile['signal'],
                volume_profile['strength'],
                volume_profile['reason'][:100]
            ))
        else:
            logger.warning(f"📊 Volume Profile FAILED: {volume_profile.get('reason', 'Unknown error')}")
            votes.append(AIVote("Volume Profile (POC)", "NEUTRAL", 5, "Data unavailable"))
        
        # 7. CVD - Cumulative Volume Delta (NEW!)
        if cvd.get('signal') != 'ERROR':
            votes.append(AIVote(
                "CVD (Buy/Sell Pressure)",
                cvd['signal'],
                cvd['strength'],
                cvd['reason'][:100]
            ))
        else:
            logger.warning(f"📈 CVD FAILED: {cvd.get('reason', 'Unknown error')}")
            votes.append(AIVote("CVD (Buy/Sell Pressure)", "NEUTRAL", 5, "Data unavailable"))
        
        # 8. News Sentiment Vote
        news_sentiment = news.get('sentiment', 'NEUTRAL')
        news_conf = news.get('confidence', 5)
        
        votes.append(AIVote(
            "GPT-4 Haberler",
            news_sentiment,
            news_conf,
            news.get('summary', 'Haber analizi')[:100]
        ))
        
        if whale_data and isinstance(whale_data, dict) and whale_data.get('score', 0) > 0:
            score = whale_data.get('score', 5)
            sentiment = whale_data.get('sentiment', 'NEUTRAL')
            flow = whale_data.get('status', 'NEUTRAL')
            
            # Logic: Inflow = Bearish, Outflow = Bullish
            vote_type = "NEUTRAL"
            if flow == "INFLOW":
                vote_type = "BEARISH"
            elif flow == "OUTFLOW":
                vote_type = "BULLISH"
            elif sentiment == "BULLISH":
                vote_type = "BULLISH"
            elif sentiment == "BEARISH":
                vote_type = "BEARISH"
                
            confidence = whale_data.get('confidence', 5)
            
            # Special Weight: Whales are smart money
            if confidence >= 7:
                 confidence += 1
                 
            votes.append(AIVote("Whale Hunter", vote_type, confidence, f"Moves: {str(whale_data.get('moves', []))[:50]}"))

        # 9. LIQUIDATION HEATMAP (Short/Long Squeeze) - NEW
        if liq_data and isinstance(liq_data, dict):
             signal = liq_data.get('signal', 'NEUTRAL')
             risk_score = liq_data.get('risk_score', 2)
             magnet_price = liq_data.get('magnet_price')
             
             if signal == "MAGNET_UP":
                 # Price magnetically pulled UP to kill shorts -> Bullish
                 votes.append(AIVote("Liquidation Engine", "BULLISH", risk_score, f"Short Squeeze Magnet @ ${magnet_price:.2f}"))
             elif signal == "MAGNET_DOWN":
                 # Price magnetically pulled DOWN to kill longs -> Bearish
                 votes.append(AIVote("Liquidation Engine", "BEARISH", risk_score, f"Long Squeeze Magnet @ ${magnet_price:.2f}"))

        # 10. PRE-SPIKE DETECTOR (Early Warning) - NEW!
        if self.pre_spike:
            # Extract scores from existing signals
            volume_score = volume_profile.get('strength', 5)  # 0-10
            orderbook_score = orderbook.get('strength', 5)    # 0-10
            tape_score = 5  # Placeholder (would come from tape_reader if implemented)
            funding_score = min(abs(funding.get('rate_percent', 0)) * 2, 10) if funding.get('signal') != 'ERROR' else 5
            
            pre_spike_analysis = self.pre_spike.analyze(
                volume_score=volume_score,
                orderbook_score=orderbook_score,
                tape_score=tape_score,
                funding_score=funding_score,
                symbol=data.get('current_price', 'UNKNOWN')  # Symbol placeholder
            )
            
            # If PRE-PUMP detected, give it MASSIVE weight (it's an early warning!)
            if pre_spike_analysis['is_pre_pump']:
                signal_type = "BULLISH" if volume_score > 6 else "NEUTRAL"  # Direction based on volume
                confidence = pre_spike_analysis['confidence']
                
                logger.warning(f"🚨 PRE-PUMP DETECTED! Score: {pre_spike_analysis['combined_score']:.1f}/10 | Lead: {pre_spike_analysis['lead_time_estimate']}")
                
                # Give 3 votes for pre-pump (it's ACTIONABLE intelligence)
                for i in range(3):
                    votes.append(AIVote(
                        f"🚨 Pre-Pump Warning #{i+1}",
                        signal_type,
                        confidence,
                        f"Early Warning: {pre_spike_analysis['signal']} ({pre_spike_analysis['combined_score']:.1f}/10)"
                    ))
            elif pre_spike_analysis['signal'] == 'BUILDING':
                # BUILDING phase = 1 moderate vote
                votes.append(AIVote(
                    "🔍 Pre-Pump Building",
                    "NEUTRAL",
                    pre_spike_analysis['confidence'],
                    f"Momentum building ({pre_spike_analysis['combined_score']:.1f}/10)"
                ))

        return votes
    
    def provide_rl_feedback(self, trade_result: dict):
        """
        Provide feedback to RL Agent from a closed trade.
        
        This should be called by PositionManager or main.py when a trade closes.
        
        Args:
            trade_result: Dict from PositionManager.close_position() containing P&L, etc.
        """
        if not self.rl_agent or not trade_result:
            return
            
        # Use the market state at the time of decision
        market_state = getattr(self, 'current_market_state', 'LOW_VOL_RANGE')
        
        # Trigger learning
        self.rl_agent.learn_from_trade(trade_result, market_state)
        
        logger.info(f"🎯 RL Feedback provided: {trade_result.get('outcome')} ({trade_result.get('pnl_pct'):+.2f}%)")
