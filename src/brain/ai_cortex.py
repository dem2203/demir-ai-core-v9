import logging
import asyncio
from typing import Dict
from src.brain.macro import MacroBrain
from src.brain.technical_analyzer import TechnicalAnalyzer
from src.brain.price_action_detector import PriceActionDetector
from src.brain.market_microstructure import MarketMicrostructure
from src.brain.claude_strategist import ClaudeStrategist
from src.brain.news_sentiment import NewsSentimentAnalyzer
from src.brain.deepseek_validator import DeepSeekValidator
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
    def __init__(self, binance: BinanceAPI):
        self.binance = binance
        self.macro = MacroBrain()
        self.technical = TechnicalAnalyzer()
        self.price_action = PriceActionDetector()
        self.market_micro = MarketMicrostructure(binance)  # NEW: Professional signals
        self.claude = ClaudeStrategist()
        self.news = NewsSentimentAnalyzer()
        self.deepseek = DeepSeekValidator()
        
        # Consensus requirements
        self.MIN_CONSENSUS = 2  # At least 2/3 AIs must agree
        
        # Performance tracking for self-learning
        from src.utils.signal_tracker import SignalPerformanceTracker
        self.tracker = SignalPerformanceTracker()
        
    async def think(self, symbol: str) -> DirectorDecision:
        """
        PROFESSIONAL AI decision loop with MARKET MICROSTRUCTURE
        """
        logger.info(f"🧠 AI Cortex: Professional analizi başlatılıyor: {symbol}...")
        
        try:
            # 1. Gather ALL Data in Parallel (including professional signals)
            logger.info("📡 Tüm kaynaklardan veri çekiliyor (order book, funding, volume)...")
            
            macro_task = self.macro.analyze_world()
            news_task = self.news.analyze_sentiment()
            chart_task = self._analyze_chart_professional(symbol)
            
            # PROFESSIONAL CRYPTO SIGNALS (NEW!)
            df = await self.binance.fetch_candles(symbol, limit=100)
            
            orderbook_task = self.market_micro.analyze_orderbook_imbalance(symbol)
            funding_task = self.market_micro.analyze_funding_rate(symbol)
            volume_profile_task = self.market_micro.analyze_volume_profile(df)
            cvd_task = self.market_micro.analyze_cvd(df)
            
            # Gather all in parallel
            macro_data, news_data, chart_analysis, orderbook_data, funding_data, volume_profile, cvd_data = await asyncio.gather(
                macro_task, news_task, chart_task,
                orderbook_task, funding_task, volume_profile_task, cvd_task
            )
            
            # Log professional signals
            if orderbook_data['signal'] != 'ERROR':
                logger.info(f"📊 Order Book: {orderbook_data['signal']} ({orderbook_data['strength']}/10) - {orderbook_data['reason'][:50]}")
            if funding_data['signal'] != 'ERROR':
                logger.info(f"💰 Funding: {funding_data['signal']} ({funding_data['strength']}/10) - {funding_data['reason'][:50]}")
            if volume_profile['signal'] != 'ERROR':
                logger.info(f"📈 Volume Profile: {volume_profile['signal']} ({volume_profile['strength']}/10)")
            if cvd_data['signal'] != 'ERROR':
                logger.info(f"📊 CVD: {cvd_data['signal']} ({cvd_data['strength']}/10)")
            
            # 1.5 PRICE ACTION EARLY DETECTION
            price_action = self.price_action.analyze_price_action(df, symbol)
            
            if price_action['strength'] >= 7:
                logger.warning(f"🚨 EARLY SIGNAL: {price_action['signal']} (Strength: {price_action['strength']}/10)")
                for indicator in price_action['indicators']:
                    logger.info(f"   {indicator}")
            
            # 2. Collect votes (NOW INCLUDING PROFESSIONAL SIGNALS!)
            logger.info("🗳️ AI oyları toplanıyor (+ professional market signals)...")
            votes = self._collect_votes_professional(
                macro_data, chart_analysis, news_data, price_action,
                orderbook_data, funding_data, volume_profile, cvd_data
            )
            
            # 3. Claude Strategic Reasoning (WITH FEEDBACK)
            logger.info("🧠 Claude tüm girdileri analiz ediyor...")
            performance_feedback = self.tracker.get_ai_feedback_prompt()
            strategy = await self.claude.formulate_strategy(macro_data, chart_analysis, news_data, performance_feedback)
            
            # Add Claude's vote
            claude_vote = self._extract_claude_vote(strategy)
            votes.append(claude_vote)
            
            # Store as instance variable for use in other methods
            self.votes = votes
            
            # 4. DeepSeek Cross-Validation
            logger.info("🔍 DeepSeek kararları doğruluyor...")
            validation = await self.deepseek.validate(votes, chart_analysis, macro_data)
            
            # DeepSeek penalty reduction (was too harsh at -4, now max -2)
            if validation.get('confidence_adjustment', 0) < -2:
                logger.warning(f"⚠️ DeepSeek penalty reduced from {validation['confidence_adjustment']} to -2 (was too harsh)")
                validation['confidence_adjustment'] = -2
            
            # 5. Calculate consensus
            consensus_result = self._calculate_consensus(votes)
            
            position = consensus_result['position']
            confidence = consensus_result['confidence']
            
            # Apply DeepSeek confidence adjustment
            if validation.get('confidence_adjustment'):
                confidence += validation['confidence_adjustment']
                confidence = max(1, min(10, confidence))
            
            # DeepSeek rejection
            if validation.get('rejected'):
                position = "CASH"
                confidence = 3
                logger.warning(f"⚠️ DeepSeek rejected decision: {validation.get('concerns')}")
            
            # 6. CALCULATE POSITION SIZE & STOPS (NEW!)
            stop_loss, take_profit, position_size = None, None, None
            
            if position in ["LONG", "SHORT"] and confidence >= 6:
                # Get performance stats for Kelly
                perf_stats = self.tracker.get_performance_stats()
                
                # Calculate ATR-based stops
                from src.risk.position_sizer import ATRStopCalculator
                atr_calc = ATRStopCalculator()
                atr = atr_calc.calculate_atr(df)
                current_price = df['close'].iloc[-1]
                
                stops = atr_calc.calculate_stops(
                    entry_price=current_price,
                    atr=atr,
                    direction=position,
                    multiplier=2.0
                )
                
                stop_loss = stops['stop_loss']
                take_profit = stops['take_profit']
                
                # Calculate Kelly position size
                if 'message' not in perf_stats:
                    from src.risk.position_sizer import KellyPositionSizer
                    kelly = KellyPositionSizer()
                    
                    # Dummy account balance - will be read from config/binance later
                    account_balance = 1000  # USDT
                    
                    win_rate = perf_stats['win_rate'] / 100
                    avg_win_pct = 0.03  # 3% average win
                    avg_loss_pct = 0.015  # 1.5% average loss (2:1 R:R)
                    
                    sizing = kelly.calculate_position_size(
                        account_balance=account_balance,
                        win_rate=win_rate,
                        avg_win_pct=avg_win_pct,
                        avg_loss_pct=avg_loss_pct,
                        current_confidence=confidence
                    )
                    
                    position_size = sizing['position_value'] / current_price  # Convert to quantity
            
            # Build detailed reasoning with validation
            reasoning = self._build_reasoning_with_votes(macro_data, chart_analysis, news_data, strategy, votes, validation)
            
            decision = DirectorDecision(
                symbol=symbol,
                position=position,
                reasoning=reasoning,
                confidence=confidence,
                risk_level=strategy.get('risk_level', 'MEDIUM'),
                entry_conditions=strategy.get('entry_conditions', 'Wait for confirmation'),
                votes=votes,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size
            )
            
            # Log consensus + RISK MANAGEMENT
            logger.info(f"\n{decision.get_consensus_report()}")
            logger.info(f"✅ Final Decision: {position} (Confidence: {confidence}/10)")
            if stop_loss and take_profit:
                logger.info(f"🎯 Risk Management: SL=${stop_loss:.2f} | TP=${take_profit:.2f}")
            if position_size:
                logger.info(f"💰 Position Size: {position_size:.4f} {symbol[:3]}")
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
    
    def _collect_votes(self, macro, chart, news, price_action) -> list:
        """Collect votes from Macro, Technical Analysis, News AI, and Price Action"""
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
        
        # 2. Technical Analysis Vote
        chart_trend = chart.get('trend', 'UNKNOWN')
        chart_strength = chart.get('strength', 0.5)
        
        if chart_trend == 'BULLISH':
            chart_vote = "BULLISH"
            # Boost confidence for clear technical signals
            chart_conf = max(int(chart_strength * 10), 6)  # Minimum 6 for clear trend
        elif chart_trend == 'BEARISH':
            chart_vote = "BEARISH"
            chart_conf = max(int(chart_strength * 10), 6)  # Minimum 6 for clear trend
        else:
            chart_vote = "NEUTRAL"
            chart_conf = 5
            
        votes.append(AIVote(
            "Teknik Analiz (RSI/MACD)",
            chart_vote,
            chart_conf,
            chart.get('analysis', 'Teknik analiz')[:100]
        ))
        
        # 3. Price Action Vote (NEW!)
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
        
        # 4. News Sentiment Vote
        news_sentiment = news.get('sentiment', 'NEUTRAL')
        news_conf = news.get('confidence', 5)
        
        votes.append(AIVote(
            "GPT-4 Haberler",
            news_sentiment,
            news_conf,
            news.get('summary', 'Haber analizi')[:100]
        ))
        
        return votes
    
    async def _analyze_chart_professional(self, symbol: str) -> dict:
        """
        Professional chart analysis using technical indicators
        """
        try:
            df = await self.binance.fetch_candles(symbol, limit=100)
            if df.empty:
                return {"trend": "UNKNOWN", "analysis": "Veri yok"}
            
            # Use professional TA
            analysis = self.technical.analyze(df, symbol)
            return analysis
            
        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
            return {"trend": "ERROR", "analysis": str(e)}
    
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
            "Claude Stratejist",  # Turkish name
            vote,
            confidence,
            strategy.get('reasoning', 'Stratejik analiz')[:100]
        )
    
    def _calculate_consensus(self, votes: list) -> dict:
        """Calculate consensus from all AI votes"""
        bullish_count = sum(1 for v in votes if v.vote == "BULLISH")
        bearish_count = sum(1 for v in votes if v.vote == "BEARISH")
        neutral_count = sum(1 for v in votes if v.vote == "NEUTRAL")
        
        total_votes = len(votes)
        
        # Determine position based on majority
        if bullish_count >= self.MIN_CONSENSUS:
            position = "LONG"
            confidence = min((bullish_count / total_votes) * 10, 10)
        elif bearish_count >= self.MIN_CONSENSUS:
            position = "SHORT"
            confidence = min((bearish_count / total_votes) * 10, 10)
        else:
            position = "CASH"
            confidence = 3  # Low confidence, no consensus
            
        # Boost confidence if unanimous
        if bullish_count == total_votes or bearish_count == total_votes:
            confidence = 10
            
        return {
            'position': position,
            'confidence': int(confidence),
            'bullish': bullish_count,
            'bearish': bearish_count,
            'neutral': neutral_count
        }
    
    def _build_reasoning_with_votes(self, macro, chart, news, strategy, votes, validation) -> str:
        """Create human-readable reasoning with vote details and validation (TURKISH)"""
        parts = []
        
        # Show all votes
        parts.append("🗳️ YAPAY ZEKA OYLAMASI:")
        for vote in votes:
            emoji = "🟢" if vote.vote == "BULLISH" else "🔴" if vote.vote == "BEARISH" else "⚪"
            vote_tr = {"BULLISH": "YÜKSELİŞ", "BEARISH": "DÜŞÜŞ", "NEUTRAL": "NÖTR", "MIXED": "KARIŞIK"}.get(vote.vote, vote.vote)
            parts.append(f"{emoji} {vote.name}: {vote_tr} ({vote.confidence}/10)")
        
        # DeepSeek validation
        if validation.get('confidence_adjustment') != 0:
            parts.append(f"\n🔍 DEEPSEEK DOĞRULAMA:")
            parts.append(f"  Güven Ayarı: {validation.get('confidence_adjustment'):+d}")
            if validation.get('concerns'):
                parts.append(f"  {validation.get('concerns')[:150]}")
        
        parts.append("\n📊 DETAYLI ANALİZ:")
        
        # Macro
        parts.append(f"🌍 MAKRO: {macro.get('regime', 'BİLİNMİYOR')}")
        if macro.get('reasoning'):
            parts.append("  " + " | ".join(macro['reasoning'][:2]))
        
        # Chart (NOW WITH PROFESSIONAL TA!)
        parts.append(f"\n📈 TEKNİK ANALİZ: {chart.get('trend', 'YOK')}")
        if chart.get('analysis'):
            parts.append(f"  {chart['analysis'][:200]}")
        
        # News
        parts.append(f"\n📰 HABERLER (GPT-4): {news.get('sentiment', 'YOK')}")
        
        # Strategy
        parts.append(f"\n🧠 CLAUDE SON KARAR:")
        parts.append(f"  {strategy.get('reasoning', 'YOK')[:200]}")
        
        return "\n".join(parts)
    def _collect_votes_professional(self, macro, chart, news, price_action, 
                                     orderbook, funding, volume_profile, cvd) -> list:
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
        
        # 2. Technical Analysis Vote
        chart_trend = chart.get('trend', 'UNKNOWN')
        chart_strength = chart.get('strength', 0.5)
        
        if chart_trend == 'BULLISH':
            chart_vote = "BULLISH"
            chart_conf = max(int(chart_strength * 10), 6)
        elif chart_trend == 'BEARISH':
            chart_vote = "BEARISH"
            chart_conf = max(int(chart_strength * 10), 6)
        else:
            chart_vote = "NEUTRAL"
            chart_conf = 5
            
        votes.append(AIVote(
            "Teknik Analiz (RSI/MACD)",
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
        
        # 5. FUNDING RATE (NEW!)
        if funding.get('signal') != 'ERROR':
            votes.append(AIVote(
                "Funding Rate",
                funding['signal'],
                funding['strength'],
                funding['reason'][:100]
            ))
        
        # 6. VOLUME PROFILE (NEW!)
        if volume_profile.get('signal') != 'ERROR':
            votes.append(AIVote(
                "Volume Profile (POC)",
                volume_profile['signal'],
                volume_profile['strength'],
                volume_profile['reason'][:100]
            ))
        
        # 7. CVD - Cumulative Volume Delta (NEW!)
        if cvd.get('signal') != 'ERROR':
            votes.append(AIVote(
                "CVD (Buy/Sell Pressure)",
                cvd['signal'],
                cvd['strength'],
                cvd['reason'][:100]
            ))
        
        # 8. News Sentiment Vote
        news_sentiment = news.get('sentiment', 'NEUTRAL')
        news_conf = news.get('confidence', 5)
        
        votes.append(AIVote(
            "GPT-4 Haberler",
            news_sentiment,
            news_conf,
            news.get('summary', 'Haber analizi')[:100]
        ))
        
        return votes
