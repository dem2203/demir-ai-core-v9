import logging
import asyncio
from typing import Dict
from src.brain.macro import MacroBrain
from src.brain.chart_generator import ChartGenerator
from src.brain.vision_analyst import GeminiVisionAnalyst
from src.brain.claude_strategist import ClaudeStrategist
from src.brain.news_sentiment import NewsSentimentAnalyzer
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
                 risk_level: str, entry_conditions: str, votes: list):
        self.symbol = symbol
        self.position = position  # LONG, SHORT, CASH
        self.reasoning = reasoning
        self.confidence = confidence  # 1-10
        self.risk_level = risk_level  # HIGH, MEDIUM, LOW
        self.entry_conditions = entry_conditions
        self.votes = votes  # List of AIVote objects
        
    def get_consensus_report(self) -> str:
        """Generate human-readable consensus report"""
        lines = ["🗳️ AI VOTING RESULTS:"]
        lines.append("━━━━━━━━━━━━━━━━━━")
        
        for vote in self.votes:
            emoji = "🟢" if vote.vote == "BULLISH" else "🔴" if vote.vote == "BEARISH" else "⚪"
            lines.append(f"{emoji} {vote.name}: {vote.vote} ({vote.confidence}/10)")
        
        # Count votes
        bullish = sum(1 for v in self.votes if v.vote == "BULLISH")
        bearish = sum(1 for v in self.votes if v.vote == "BEARISH")
        neutral = sum(1 for v in self.votes if v.vote == "NEUTRAL")
        
        lines.append(f"\n📊 Consensus: {bullish} BULL | {bearish} BEAR | {neutral} NEUTRAL")
        return "\n".join(lines)

class AICortex:
    """
    The True AI Brain with 4-AI Voting System
    """
    def __init__(self, binance: BinanceAPI):
        self.binance = binance
        self.macro = MacroBrain()
        self.chart_gen = ChartGenerator()
        self.gemini = GeminiVisionAnalyst()
        self.claude = ClaudeStrategist()
        self.news = NewsSentimentAnalyzer()
        
        # Consensus requirements
        self.MIN_CONSENSUS = 3  # At least 3/4 AIs must agree
        
        # Performance tracking for self-learning
        from src.utils.signal_tracker import SignalPerformanceTracker
        self.tracker = SignalPerformanceTracker()
        
    async def think(self, symbol: str) -> DirectorDecision:
        """
        Main AI decision loop with voting consensus.
        """
        logger.info(f"🧠 AI Cortex: Starting 4-AI analysis for {symbol}...")
        
        try:
            # 1. Gather Data in Parallel
            logger.info("📡 Fetching data from all sources...")
            macro_task = self.macro.analyze_world()
            news_task = self.news.analyze_sentiment()
            chart_task = self._analyze_chart(symbol)
            
            macro_data, news_data, chart_analysis = await asyncio.gather(
                macro_task, news_task, chart_task
            )
            
            # 2. Get individual AI votes
            logger.info("🗳️ Collecting AI votes...")
            votes = self._collect_votes(macro_data, chart_analysis, news_data)
            
            # 3. Claude Strategic Reasoning (Final arbiter) - WITH PERFORMANCE FEEDBACK
            logger.info("🧠 Claude analyzing all inputs...")
            performance_feedback = self.tracker.get_ai_feedback_prompt()
            strategy = await self.claude.formulate_strategy(macro_data, chart_analysis, news_data, performance_feedback)
            
            # Add Claude's vote
            claude_vote = self._extract_claude_vote(strategy)
            votes.append(claude_vote)
            
            # 4. Calculate consensus
            consensus_result = self._calculate_consensus(votes)
            
            # 5. Formulate Final Decision
            position = consensus_result['position']
            confidence = consensus_result['confidence']
            
            # Build detailed reasoning
            reasoning = self._build_reasoning_with_votes(macro_data, chart_analysis, news_data, strategy, votes)
            
            decision = DirectorDecision(
                symbol=symbol,
                position=position,
                reasoning=reasoning,
                confidence=confidence,
                risk_level=strategy.get('risk_level', 'MEDIUM'),
                entry_conditions=strategy.get('entry_conditions', 'Wait for confirmation'),
                votes=votes
            )
            
            # Log consensus
            logger.info(f"\n{decision.get_consensus_report()}")
            logger.info(f"✅ Final Decision: {position} (Confidence: {confidence}/10)\n")
            
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
    
    def _collect_votes(self, macro, chart, news) -> list:
        """Collect votes from Macro, Gemini Vision, and News AI"""
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
            "Macro Brain (VIX/DXY)",
            macro_vote,
            macro_conf,
            f"Score: {macro_score} | {macro.get('regime', 'UNKNOWN')}"
        ))
        
        # 2. Gemini Vision Vote
        gemini_trend = chart.get('trend', 'UNKNOWN')
        if gemini_trend == 'BULLISH':
            gemini_vote = "BULLISH"
            gemini_conf = 8
        elif gemini_trend == 'BEARISH':
            gemini_vote = "BEARISH"
            gemini_conf = 8
        else:
            gemini_vote = "NEUTRAL"
            gemini_conf = 5
            
        votes.append(AIVote(
            "Gemini Vision",
            gemini_vote,
            gemini_conf,
            chart.get('analysis', 'Chart analysis')[:100]
        ))
        
        # 3. News Sentiment Vote
        news_sentiment = news.get('sentiment', 'NEUTRAL')
        news_conf = news.get('confidence', 5)
        
        votes.append(AIVote(
            "GPT-4 News",
            news_sentiment,
            news_conf,
            news.get('summary', 'News analysis')[:100]
        ))
        
        return votes
    
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
            "Claude Strategist",
            vote,
            confidence,
            strategy.get('reasoning', 'Strategic analysis')[:100]
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
    
    async def _analyze_chart(self, symbol: str) -> dict:
        """Generate chart and analyze with Gemini Vision."""
        try:
            df = await self.binance.fetch_candles(symbol, limit=100)
            if df.empty:
                return {"analysis": "No data", "trend": "UNKNOWN"}
            
            chart_path = self.chart_gen.generate_chart(symbol, df)
            if not chart_path:
                return {"analysis": "Chart generation failed", "trend": "UNKNOWN"}
            
            analysis = await self.gemini.analyze_chart(chart_path, symbol)
            return analysis
            
        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
            return {"analysis": str(e), "trend": "ERROR"}
    
    def _build_reasoning_with_votes(self, macro, chart, news, strategy, votes) -> str:
        """Create human-readable reasoning with vote details"""
        parts = []
        
        # Show all votes
        parts.append("🗳️ AI VOTING:")
        for vote in votes:
            emoji = "🟢" if vote.vote == "BULLISH" else "🔴" if vote.vote == "BEARISH" else "⚪"
            parts.append(f"{emoji} {vote.name}: {vote.vote} ({vote.confidence}/10)")
        
        parts.append("\n📊 DETAILED ANALYSIS:")
        
        # Macro
        parts.append(f"🌍 MACRO: {macro.get('regime', 'UNKNOWN')}")
        if macro.get('reasoning'):
            parts.append("  " + " | ".join(macro['reasoning'][:2]))
        
        # Chart
        parts.append(f"\n📈 CHART (Gemini Vision): {chart.get('trend', 'N/A')}")
        if chart.get('analysis'):
            parts.append(f"  {chart['analysis'][:150]}")
        
        # News
        parts.append(f"\n📰 NEWS (GPT-4): {news.get('sentiment', 'N/A')}")
        
        # Strategy
        parts.append(f"\n🧠 CLAUDE FINAL VERDICT:")
        parts.append(f"  {strategy.get('reasoning', 'N/A')[:200]}")
        
        return "\n".join(parts)
