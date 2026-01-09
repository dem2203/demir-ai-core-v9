import logging
import asyncio
from typing import Dict
from src.brain.macro import MacroBrain
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
        lines = ["🗳️ YAPAY ZEKA OY SONUÇLARI:"]
        lines.append("━━━━━━━━━━━━━━━━━━")
        
        for vote in self.votes:
            emoji = "🟢" if vote.vote == "BULLISH" else "🔴" if vote.vote == "BEARISH" else "⚪"
            lines.append(f"{emoji} {vote.name}: {vote.vote} ({vote.confidence}/10)")
        
        # Count votes
        bullish = sum(1 for v in self.votes if v.vote == "BULLISH")
        bearish = sum(1 for v in self.votes if v.vote == "BEARISH")
        neutral = sum(1 for v in self.votes if v.vote == "NEUTRAL")
        
        lines.append(f"\n📊 Konsensus: {bullish} YÜKSELİŞ | {bearish} DÜŞÜŞ | {neutral} NÖTR")
        return "\n".join(lines)

class AICortex:
    """
    4-AI Voting System (Gemini Vision REMOVED)
    """
    def __init__(self, binance: BinanceAPI):
        self.binance = binance
        self.macro = MacroBrain()
        self.claude = ClaudeStrategist()
        self.news = NewsSentimentAnalyzer()
        self.deepseek = DeepSeekValidator()  # Cross validator
        
        # Consensus requirements (now 3 AIs + Claude)
        self.MIN_CONSENSUS = 2  # At least 2/3 AIs must agree (lowered from 3)
        
        # Performance tracking for self-learning
        from src.utils.signal_tracker import SignalPerformanceTracker
        self.tracker = SignalPerformanceTracker()
        
    async def think(self, symbol: str) -> DirectorDecision:
        """
        Main AI decision loop with 4-AI voting (NO GEMINI).
        """
        logger.info(f"🧠 AI Cortex: 4-AI analizi başlatılıyor: {symbol}...")
        
        try:
            # 1. Gather Data in Parallel (NO CHART)
            logger.info("📡 Tüm kaynaklardan veri çekiliyor...")
            macro_task = self.macro.analyze_world()
            news_task = self.news.analyze_sentiment()
            
            macro_data, news_data = await asyncio.gather(
                macro_task, news_task
            )
            
            # 2. Get individual AI votes (NO GEMINI)
            logger.info("🗳️ AI oyları toplanıyor...")
            votes = self._collect_votes(macro_data, news_data)
            
            # 3. Claude Strategic Reasoning (WITH FEEDBACK)
            logger.info("🧠 Claude tüm girdileri analiz ediyor...")
            performance_feedback = self.tracker.get_ai_feedback_prompt()
            strategy = await self.claude.formulate_strategy(macro_data, {}, news_data, performance_feedback)
            
            # Add Claude's vote
            claude_vote = self._extract_claude_vote(strategy)
            votes.append(claude_vote)
            
            # 4. DeepSeek Cross-Validation (NO CHART DATA)
            logger.info("🔍 DeepSeek kararları doğruluyor...")
            validation = await self.deepseek.validate(votes, {}, macro_data)
            
            # 5. Calculate consensus
            consensus_result = self._calculate_consensus(votes)
            
            # Apply DeepSeek confidence adjustment
            raw_confidence = consensus_result['confidence']
            adjusted_confidence = max(1, min(10, raw_confidence + validation.get('confidence_adjustment', 0)))
            
            # 6. Formulate Final Decision
            position = consensus_result['position']
            confidence = adjusted_confidence
            
            # If DeepSeek flagged as invalid, force CASH
            if not validation.get('is_valid', True):
                position = "CASH"
                confidence = 3
                logger.warning(f"⚠️ DeepSeek rejected decision: {validation.get('concerns')}")
            
            # Build detailed reasoning with validation
            reasoning = self._build_reasoning_with_votes(macro_data, news_data, strategy, votes, validation)
            
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
    
    def _collect_votes(self, macro, news) -> list:
        """Collect votes from Macro and News AI (NO GEMINI)"""
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
        
        # 2. News Sentiment Vote
        news_sentiment = news.get('sentiment', 'NEUTRAL')
        news_conf = news.get('confidence', 5)
        
        votes.append(AIVote(
            "GPT-4 Haberler",
            news_sentiment,
            news_conf,
            news.get('summary', 'Haber analizi')[:100]
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
    
    def _build_reasoning_with_votes(self, macro, news, strategy, votes, validation) -> str:
        """Create human-readable reasoning with vote details and validation (TURKISH)"""
        parts = []
        
        # Show all votes
        parts.append("🗳️ YAPAY ZEKA OYLAMASI:")
        for vote in votes:
            emoji = "🟢" if vote.vote == "BULLISH" else "🔴" if vote.vote == "BEARISH" else "⚪"
            parts.append(f"{emoji} {vote.name}: {vote.vote} ({vote.confidence}/10)")
        
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
        
        # News (NO CHART SECTION)
        parts.append(f"\n📰 HABERLER (GPT-4): {news.get('sentiment', 'YOK')}")
        
        # Strategy
        parts.append(f"\n🧠 CLAUDE SON KARAR:")
        parts.append(f"  {strategy.get('reasoning', 'YOK')[:200]}")
        
        return "\n".join(parts)
