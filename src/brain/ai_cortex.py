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

class DirectorDecision:
    """
    The final output from the AI Cortex.
    """
    def __init__(self, symbol: str, position: str, reasoning: str, confidence: int, 
                 risk_level: str, entry_conditions: str):
        self.symbol = symbol
        self.position = position  # LONG, SHORT, CASH
        self.reasoning = reasoning
        self.confidence = confidence  # 1-10
        self.risk_level = risk_level  # HIGH, MEDIUM, LOW
        self.entry_conditions = entry_conditions

class AICortex:
    """
    The True AI Brain.
    Orchestrates 4 AI models to make trading decisions.
    """
    def __init__(self, binance: BinanceAPI):
        self.binance = binance
        self.macro = MacroBrain()
        self.chart_gen = ChartGenerator()
        self.gemini = GeminiVisionAnalyst()
        self.claude = ClaudeStrategist()
        self.news = NewsSentimentAnalyzer()
        
    async def think(self, symbol: str) -> DirectorDecision:
        """
        Main AI decision loop for a single symbol.
        """
        logger.info(f"🧠 AI Cortex thinking about {symbol}...")
        
        try:
            # 1. Gather Data in Parallel
            macro_task = self.macro.analyze_world()
            news_task = self.news.analyze_sentiment()
            chart_task = self._analyze_chart(symbol)
            
            macro_data, news_data, chart_analysis = await asyncio.gather(
                macro_task, news_task, chart_task
            )
            
            # 2. Claude Strategic Reasoning
            strategy = await self.claude.formulate_strategy(macro_data, chart_analysis, news_data)
            
            # 3. Formulate Final Decision
            position = strategy.get('position', 'CASH')
            reasoning = self._build_reasoning(macro_data, chart_analysis, news_data, strategy)
            confidence = self._calculate_confidence(macro_data, chart_analysis, news_data, strategy)
            
            decision = DirectorDecision(
                symbol=symbol,
                position=position,
                reasoning=reasoning,
                confidence=confidence,
                risk_level=strategy.get('risk_level', 'MEDIUM'),
                entry_conditions=strategy.get('entry_conditions', 'Wait for confirmation')
            )
            
            logger.info(f"✅ Decision for {symbol}: {position} (Confidence: {confidence}/10)")
            return decision
            
        except Exception as e:
            logger.error(f"AI Cortex error: {e}")
            return DirectorDecision(
                symbol=symbol,
                position="CASH",
                reasoning=f"Error in AI processing: {str(e)}",
                confidence=0,
                risk_level="HIGH",
                entry_conditions="System error - stay cash"
            )
    
    async def _analyze_chart(self, symbol: str) -> dict:
        """
        Generate chart and analyze with Gemini Vision.
        """
        try:
            # Fetch data
            df = await self.binance.fetch_candles(symbol, limit=100)
            if df.empty:
                return {"analysis": "No data", "trend": "UNKNOWN"}
            
            # Generate chart
            chart_path = self.chart_gen.generate_chart(symbol, df)
            if not chart_path:
                return {"analysis": "Chart generation failed", "trend": "UNKNOWN"}
            
            # Analyze with Gemini Vision
            analysis = await self.gemini.analyze_chart(chart_path, symbol)
            return analysis
            
        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
            return {"analysis": str(e), "trend": "ERROR"}
    
    def _build_reasoning(self, macro, chart, news, strategy) -> str:
        """
        Create human-readable reasoning from all AI outputs.
        """
        parts = []
        
        # Macro
        parts.append(f"🌍 MACRO: {macro.get('regime', 'UNKNOWN')} regime")
        if macro.get('reasoning'):
            parts.append(" | ".join(macro['reasoning'][:2]))
        
        # Chart
        parts.append(f"\n📊 CHART: {chart.get('trend', 'N/A')} trend")
        if chart.get('recommendation'):
            parts.append(f"Gemini says: {chart['recommendation']}")
        
        # News
        parts.append(f"\n📰 NEWS: {news.get('sentiment', 'N/A')} sentiment")
        
        # Strategy
        parts.append(f"\n🧠 CLAUDE DECISION: {strategy.get('reasoning', 'N/A')}")
        
        return "\n".join(parts)
    
    def _calculate_confidence(self, macro, chart, news, strategy) -> int:
        """
        Calculate overall confidence (1-10) based on AI agreement.
        """
        # If all AIs agree on direction, confidence is high
        macro_bullish = macro.get('score', 0) > 10
        chart_bullish = chart.get('trend', '') == 'BULLISH'
        news_bullish = news.get('sentiment', '') == 'BULLISH'
        
        position = strategy.get('position', 'CASH')
        
        agreement_count = 0
        if position == 'LONG':
            if macro_bullish: agreement_count += 1
            if chart_bullish: agreement_count += 1
            if news_bullish: agreement_count += 1
        elif position == 'SHORT':
            if not macro_bullish: agreement_count += 1
            if chart.get('trend') == 'BEARISH': agreement_count += 1
            if news.get('sentiment') == 'BEARISH': agreement_count += 1
        
        # Base confidence from strategy
        base = strategy.get('confidence', 5) if 'confidence' in strategy else 5
        
        # Boost if multiple AIs agree
        bonus = agreement_count * 2
        
        return min(base + bonus, 10)
