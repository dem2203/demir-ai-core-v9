import anthropic
import logging
from src.config import Config

logger = logging.getLogger("CLAUDE_STRATEGIST")

class ClaudeStrategist:
    """
    Uses Claude for deep macro reasoning and strategy formulation.
    """
    def __init__(self):
        if Config.ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        else:
            self.client = None
            logger.warning("⚠️ Claude API Key missing. Strategic reasoning disabled.")
    
    async def formulate_strategy(self, macro_data: dict, chart_analysis: dict, news_sentiment: dict) -> dict:
        """
        Ask Claude to act as a hedge fund manager and formulate trading strategy.
        """
        if not self.client:
            return {"directive": "NEUTRAL", "reasoning": "Claude unavailable"}
            
        try:
            # Build context
            context = f"""**Market Macro Data:**
- VIX (Fear Index): {macro_data.get('vix', 'N/A')}
- DXY (Dollar Index): {macro_data.get('dxy', 'N/A')}
- BTC Dominance: {macro_data.get('btc_dominance', 'N/A')}%
- Market Regime: {macro_data.get('regime', 'UNKNOWN')}

**Chart Technical Analysis (Gemini Vision):**
{chart_analysis.get('analysis', 'No analysis available')}

**News Sentiment:**
{news_sentiment.get('summary', 'No news data')}
Sentiment Score: {news_sentiment.get('sentiment', 'NEUTRAL')}"""

            prompt = f"""You are a professional hedge fund manager specializing in crypto trading. 

{context}

Based on this data, provide your strategic directive for trading BTC/ETH:

1. **Overall Market Assessment**: What is the current market environment? (Risk-On/Risk-Off/Uncertain)
2. **Recommended Position**: Should we be LONG, SHORT, or stay in CASH?
3. **Risk Level**: HIGH, MEDIUM, or LOW risk environment?
4. **Reasoning**: Explain your logic step by step (macro → sentiment → technical → conclusion)
5. **Entry Conditions**: What specific conditions must be met before entering a trade?
6. **Stop Loss Strategy**: Where should protective stops be placed?

Respond in JSON format with keys: assessment, position, risk_level, reasoning, entry_conditions, stop_strategy"""

            # Call Claude
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text
            
            # Parse JSON
            try:
                import json
                if "{" in response_text and "}" in response_text:
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    result = json.loads(response_text[start:end])
                else:
                    result = {"reasoning": response_text, "position": "WAIT"}
            except:
                result = {"reasoning": response_text, "position": "WAIT"}
                
            logger.info(f"🧠 Claude Strategy: {result.get('position', 'WAIT')} - {result.get('assessment', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {"directive": "ERROR", "reasoning": str(e)}
