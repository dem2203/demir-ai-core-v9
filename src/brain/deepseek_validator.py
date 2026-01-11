import logging
from openai import AsyncOpenAI  # FIX 2.3: Use AsyncOpenAI
from src.config import Config

logger = logging.getLogger("DEEPSEEK_VALIDATOR")

class DeepSeekValidator:
    """
    DeepSeek Cross-Validator - Checks other AIs for hallucinations and errors.
    Acts as the final safety check before trading decisions.
    """
    def __init__(self):
        if Config.DEEPSEEK_API_KEY:
            # FIX 2.3: Use AsyncOpenAI to prevent blocking event loop
            self.client = AsyncOpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com"
            )
        else:
            self.client = None
            logger.warning("⚠️ DeepSeek API Key missing. Cross-validation disabled.")
    
    async def validate(self, ai_votes: list, chart_data: dict, macro_data: dict) -> dict:
        """
        Cross-validate other AI decisions.
        Returns: {is_valid: bool, concerns: str, confidence_adjustment: int}
        """
        if not self.client:
            return {
                "is_valid": True,
                "concerns": "DeepSeek unavailable",
                "confidence_adjustment": 0
            }
        
        try:
            # Build context
            votes_summary = "\n".join([
                f"- {vote.name}: {vote.vote} ({vote.confidence}/10)"
                for vote in ai_votes
            ])
            
            prompt = f"""You are DeepSeek, an AI validator checking other AIs for hallucinations and errors.

**Other AI Votes:**
{votes_summary}

**Chart Analysis:**
Trend: {chart_data.get('trend', 'N/A')}
Analysis: {chart_data.get('analysis', 'N/A')[:300]}

**Macro Data:**
VIX: {macro_data.get('vix', 'N/A')}
DXY: {macro_data.get('dxy', 'N/A')}
Regime: {macro_data.get('regime', 'N/A')}

**Your Task:**
1. Do you see any obvious hallucinations or errors in the AI votes?
2. Is there a mismatch between chart analysis and votes?
3. Are the AIs being too optimistic or pessimistic given the data?
4. Should we adjust confidence UP (+1 to +3) or DOWN (-1 to -3)?

Respond in JSON: {{"is_valid": true/false, "concerns": "your concerns", "confidence_adjustment": -3 to +3}}"""

            response = await self.client.chat.completions.create(  # FIX 2.3: await async call
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a critical AI validator. Be skeptical."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON
            try:
                import json
                if "{" in response_text and "}" in response_text:
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    result = json.loads(response_text[start:end])
                else:
                    result = {
                        "is_valid": True,
                        "concerns": response_text,
                        "confidence_adjustment": 0
                    }
            except:
                result = {
                    "is_valid": True,
                    "concerns": response_text,
                    "confidence_adjustment": 0
                }
            
            logger.info(f"🔍 DeepSeek Validation: Valid={result.get('is_valid')} | Adjustment={result.get('confidence_adjustment')}")
            
            return result
            
        except Exception as e:
            logger.error(f"DeepSeek validation error: {e}")
            return {
                "is_valid": True,
                "concerns": f"Error: {str(e)}",
                "confidence_adjustment": 0
            }
