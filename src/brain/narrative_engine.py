import logging
import os
from datetime import datetime
from typing import Dict
from src.config.settings import Config

# Try importing OpenAI, handle if missing
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger("NARRATIVE_ENGINE")

class NarrativeEngine:
    """
    PHASE 18: NARRATIVE ENGINE
    Uses GPT-4o to generate human-readable explanations ("The Why") 
    for AI trading decisions.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key and OpenAI:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("📝 Narrative Engine: CONNECTED (GPT-4o)")
            except Exception as e:
                logger.error(f"Narrative Engine connection failed: {e}")
        else:
            logger.warning("⚠️ OpenAI API Key or library missing. Narrative Engine disabled.")
            
    def generate_explanation(self, symbol: str, action: str, brain_state: Dict) -> str:
        """
        Generate a natural language explanation for the trade decision.
        """
        if not self.client:
            return "Analysis not available (GPT-4o disabled)."
            
        try:
            # Construct context prompt
            context = f"""
            Analyze this trading setup for {symbol} and explain WHY the decision is {action}.
            
            MARKET DATA:
            - Price: ${brain_state.get('price', 0)}
            - RSI: {brain_state.get('rsi', 'N/A')}
            - MACD: {brain_state.get('macd_hist', 'N/A')}
            - Trend: {brain_state.get('trend_bias', 'N/A')} (HTF: {brain_state.get('htf_direction', 'N/A')})
            - Volume: {brain_state.get('volume_flow', 'N/A')}
            
            AI BRAIN:
            - Pattern: {brain_state.get('pattern', 'None')}
            - Sentiment: {brain_state.get('sentiment', 'Neutral')} (F&G: {brain_state.get('fear_greed', 50)})
            - Macro Score: {brain_state.get('macro_score', 0)}
            - AI Confidence: {brain_state.get('ai_confidence', 0)}%
            
            Write a single, professional paragraph (max 2 sentences) explaining the rationale. 
            Tone: Analytical hedge fund manager.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an elite crypto trading AI explaining your moves."},
                    {"role": "user", "content": context}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate narrative: {e}")
            return "Explanation generation failed."
