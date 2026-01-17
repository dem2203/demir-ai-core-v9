"""
Grok Sentiment Analyzer - X/Twitter Real-Time Social Sentiment

Uses xAI's Grok API to analyze:
- X/Twitter trending topics
- Whale activity mentions
- Crypto influencer sentiment
- FOMO/FUD detection
- Breaking news (real-time)

Key Advantage: 0-5 second latency (vs 15-30 min traditional news)
"""

import logging
from typing import Dict
from openai import OpenAI  # xAI uses OpenAI SDK
import os

logger = logging.getLogger("GROK_SENTIMENT")

class GrokSentimentAnalyzer:
    """
    Real-time social sentiment analysis using Grok (xAI).
    
    Features:
    - X/Twitter sentiment tracking
    - FOMO/FUD detection
    - Whale activity mentions
    - Influencer sentiment
    - Breaking news alerts
    
    Lead Time: 30s-5min before traditional news
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Grok client.
        
        Args:
            api_key: xAI API key (from console.x.ai)
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        
        if not self.api_key:
            logger.warning("⚠️ Grok API key not found. Social sentiment disabled.")
            self.client = None
        else:
            try:
                # xAI uses OpenAI SDK with custom base URL
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1"
                )
                logger.info("🤖 Grok Sentiment Analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Grok: {e}")
                self.client = None
    
    async def run_custom_prompt(self, prompt: str) -> Dict:
        """
        Execute a custom prompt using the Grok API.
        Useful for specialized tasks like Whale Analysis.
        """
        if not self.client:
            return {}
            
        try:
            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {"role": "system", "content": "You are a crypto analytics AI. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            
            # Simple parsing
            import json
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Grok custom prompt error: {e}")
            return {}

    def analyze_social_sentiment(self, symbol: str) -> Dict:
        """
        Analyze X/Twitter sentiment for a crypto symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
        
        Returns:
            {
                "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
                "confidence": 0-10,
                "fomo_score": 0-10,
                "fud_score": 0-10,
                "whale_activity": bool,
                "trending": bool,
                "reasoning": str
            }
        """
        if not self.client:
            return self._fallback_response()
        
        # Extract base symbol (BTC from BTCUSDT)
        base_symbol = symbol.replace("USDT", "").replace("PERP", "")
        
        try:
            # Grok prompt for social sentiment
            prompt = f"""Analyze current X/Twitter sentiment for ${base_symbol} (Bitcoin/Ethereum).

Focus on:
1. **FOMO Detection:** Are people FOMOing? (tweets like "to the moon", "buying the dip")
2. **FUD Detection:** Is there fear/panic? (tweets like "crash incoming", "sell now")
3. **Whale Activity:** Any mentions of large wallet movements or whale buys/sells?
4. **Influencer Sentiment:** What are top crypto influencers saying?
5. **Breaking News:** Any major news in last 5 minutes?

Return ONLY this JSON format:
{{
    "sentiment": "BULLISH" or "BEARISH" or "NEUTRAL",
    "confidence": 1-10,
    "fomo_score": 0-10,
    "fud_score": 0-10,
    "whale_activity": true/false,
    "trending": true/false,
    "key_signals": "brief 1-line summary"
}}

Be objective. High confidence only if strong consensus."""

            # Call Grok API
            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {"role": "system", "content": "You are a crypto social sentiment analyzer. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temp for consistency
                max_tokens=200
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON
            import json
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Validate and add defaults
            sentiment = result.get("sentiment", "NEUTRAL").upper()
            confidence = min(10, max(1, int(result.get("confidence", 5))))
            fomo = min(10, max(0, int(result.get("fomo_score", 0))))
            fud = min(10, max(0, int(result.get("fud_score", 0))))
            whale = result.get("whale_activity", False)
            trending = result.get("trending", False)
            signals = result.get("key_signals", "No significant activity")
            
            # ===== MANIPULATION DETECTION (NEW) =====
            manipulation_flags = []
            
            # Check 1: Extreme one-sided sentiment (coordinated attack)
            if fomo >= 8 and fud <= 2:
                confidence = max(1, confidence - 2)
                manipulation_flags.append("Extreme FOMO")
            
            if fud >= 8 and fomo <= 2:
                confidence = max(1, confidence - 2)
                manipulation_flags.append("Extreme FUD")
            
            # Check 2: Both high FOMO and FUD (confused market = noise)
            if fomo >= 7 and fud >= 7:
                confidence = max(1, confidence - 3)
                manipulation_flags.append("Conflicting signals")
                sentiment = "NEUTRAL"  # Override to neutral
            
            # Append warnings to reasoning
            if manipulation_flags:
                signals += f" | ⚠️ {', '.join(manipulation_flags)}"
            
            logger.info(f"📱 Grok Sentiment ({base_symbol}): {sentiment} | FOMO: {fomo} | FUD: {fud} | Conf: {confidence}")
            if manipulation_flags:
                logger.warning(f"🚨 Manipulation flags: {manipulation_flags}")
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "fomo_score": fomo,
                "fud_score": fud,
                "whale_activity": whale,
                "trending": trending,
                "reasoning": signals,
                "manipulation_detected": len(manipulation_flags) > 0
            }
        
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> Dict:
        """Fallback when Grok unavailable"""
        return {
            "sentiment": "NEUTRAL",
            "confidence": 1,
            "fomo_score": 0,
            "fud_score": 0,
            "whale_activity": False,
            "trending": False,
            "reasoning": "Grok unavailable"
        }
    
    def get_vote_weight(self, analysis: Dict) -> int:
        """
        Calculate vote weight based on social sentiment strength.
        
        SAFETY: Max 3 votes to prevent Grok dominance
        
        Args:
            analysis: Result from analyze_social_sentiment()
        
        Returns:
            vote_weight: 1-3 votes (capped for safety)
        """
        confidence = analysis["confidence"]
        fomo = analysis["fomo_score"]
        fud = analysis["fud_score"]
        whale = analysis["whale_activity"]
        trending = analysis["trending"]
        manipulation = analysis.get("manipulation_detected", False)
        
        # Base weight (conservative)
        weight = 1  # Start low
        
        # Boost for high confidence (but not extreme)
        if confidence >= 7 and confidence <= 9:
            weight += 1
        
        # Boost for moderate signals (not extremes)
        if (fomo >= 6 and fomo <= 8) or (fud >= 6 and fud <= 8):
            weight += 1
        
        # Boost for whale activity (reliable signal)
        if whale:
            weight += 1
        
        # PENALTY: Manipulation detected
        if manipulation:
            weight = max(1, weight - 1)
            logger.warning("🚨 Grok vote weight reduced due to manipulation flags")
        
        # CAP at 3 votes (NEVER dominate decision)
        return min(3, weight)
