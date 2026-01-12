import logging
import aiohttp
import json
import asyncio
from typing import Dict, List
from src.config import Config
from src.utils.retry import async_retry

logger = logging.getLogger("NEWS_SENTIMENT")

class NewsSentimentAnalyzer:
    """
    Multi-AI News Sentiment Analyzer
    - Primary: GROQ (Llama 3.1 70B) - free, unlimited
    - Fallback: Gemini Flash - fast and free
    - Legacy: OpenAI GPT-4 (optional)
    """
    def __init__(self):
        # Initialize GROQ client (primary)
        self.groq_client = None
        if Config.GROQ_API_KEY:
            try:
                from openai import AsyncOpenAI
                self.groq_client = AsyncOpenAI(
                    api_key=Config.GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1"
                )
                logger.info("✅ GROQ (Llama 3.1) initialized for news sentiment")
            except Exception as e:
                logger.warning(f"⚠️ GROQ init failed: {e}")
        
        # Initialize Gemini (fallback)
        self.gemini_client = None
        if Config.GOOGLE_API_KEY:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=Config.GOOGLE_API_KEY)
                logger.info("✅ Gemini Flash initialized as fallback")
            except Exception as e:
                logger.warning(f"⚠️ Gemini init failed: {e}")
        
        # Legacy OpenAI (optional)
        self.openai_client = None
        if Config.OPENAI_API_KEY:
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("✅ OpenAI GPT-4 available (legacy)")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI init failed: {e}")
        
        if not any([self.groq_client, self.gemini_client, self.openai_client]):
            logger.warning("⚠️ No AI services available for news sentiment")
        
        self.news_sources = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/"
        ]
    
    @async_retry(max_attempts=3, base_delay=2)
    async def get_latest_headlines(self) -> List[str]:
        """Fetch recent crypto news headlines"""
        headlines = []
        
        try:
            async with aiohttp.ClientSession() as session:
                if Config.NEWSAPI_KEY:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": "bitcoin OR ethereum OR crypto",
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 10,
                        "apiKey": Config.NEWSAPI_KEY
                    }
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            headlines = [article['title'] for article in data.get('articles', [])[:10]]
                else:
                    # Fallback: Generic headlines
                    headlines = [
                        "Bitcoin reaches new highs amid institutional adoption",
                        "Ethereum upgrade scheduled for next month",
                        "Crypto market shows mixed signals"
                    ]
        except Exception as e:
            logger.warning(f"News fetch error: {e}")
            headlines = ["Market data unavailable"]
            
        return headlines
    
    async def _analyze_with_groq(self, prompt: str) -> Dict[str, any]:
        """Analyze using GROQ (Llama 3.3)"""
        try:
            response = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Updated: 3.1 was decommissioned
                messages=[
                    {"role": "system", "content": "You are a crypto market analyst. Respond in JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON
            if "{" in result_text and "}" in result_text:
                start = result_text.find("{")
                end = result_text.rfind("}") + 1
                return json.loads(result_text[start:end])
            else:
                return {"sentiment": "NEUTRAL", "summary": result_text}
                
        except Exception as e:
            logger.error(f"GROQ error: {e}")
            raise
    
    async def _analyze_with_gemini(self, prompt: str) -> Dict[str, any]:
        """Analyze using Gemini Flash"""
        try:
            response = await asyncio.to_thread(
                self.gemini_client.models.generate_content,
                model='models/gemini-1.5-flash-latest',  # Fix: Add 'models/' prefix
                contents=prompt
            )
            result_text = response.text
            
            # Parse JSON
            if "{" in result_text and "}" in result_text:
                start = result_text.find("{")
                end = result_text.rfind("}") + 1
                return json.loads(result_text[start:end])
            else:
                return {"sentiment": "NEUTRAL", "summary": result_text}
                
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            raise
    
    async def _analyze_with_openai(self, prompt: str) -> Dict[str, any]:
        """Legacy: Analyze using OpenAI GPT-4"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a crypto market analyst. Respond in JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON
            if "{" in result_text and "}" in result_text:
                start = result_text.find("{")
                end = result_text.rfind("}") + 1
                return json.loads(result_text[start:end])
            else:
                return {"sentiment": "NEUTRAL", "summary": result_text}
                
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            raise
    
    @async_retry(max_attempts=3, base_delay=2)
    async def analyze_sentiment(self) -> Dict[str, any]:
        """
        Analyze news sentiment with multi-AI fallback:
        1. Try GROQ (primary, free)
        2. Try Gemini (fallback, free)
        3. Try OpenAI (legacy, if available)
        """
        headlines = await self.get_latest_headlines()
        
        if not headlines:
            return {"sentiment": "NEUTRAL", "summary": "No news data"}
        
        # Prepare prompt
        news_text = "\n".join([f"- {h}" for h in headlines])
        prompt = f"""Analyze the following crypto news headlines and determine market sentiment:

{news_text}

Provide:
1. Overall Sentiment: BULLISH, BEARISH, or NEUTRAL
2. Confidence: How strong is the sentiment (1-10)?
3. Key Themes: What are the main narratives?
4. Trading Impact: How might this affect BTC/ETH prices?

Respond in JSON format with keys: sentiment, confidence, themes, impact"""
        
        # Try GROQ first (primary)
        if self.groq_client:
            try:
                result = await self._analyze_with_groq(prompt)
                logger.info(f"📰 News Sentiment (GROQ): {result.get('sentiment', 'N/A')} (Confidence: {result.get('confidence', 'N/A')})")
                return result
            except Exception as e:
                logger.warning(f"GROQ failed, trying Gemini: {e}")
        
        # Fallback to Gemini
        if self.gemini_client:
            try:
                result = await self._analyze_with_gemini(prompt)
                logger.info(f"📰 News Sentiment (Gemini): {result.get('sentiment', 'N/A')} (Confidence: {result.get('confidence', 'N/A')})")
                return result
            except Exception as e:
                logger.warning(f"Gemini failed, trying OpenAI: {e}")
        
        # Last resort: OpenAI (if available)
        if self.openai_client:
            try:
                result = await self._analyze_with_openai(prompt)
                logger.info(f"📰 News Sentiment (OpenAI): {result.get('sentiment', 'N/A')} (Confidence: {result.get('confidence', 'N/A')})")
                return result
            except Exception as e:
                logger.error(f"All AI services failed for news sentiment: {e}")
        
        # All failed - return neutral
        return {"sentiment": "NEUTRAL", "summary": "All AI services unavailable"}
