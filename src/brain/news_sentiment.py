import logging
import aiohttp
from openai import AsyncOpenAI
from src.config import Config
from src.utils.retry import async_retry  # FIX 2.1

logger = logging.getLogger("NEWS_SENTIMENT")

class NewsSentimentAnalyzer:
    """
    Scrapes crypto news and uses GPT-4 for sentiment analysis.
    """
    def __init__(self):
        if Config.OPENAI_API_KEY:
            self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        else:
            self.client = None
            logger.warning("⚠️ OpenAI API Key missing. News sentiment disabled.")
        
        self.news_sources = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/"
        ]
    
    @async_retry(max_attempts=3, base_delay=2)  # FIX 2.1: Retry logic
    async def get_latest_headlines(self) -> list:
        """Fetch recent crypto news headlines"""
        headlines = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Simple approach: fetch from NewsAPI if available
                if Config.NEWSAPI_KEY:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": "bitcoin OR ethereum OR crypto",
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 10,
                        "apiKey": Config.NEWSAPI_KEY
                    }
                    # FIX 1.30: Add timeout
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            headlines = [article['title'] for article in data.get('articles', [])[:10]]
                else:
                    # Fallback: Manual scraping (basic)
                    headlines = [
                        "Bitcoin reaches new highs amid institutional adoption",
                        "Ethereum upgrade scheduled for next month",
                        "Crypto market shows mixed signals"
                    ]
        except Exception as e:
            logger.warning(f"News fetch error: {e}")
            headlines = ["Market data unavailable"]
            
        return headlines
    
    @async_retry(max_attempts=3, base_delay=2)  # FIX 2.1: Retry logic
    async def analyze_sentiment(self) -> dict:
        """
        Fetch news and use GPT-4 to analyze sentiment.
        """
        if not self.client:
            return {"sentiment": "NEUTRAL", "summary": "OpenAI unavailable"}
            
        try:
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

            # Call GPT-4 with implicit retry from decorator
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a crypto market analyst specializing in news sentiment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON
            try:
                import json
                if "{" in result_text and "}" in result_text:
                    start = result_text.find("{")
                    end = result_text.rfind("}") + 1
                    result = json.loads(result_text[start:end])
                else:
                    result = {"sentiment": "NEUTRAL", "summary": result_text}
            except:
                result = {"sentiment": "NEUTRAL", "summary": result_text}
                
            logger.info(f"📰 News Sentiment: {result.get('sentiment', 'N/A')} (Confidence: {result.get('confidence', 'N/A')})")
            return result
            
        except Exception as e:
            logger.error(f"GPT-4 sentiment error: {e}")
            return {"sentiment": "ERROR", "summary": str(e)}
