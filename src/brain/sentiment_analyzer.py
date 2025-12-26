import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import time

logger = logging.getLogger("SENTIMENT_ANALYZER")

class SentimentAnalyzer:
    """
    SENTIMENT ANALYSIS - Market Psychology Monitor (FAIL FAST MODE)
    
    ⚠️ FAIL FAST: Veri alınamazsa None döner, NEUTRAL FALLBACK YOK!
       Sinyal üretimi sentiment verisine bağlıysa DURDURULMALIDIR.
    
    Data Sources:
    1. Fear & Greed Index (Alternative.me) - Free API
    2. Reddit Sentiment (r/cryptocurrency, r/Bitcoin) - PRAW
    
    Outputs composite sentiment score (-1 to 1):
    - Negative: Fear, panic, bearish posts
    - Positive: Greed, euphoria, bullish posts
    - None: Data unavailable (FAIL FAST)
    """
    
    def __init__(self):
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.cache = {}
        self.cache_duration = 1800  # 30 minutes
        
        # Reddit setup (optional - only if API keys available)
        self.reddit_available = False
        try:
            import praw
            import os
            
            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            
            if client_id and client_secret:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="DevirAI/1.0"
                )
                self.reddit_available = True
                logger.info("📰 Reddit Sentiment: ACTIVE")
            else:
                logger.warning("⚠️ Reddit credentials not found. Using Fear & Greed only.")
        except Exception as e:
            logger.warning(f"⚠️ Reddit initialization failed: {e}")
            
    def get_sentiment(self, symbol: str = "BTC") -> Dict:
        """
        Get combined sentiment analysis.
        Returns composite score and breakdown.
        """
        # Check cache
        cache_key = f"sentiment_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                logger.info("📊 Using cached sentiment data")
                return cached_data
        
        # Fetch new sentiment
        fear_greed = self._get_fear_greed()
        
        # FAIL FAST: Fear & Greed alınamazsa None döndür
        if fear_greed is None:
            logger.warning("❌ FAIL FAST: Sentiment veri yok, sinyal üretilmemeli")
            return None
            
        reddit_score = self._get_reddit_sentiment(symbol) if self.reddit_available else 0
        
        # Composite calculation
        # Fear & Greed: 0-100 → normalize to -1 to 1
        fg_normalized = (fear_greed - 50) / 50.0  # 0 = -1 (fear), 100 = 1 (greed)
        
        # Weight: 70% Fear & Greed, 30% Reddit (if available)
        if self.reddit_available:
            composite = fg_normalized * 0.7 + reddit_score * 0.3
        else:
            composite = fg_normalized
        
        # Classify sentiment
        if composite > 0.3:
            sentiment_label = "BULLISH"
        elif composite < -0.3:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"
        
        result = {
            "composite_score": round(composite, 2),  # -1 to 1
            "sentiment": sentiment_label,
            "fear_greed_index": fear_greed,  # 0-100
            "reddit_score": round(reddit_score, 2) if self.reddit_available else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        self.cache[cache_key] = (time.time(), result)
        
        logger.info(f"📊 Sentiment: {sentiment_label} | Score: {composite:.2f} | F&G: {fear_greed}")
        return result
        
    def _get_fear_greed(self) -> Optional[int]:
        """
        Fetch Fear & Greed Index from Alternative.me
        
        FAIL FAST: Veri alınamazsa None döner.
        
        Returns: 0 (Extreme Fear) to 100 (Extreme Greed), or None if unavailable
        """
        try:
            response = requests.get(self.fear_greed_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # API returns current value
            fg_value = int(data['data'][0]['value'])
            logger.debug(f"Fear & Greed Index: {fg_value}")
            return fg_value
            
        except Exception as e:
            logger.error(f"❌ FAIL FAST: Fear & Greed alınamadı: {e}")
            return None  # FAIL FAST: Veri yok
            
    def _get_reddit_sentiment(self, symbol: str) -> float:
        """
        Analyze Reddit sentiment for a symbol.
        Returns: -1 (very bearish) to 1 (very bullish)
        """
        if not self.reddit_available:
            return 0.0
            
        try:
            # Map symbols to subreddits
            subreddit_map = {
                "BTC": ["cryptocurrency", "Bitcoin"],
                "ETH": ["cryptocurrency", "ethereum"],
                "LTC": ["cryptocurrency", "litecoin"]
            }
            
            subreddits = subreddit_map.get(symbol, ["cryptocurrency"])
            
            positive_words = ["moon", "bull", "bullish", "buy", "pump", "rocket", "gain", "up", "green"]
            negative_words = ["bear", "bearish", "dump", "crash", "sell", "down", "red", "fear", "drop"]
            
            total_score = 0
            post_count = 0
            
            for sub_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(sub_name)
                    
                    # Analyze top 10 hot posts
                    for post in subreddit.hot(limit=10):
                        title_lower = post.title.lower()
                        
                        # Count sentiment words
                        pos_count = sum(1 for word in positive_words if word in title_lower)
                        neg_count = sum(1 for word in negative_words if word in title_lower)
                        
                        # Score this post
                        if pos_count > neg_count:
                            total_score += 1
                        elif neg_count > pos_count:
                            total_score -= 1
                        
                        post_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch r/{sub_name}: {e}")
                    continue
            
            if post_count == 0:
                return 0.0
            
            # Normalize to -1 to 1
            reddit_score = total_score / post_count
            return max(-1.0, min(1.0, reddit_score))
            
        except Exception as e:
            logger.error(f"Reddit sentiment analysis failed: {e}")
            return 0.0
            
    def get_status(self) -> Dict:
        """Get analyzer status"""
        return {
            "fear_greed_active": True,
            "reddit_active": self.reddit_available,
            "cache_size": len(self.cache)
        }
