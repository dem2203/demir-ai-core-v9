# -*- coding: utf-8 -*-
"""
DEMIR AI - Twitter Sentiment Scraper
Twitter/X sentiment analysis via Nitter (no API key!)

PHASE 45: High-Value Scraper
- Nitter instance scraping (Twitter frontend without API)
- Top crypto influencer tracking
- Hashtag trending analysis
- Sentiment scoring
"""
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

logger = logging.getLogger("TWITTER_SENTIMENT")


@dataclass
class Tweet:
    """Tweet data"""
    username: str
    text: str
    timestamp: datetime
    likes: int
    retweets: int
    replies: int
    sentiment: str  # BULLISH, BEARISH, NEUTRAL


class TwitterSentimentScraper:
    """
    Twitter Sentiment Scraper via Nitter
    
    Nitter = Privacy-focused Twitter frontend (no API needed)
    Tracks crypto influencers and hashtag trends
    """
    
    # Public Nitter instances (rotate if one fails)
    NITTER_INSTANCES = [
        'https://nitter.net',
        'https://nitter.1d4.us',
        'https://nitter.kavin.rocks',
    ]
    
    # Top crypto influencers to track
    CRYPTO_INFLUENCERS = [
        'APompliano',      # Anthony Pompliano
        'CryptoCobain',    # Crypto Cobain
        'VitalikButerin',  # Vitalik
        'CZ_Binance',      # CZ
        'elonmusk',        # Elon (crypto tweets)
        'DocumentingBTC',  # Bitcoin documenter
        'tier10k',         # Degen trader
    ]
    
    # Sentiment keywords
    BULLISH_KEYWORDS = [
        'bullish', 'moon', 'pump', 'buy', 'long', 'calls', 'breakout',
        'rally', 'surge', 'ATH', 'buying', 'accumulate', 'hodl',
        '🚀', '💎', '🐂', '🔥', '📈', 'LFG'
    ]
    
    BEARISH_KEYWORDS = [
        'bearish', 'dump', 'sell', 'short', 'puts', 'crash', 'drop',
        'fall', 'dip', 'correction', 'selling', 'exit', 'rekt',
        '📉', '🐻', '💀', '⚠️', 'scam', 'rugpull'
    ]
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html',
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 600  # 10 minutes
        self.last_fetch = {}
        self.current_instance = self.NITTER_INSTANCES[0]
    
    def get_influencer_sentiment(self, hours: int = 24) -> Dict:
        """
        Aggregate sentiment from top crypto influencers.
        
        Returns:
            {
                'score': 0-100 (50=neutral, >50=bullish, <50=bearish),
                'sentiment': 'BULLISH'/'BEARISH'/'NEUTRAL',
                'tweet_count': int,
                'bullish_count': int,
                'bearish_count': int,
                'top_tweets': List[Tweet],
                'summary': str
            }
        """
        cache_key = f'influencer_sentiment_{hours}h'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        all_tweets = []
        
        # Fetch tweets from each influencer
        for username in self.CRYPTO_INFLUENCERS[:5]:  # Top 5 to avoid rate limits
            try:
                tweets = self._fetch_user_tweets(username, limit=10)
                all_tweets.extend(tweets)
            except Exception as e:
                logger.debug(f"Failed to fetch {username}: {e}")
                continue
        
        if not all_tweets:
            return self._empty_result()
        
        # Filter by time
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_tweets = [t for t in all_tweets if t.timestamp > cutoff]
        
        if not recent_tweets:
            return self._empty_result()
        
        # Analyze sentiment
        bullish_count = sum(1 for t in recent_tweets if t.sentiment == 'BULLISH')
        bearish_count = sum(1 for t in recent_tweets if t.sentiment == 'BEARISH')
        neutral_count = len(recent_tweets) - bullish_count - bearish_count
        
        # Calculate score (0-100)
        if len(recent_tweets) > 0:
            score = 50 + ((bullish_count - bearish_count) / len(recent_tweets)) * 50
            score = max(0, min(100, score))
        else:
            score = 50
        
        # Determine overall sentiment
        if score >= 60:
            sentiment = 'BULLISH'
        elif score <= 40:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        # Sort by engagement
        top_tweets = sorted(recent_tweets, key=lambda t: t.likes + t.retweets, reverse=True)[:10]
        
        result = {
            'score': score,
            'sentiment': sentiment,
            'tweet_count': len(recent_tweets),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'top_tweets': top_tweets,
            'summary': self._generate_summary(score, sentiment, recent_tweets),
            'timestamp': datetime.now()
        }
        
        self._set_cache(cache_key, result)
        logger.info(f"Twitter sentiment: {sentiment} ({score}/100) from {len(recent_tweets)} tweets")
        
        return result
    
    def get_hashtag_trending(self, hashtag: str = '#Bitcoin') -> Dict:
        """
        Get trending analysis for a specific hashtag.
        """
        try:
            # Nitter search endpoint
            url = f"{self.current_instance}/search?f=tweets&q={hashtag}"
            
            response = requests.get(url, headers=self.HEADERS, timeout=15)
            
            if response.status_code != 200:
                return {'tweet_count': 0, 'sentiment': 'NEUTRAL'}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            tweets = self._parse_search_results(soup)
            
            if not tweets:
                return {'tweet_count': 0, 'sentiment': 'NEUTRAL'}
            
            # Same sentiment analysis as influencers
            bullish = sum(1 for t in tweets if t.sentiment == 'BULLISH')
            bearish = sum(1 for t in tweets if t.sentiment == 'BEARISH')
            
            score = 50 + ((bullish - bearish) / len(tweets)) * 50
            sentiment = 'BULLISH' if score >= 60 else 'BEARISH' if score <= 40 else 'NEUTRAL'
            
            return {
                'hashtag': hashtag,
                'tweet_count': len(tweets),
                'score': score,
                'sentiment': sentiment,
                'bullish_count': bullish,
                'bearish_count': bearish
            }
            
        except Exception as e:
            logger.warning(f"Hashtag trending failed: {e}")
            return {'tweet_count': 0, 'sentiment': 'NEUTRAL'}
    
    def format_for_telegram(self, hours: int = 24) -> str:
        """Telegram formatı"""
        data = self.get_influencer_sentiment(hours)
        
        msg = "🐦 *Twitter Sentiment (Influencers)*\\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\\n\\n"
        
        score = data['score']
        sentiment = data['sentiment']
        
        # Score visualization
        bar_length = int(score / 10)
        score_bar = "█" * bar_length + "░" * (10 - bar_length)
        
        if sentiment == 'BULLISH':
            emoji = "🟢"
        elif sentiment == 'BEARISH':
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        msg += f"**Sentiment Score:** [{score_bar}]\\n"
        msg += f"{emoji} **{sentiment}** ({score:.0f}/100)\\n\\n"
        
        msg += f"📊 **Tweets Analyzed:** {data['tweet_count']}\\n"
        msg += f"🟢 Bullish: {data['bullish_count']}\\n"
        msg += f"🔴 Bearish: {data['bearish_count']}\\n"
        msg += f"⚪ Neutral: {data['neutral_count']}\\n\\n"
        
        # Top tweet
        if data['top_tweets']:
            top = data['top_tweets'][0]
            msg += f"**Top Tweet** (@{top.username}):\\n"
            msg += f"_{top.text[:100]}..._\\n"
            msg += f"❤️ {top.likes} | 🔁 {top.retweets}\\n\\n"
        
        msg += f"💡 _{data['summary']}_\\n"
        msg += f"\\n⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        
        return msg
    
    # =========================================
    # PRIVATE METHODS
    # =========================================
    
    def _fetch_user_tweets(self, username: str, limit: int = 10) -> List[Tweet]:
        """Fetch recent tweets from a user via Nitter"""
        try:
            url = f"{self.current_instance}/{username}"
            
            response = requests.get(url, headers=self.HEADERS, timeout=15)
            
            if response.status_code != 200:
                # Try next Nitter instance
                self._rotate_instance()
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            tweets = self._parse_tweets(soup, username, limit)
            
            return tweets
            
        except Exception as e:
            logger.debug(f"Failed to fetch tweets for {username}: {e}")
            return []
    
    def _parse_tweets(self, soup: BeautifulSoup, username: str, limit: int) -> List[Tweet]:
        """Parse tweets from Nitter HTML"""
        tweets = []
        
        # Find tweet containers (Nitter HTML structure)
        tweet_containers = soup.find_all('div', class_='timeline-item')[:limit]
        
        for container in tweet_containers:
            try:
                # Extract text
                tweet_content = container.find('div', class_='tweet-content')
                if not tweet_content:
                    continue
                
                text = tweet_content.get_text(strip=True)
                
                # Skip retweets, only original content
                if text.startswith('RT @'):
                    continue
                
                # Extract engagement metrics
                stats = container.find('div', class_='tweet-stats')
                likes = 0
                retweets = 0
                replies = 0
                
                if stats:
                    likes_elem = stats.find('span', class_='icon-heart')
                    if likes_elem and likes_elem.parent:
                        likes = self._parse_number(likes_elem.parent.get_text())
                    
                    rt_elem = stats.find('span', class_='icon-retweet')
                    if rt_elem and rt_elem.parent:
                        retweets = self._parse_number(rt_elem.parent.get_text())
                
                # Determine sentiment
                sentiment = self._analyze_sentiment(text)
                
                tweet = Tweet(
                    username=username,
                    text=text,
                    timestamp=datetime.now(),  # Nitter doesn't always show exact time
                    likes=likes,
                    retweets=retweets,
                    replies=replies,
                    sentiment=sentiment
                )
                
                tweets.append(tweet)
                
            except Exception as e:
                logger.debug(f"Failed to parse tweet: {e}")
                continue
        
        return tweets
    
    def _parse_search_results(self, soup: BeautifulSoup) -> List[Tweet]:
        """Parse search results"""
        # Similar to _parse_tweets but for search results
        return self._parse_tweets(soup, 'search', 20)
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze tweet sentiment using keyword matching"""
        text_lower = text.lower()
        
        bullish_score = sum(1 for kw in self.BULLISH_KEYWORDS if kw.lower() in text_lower)
        bearish_score = sum(1 for kw in self.BEARISH_KEYWORDS if kw.lower() in text_lower)
        
        if bullish_score > bearish_score:
            return 'BULLISH'
        elif bearish_score > bullish_score:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _parse_number(self, text: str) -> int:
        """Parse number from text (handles K, M suffixes)"""
        try:
            text = text.strip()
            if 'K' in text:
                return int(float(text.replace('K', '')) * 1000)
            elif 'M' in text:
                return int(float(text.replace('M', '')) * 1000000)
            else:
                return int(re.sub(r'[^\d]', '', text))
        except:
            return 0
    
    def _generate_summary(self, score: float, sentiment: str, tweets: List) -> str:
        """Generate text summary"""
        if score >= 70:
            return f"🚀 Strong bullish sentiment! {len(tweets)} influencer tweets lean heavily bullish."
        elif score >= 60:
            return f"🟢 Moderately bullish. Influencers showing optimism."
        elif score >= 40:
            return f"↔️ Neutral sentiment. Mixed signals from influencers."
        elif score >= 30:
            return f"🔴 Moderately bearish. Influencers showing caution."
        else:
            return f"⚠️ Strong bearish sentiment detected from {len(tweets)} influencer tweets."
    
    def _rotate_instance(self):
        """Rotate to next Nitter instance"""
        current_index = self.NITTER_INSTANCES.index(self.current_instance)
        next_index = (current_index + 1) % len(self.NITTER_INSTANCES)
        self.current_instance = self.NITTER_INSTANCES[next_index]
        logger.info(f"Rotated to Nitter instance: {self.current_instance}")
    
    def _empty_result(self) -> Dict:
        """Return empty result"""
        return {
            'score': 50,
            'sentiment': 'NEUTRAL',
            'tweet_count': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'top_tweets': [],
            'summary': 'No Twitter data available',
            'timestamp': datetime.now()
        }
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and fresh"""
        if key not in self.cache or key not in self.last_fetch:
            return False
        
        age = (datetime.now() - self.last_fetch[key]).total_seconds()
        return age < self.cache_duration
    
    def _set_cache(self, key: str, data):
        """Cache data"""
        self.cache[key] = data
        self.last_fetch[key] = datetime.now()
