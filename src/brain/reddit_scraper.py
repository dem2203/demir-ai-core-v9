"""
DEMIR AI - Reddit Sentiment Scraper
Reddit crypto topluluklarından sentiment analizi yapar.

PHASE 42: Critical Scraper - No API Key Required
- old.reddit.com JSON endpoints (auth gerektirmez)
- r/cryptocurrency, r/bitcoin, r/ethtrader
- Sentiment scoring (bullish/bearish keywords)
- Post velocity & upvote ratio tracking
"""
import logging
import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("REDDIT_SCRAPER")


@dataclass
class RedditPost:
    """Reddit post"""
    subreddit: str
    title: str
    score: int  # Upvotes
    num_comments: int
    upvote_ratio: float
    created_utc: datetime
    url: str
    sentiment: str  # BULLISH, BEARISH, NEUTRAL


class RedditScraper:
    """
    Reddit Sentiment Scraper
    
    API key gerektirmez - old.reddit.com JSON endpoints kullanır.
    
    Subreddits:
    - r/cryptocurrency (8M+ members) - Genel crypto
    - r/bitcoin (6M+ members) - BTC dominance
    - r/ethtrader (1M+ members) - ETH sentiment
    """
    
    # Sentiment keywords (Turkish + English)
    BULLISH_KEYWORDS = [
        # English
        'moon', 'bullish', 'buy', 'buying', 'pump', 'rally', 'breakout', 'explosion',
        'hodl', 'accumulate', 'bull run', 'parabolic', 'rocket', 'to the moon',
        'undervalued', 'bottom', 'dip', 'opportunity', 'adoption', 'institutional',
        # Turkish
        'yükseliş', 'alım', 'fırsat', 'patlama', 'rally', 'boğa'
    ]
    
    BEARISH_KEYWORDS = [
        # English
        'bearish', 'sell', 'selling', 'dump', 'crash', 'correction', 'bubble',
        'overvalued', 'scam', 'rugpull', 'bear market', 'collapse', 'dead',
        'ponzi', 'manipulation', 'fud', 'fear', 'panic',
        # Turkish
        'düşüş', 'satış', 'çöküş', 'balon', 'dolandırıcılık', 'ayı'
    ]
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.last_fetch = {}
        self.subreddits = ['cryptocurrency', 'bitcoin', 'ethtrader']
    
    def get_sentiment(self, hours: int = 24) -> Dict:
        """
        Reddit genel sentiment skoru.
        """
        cache_key = f'reddit_sentiment_{hours}h'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            all_posts = []
            
            # Scrape each subreddit
            for sub in self.subreddits:
                posts = self._scrape_subreddit(sub, hours)
                all_posts.extend(posts)
            
            if not all_posts:
                return {
                    'score': 50,
                    'sentiment': 'NEUTRAL',
                    'post_count': 0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'summary': 'Reddit verisi alınamadı'
                }
            
            # Calculate sentiment
            bullish = [p for p in all_posts if p.sentiment == 'BULLISH']
            bearish = [p for p in all_posts if p.sentiment == 'BEARISH']
            neutral = [p for p in all_posts if p.sentiment == 'NEUTRAL']
            
            # Weighted by upvotes (high upvote posts matter more)
            bullish_weight = sum(p.score * p.upvote_ratio for p in bullish)
            bearish_weight = sum(p.score * p.upvote_ratio for p in bearish)
            total_weight = bullish_weight + bearish_weight + 1  # +1 to avoid division by zero
            
            # Sentiment score (0-100)
            score = int((bullish_weight / total_weight) * 100) if total_weight > 0 else 50
            
            # Classification
            if score >= 70:
                sentiment = 'BULLISH'
                emoji = '🟢'
            elif score <= 30:
                sentiment = 'BEARISH'
                emoji = '🔴'
            else:
                sentiment = 'NEUTRAL'
                emoji = '⚪'
            
            summary = f"{emoji} Reddit Sentiment: {score}/100 | {len(bullish)} bullish, {len(bearish)} bearish posts"
            
            result = {
                'score': score,
                'sentiment': sentiment,
                'post_count': len(all_posts),
                'bullish_count': len(bullish),
                'bearish_count': len(bearish),
                'neutral_count': len(neutral),
                'top_posts': all_posts[:10],  # Top 10 by score
                'summary': summary,
                'timestamp': datetime.now()
            }
            
            self._set_cache(cache_key, result)
            logger.info(f"Reddit sentiment: {score}/100 ({len(all_posts)} posts analyzed)")
            
            return result
            
        except Exception as e:
            logger.warning(f"Reddit sentiment scraping failed: {e}")
            return {
                'score': 50,
                'sentiment': 'NEUTRAL',
                'post_count': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'summary': f'Hata: {e}'
            }
    
    def format_for_telegram(self, hours: int = 24) -> str:
        """Telegram formatı"""
        sentiment = self.get_sentiment(hours)
        
        msg = "💬 *Reddit Sentiment*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        score = sentiment['score']
        mood = sentiment['sentiment']
        
        # Score bar
        bar_length = int(score / 10)
        score_bar = "█" * bar_length + "░" * (10 - bar_length)
        
        msg += f"📊 Score: [{score_bar}] **{score}/100**\n"
        msg += f"😊 Mood: **{mood}**\n\n"
        
        msg += f"🟢 Bullish Posts: {sentiment['bullish_count']}\n"
        msg += f"🔴 Bearish Posts: {sentiment['bearish_count']}\n"
        msg += f"⚪ Neutral Posts: {sentiment['neutral_count']}\n\n"
        
        # Top posts
        if sentiment.get('top_posts'):
            msg += "**Top Posts:**\n"
            for post in sentiment['top_posts'][:3]:
                emoji = "🟢" if post.sentiment == 'BULLISH' else "🔴" if post.sentiment == 'BEARISH' else "⚪"
                msg += f"• {emoji} [{post.subreddit}] {post.title[:50]}...\n"
                msg += f"  ↑{post.score} ({post.upvote_ratio:.0%} upvoted)\n"
        
        msg += f"\n💡 _{sentiment['summary']}_\n"
        msg += f"⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        return msg
    
    # =========================================
    # PRIVATE HELPERS
    # =========================================
    
    def _scrape_subreddit(self, subreddit: str, hours: int) -> List[RedditPost]:
        """
        Tek subreddit'i scrape et.
        """
        posts = []
        
        try:
            # old.reddit.com JSON endpoint (no auth required!)
            url = f"https://old.reddit.com/r/{subreddit}/hot.json?limit=100"
            
            resp = requests.get(url, timeout=15, headers=self.HEADERS)
            
            if resp.status_code != 200:
                logger.warning(f"Reddit scrape failed for r/{subreddit}: {resp.status_code}")
                return posts
            
            data = resp.json()
            
            cutoff = datetime.now() - timedelta(hours=hours)
            
            for item in data.get('data', {}).get('children', []):
                try:
                    post_data = item.get('data', {})
                    
                    # Parse post
                    created_utc = datetime.fromtimestamp(post_data.get('created_utc', 0))
                    
                    if created_utc < cutoff:
                        continue
                    
                    title = post_data.get('title', '')
                    score = post_data.get('score', 0)
                    num_comments = post_data.get('num_comments', 0)
                    upvote_ratio = post_data.get('upvote_ratio', 0.5)
                    permalink = post_data.get('permalink', '')
                    
                    # Sentiment analysis
                    sentiment = self._analyze_sentiment(title)
                    
                    posts.append(RedditPost(
                        subreddit=subreddit,
                        title=title,
                        score=score,
                        num_comments=num_comments,
                        upvote_ratio=upvote_ratio,
                        created_utc=created_utc,
                        url=f"https://reddit.com{permalink}",
                        sentiment=sentiment
                    ))
                    
                except Exception as e:
                    logger.debug(f"Post parse error: {e}")
                    continue
            
            logger.info(f"Scraped {len(posts)} posts from r/{subreddit}")
            
        except Exception as e:
            logger.warning(f"Subreddit scrape failed for r/{subreddit}: {e}")
        
        # Sort by score (most upvoted first)
        posts.sort(key=lambda p: p.score, reverse=True)
        
        return posts
    
    def _analyze_sentiment(self, text: str) -> str:
        """
        Basit keyword-based sentiment analizi.
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        if bullish_count > bearish_count:
            return 'BULLISH'
        elif bearish_count > bullish_count:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
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
