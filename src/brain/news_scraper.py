"""
DEMIR AI - Crypto News Scraper
API gerektirmeden kripto haberlerini web scraping ile çeker.

Sources:
- CoinTelegraph RSS
- CoinDesk RSS
- CryptoPanic (free API)
- Bitcoin Magazine RSS

Simple sentiment analysis on headlines.
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger("NEWS_SCRAPER")


@dataclass
class NewsItem:
    """Haber öğesi"""
    title: str
    source: str
    url: str
    timestamp: datetime
    sentiment: str  # BULLISH, BEARISH, NEUTRAL
    impact: str     # HIGH, MEDIUM, LOW
    coins_mentioned: List[str]


class CryptoNewsScraper:
    """
    Kripto Haber Analizi - API gerektirmeden!
    
    RSS feed'lerden haberleri çeker ve basit sentiment analizi yapar.
    """
    
    # Pozitif kelimeler (Türkçe + İngilizce)
    BULLISH_WORDS = [
        # English
        'surge', 'soar', 'rally', 'bullish', 'breakout', 'record', 'high', 'moon',
        'pump', 'adoption', 'approved', 'approval', 'etf', 'institutional',
        'partnership', 'integration', 'upgrade', 'launch', 'milestone',
        'all-time high', 'ath', 'breakthrough', 'mass adoption', 'wins',
        # Türkçe
        'yükseliş', 'rekor', 'onay', 'kabul', 'ortaklık', 'başarı', 'artış'
    ]
    
    # Negatif kelimeler
    BEARISH_WORDS = [
        # English
        'crash', 'dump', 'bearish', 'sell-off', 'selloff', 'plunge', 'drop',
        'hack', 'hacked', 'exploit', 'scam', 'fraud', 'ban', 'banned', 'regulation',
        'sec', 'lawsuit', 'investigation', 'crackdown', 'warning', 'collapse',
        'bankruptcy', 'insolvent', 'liquidation', 'fear', 'panic', 'fud',
        # Türkçe
        'düşüş', 'çöküş', 'hack', 'dolandırıcılık', 'yasak', 'uyarı', 'risk'
    ]
    
    # Yüksek etkili kelimeler
    HIGH_IMPACT_WORDS = [
        'etf', 'sec', 'federal reserve', 'fed', 'rate', 'billion', 'million',
        'regulation', 'law', 'government', 'china', 'usa', 'europe', 'breaking',
        'urgent', 'just in', 'confirmed', 'official'
    ]
    
    # RSS Feed kaynakları
    RSS_SOURCES = {
        'cointelegraph': 'https://cointelegraph.com/rss',
        'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'bitcoinmagazine': 'https://bitcoinmagazine.com/feed'
    }
    
    def __init__(self):
        self.last_check = datetime.now() - timedelta(hours=1)
        self.cached_news = []
        self.seen_titles = set()  # Avoid duplicates
        
    def fetch_all_news(self, max_age_hours: int = 2) -> List[NewsItem]:
        """Tüm kaynaklardan haberleri çek"""
        all_news = []
        
        # 1. RSS Feeds
        for source_name, rss_url in self.RSS_SOURCES.items():
            try:
                news = self._parse_rss(source_name, rss_url)
                all_news.extend(news)
            except Exception as e:
                logger.debug(f"RSS fetch failed for {source_name}: {e}")
        
        # 2. CryptoPanic (free, no API key needed for public posts)
        try:
            panic_news = self._fetch_cryptopanic()
            all_news.extend(panic_news)
        except Exception as e:
            logger.debug(f"CryptoPanic fetch failed: {e}")
        
        # Filter by age
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        recent_news = [n for n in all_news if n.timestamp > cutoff]
        
        # Remove duplicates
        unique_news = []
        for news in recent_news:
            if news.title not in self.seen_titles:
                self.seen_titles.add(news.title)
                unique_news.append(news)
        
        # Sort by timestamp (newest first)
        unique_news.sort(key=lambda x: x.timestamp, reverse=True)
        
        self.cached_news = unique_news
        self.last_check = datetime.now()
        
        return unique_news
    
    def _parse_rss(self, source_name: str, rss_url: str) -> List[NewsItem]:
        """RSS feed'i parse et"""
        news_items = []
        
        try:
            response = requests.get(rss_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code != 200:
                return []
            
            content = response.text
            
            # Simple XML parsing without external libraries
            # Extract items between <item> tags
            items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
            
            for item in items[:10]:  # Max 10 per source
                title = self._extract_tag(item, 'title')
                link = self._extract_tag(item, 'link')
                pub_date = self._extract_tag(item, 'pubDate')
                
                if not title:
                    continue
                
                # Parse date
                timestamp = self._parse_date(pub_date) if pub_date else datetime.now()
                
                # Analyze sentiment
                sentiment, impact = self._analyze_headline(title)
                
                # Extract mentioned coins
                coins = self._extract_coins(title)
                
                news_items.append(NewsItem(
                    title=title,
                    source=source_name,
                    url=link or '',
                    timestamp=timestamp,
                    sentiment=sentiment,
                    impact=impact,
                    coins_mentioned=coins
                ))
                
        except Exception as e:
            logger.warning(f"RSS parse error for {source_name}: {e}")
        
        return news_items
    
    def _fetch_cryptopanic(self) -> List[NewsItem]:
        """CryptoPanic'ten haberleri çek (free tier)"""
        news_items = []
        
        try:
            # Public endpoint (no API key)
            url = "https://cryptopanic.com/api/v1/posts/?auth_token=FREE&public=true&kind=news"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for post in data.get('results', [])[:10]:
                    title = post.get('title', '')
                    source = post.get('source', {}).get('title', 'CryptoPanic')
                    url = post.get('url', '')
                    
                    # Parse timestamp (strip timezone to avoid comparison issues)
                    pub_date = post.get('published_at', '')
                    try:
                        ts = datetime.fromisoformat(pub_date.replace('Z', '+00:00')) if pub_date else datetime.now()
                        timestamp = ts.replace(tzinfo=None) if ts.tzinfo else ts
                    except:
                        timestamp = datetime.now()
                    
                    # Analyze
                    sentiment, impact = self._analyze_headline(title)
                    coins = self._extract_coins(title)
                    
                    # Use CryptoPanic's own sentiment if available
                    votes = post.get('votes', {})
                    if votes.get('positive', 0) > votes.get('negative', 0) * 2:
                        sentiment = 'BULLISH'
                    elif votes.get('negative', 0) > votes.get('positive', 0) * 2:
                        sentiment = 'BEARISH'
                    
                    news_items.append(NewsItem(
                        title=title,
                        source=source,
                        url=url,
                        timestamp=timestamp,
                        sentiment=sentiment,
                        impact=impact,
                        coins_mentioned=coins
                    ))
                    
        except Exception as e:
            logger.debug(f"CryptoPanic error: {e}")
        
        return news_items
    
    def _extract_tag(self, xml: str, tag: str) -> str:
        """Simple XML tag extraction"""
        pattern = f'<{tag}[^>]*>(.*?)</{tag}>'
        match = re.search(pattern, xml, re.DOTALL)
        if match:
            content = match.group(1)
            # Remove CDATA
            content = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', content, flags=re.DOTALL)
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', '', content)
            return content.strip()
        return ''
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats (returns timezone-naive datetime)"""
        try:
            # Common RSS formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S GMT',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str.strip(), fmt)
                    # Strip timezone to avoid comparison issues
                    return dt.replace(tzinfo=None) if dt.tzinfo else dt
                except:
                    continue
                    
        except:
            pass
        
        return datetime.now()
    
    def _analyze_headline(self, title: str) -> Tuple[str, str]:
        """Başlıktan sentiment ve etki analizi"""
        title_lower = title.lower()
        
        # Count matches
        bullish_count = sum(1 for word in self.BULLISH_WORDS if word in title_lower)
        bearish_count = sum(1 for word in self.BEARISH_WORDS if word in title_lower)
        high_impact = any(word in title_lower for word in self.HIGH_IMPACT_WORDS)
        
        # Determine sentiment
        if bullish_count > bearish_count:
            sentiment = 'BULLISH'
        elif bearish_count > bullish_count:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        # Determine impact
        if high_impact or 'breaking' in title_lower:
            impact = 'HIGH'
        elif bullish_count + bearish_count >= 2:
            impact = 'MEDIUM'
        else:
            impact = 'LOW'
        
        return sentiment, impact
    
    def _extract_coins(self, title: str) -> List[str]:
        """Başlıktan bahsedilen coinleri çıkar"""
        coins = []
        title_upper = title.upper()
        
        coin_keywords = {
            'BTC': ['BITCOIN', 'BTC'],
            'ETH': ['ETHEREUM', 'ETH', 'ETHER'],
            'SOL': ['SOLANA', 'SOL'],
            'XRP': ['RIPPLE', 'XRP'],
            'DOGE': ['DOGECOIN', 'DOGE'],
            'LTC': ['LITECOIN', 'LTC']
        }
        
        for coin, keywords in coin_keywords.items():
            if any(kw in title_upper for kw in keywords):
                coins.append(coin)
        
        return coins
    
    def get_market_sentiment(self) -> Dict:
        """Genel piyasa sentiment özeti"""
        if not self.cached_news:
            self.fetch_all_news()
        
        bullish = sum(1 for n in self.cached_news if n.sentiment == 'BULLISH')
        bearish = sum(1 for n in self.cached_news if n.sentiment == 'BEARISH')
        neutral = sum(1 for n in self.cached_news if n.sentiment == 'NEUTRAL')
        
        total = bullish + bearish + neutral
        if total == 0:
            return {'sentiment': 'NEUTRAL', 'score': 50, 'news_count': 0}
        
        score = ((bullish - bearish) / total * 50) + 50  # 0-100 scale
        
        if score > 60:
            sentiment = 'BULLISH'
        elif score < 40:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'sentiment': sentiment,
            'score': score,
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'news_count': total,
            'high_impact_count': sum(1 for n in self.cached_news if n.impact == 'HIGH')
        }
    
    def get_important_news(self, max_items: int = 5) -> List[NewsItem]:
        """Önemli haberleri getir (HIGH impact veya BULLISH/BEARISH)"""
        if not self.cached_news:
            self.fetch_all_news()
        
        important = [n for n in self.cached_news 
                     if n.impact == 'HIGH' or n.sentiment != 'NEUTRAL']
        
        return important[:max_items]
    
    def format_for_telegram(self) -> str:
        """Telegram için haber özeti formatla"""
        sentiment = self.get_market_sentiment()
        important = self.get_important_news(3)
        
        if not important:
            return ""
        
        # Overall sentiment
        score = sentiment['score']
        if score > 65:
            mood = "🟢 BULLISH"
        elif score < 35:
            mood = "🔴 BEARISH"
        else:
            mood = "⚪ NEUTRAL"
        
        msg = "📰 *Haber Sentimenti*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        msg += f"Genel Mood: {mood} (Skor: {score:.0f}/100)\n"
        msg += f"📊 {sentiment['bullish_count']}↑ | {sentiment['bearish_count']}↓ | {sentiment['neutral_count']}→\n\n"
        
        if important:
            msg += "*Önemli Haberler:*\n"
            for news in important:
                emoji = "🟢" if news.sentiment == 'BULLISH' else "🔴" if news.sentiment == 'BEARISH' else "⚪"
                impact = "⚡" if news.impact == 'HIGH' else ""
                msg += f"• {emoji}{impact} {news.title[:60]}...\n"
                msg += f"  _{news.source}_\n"
        
        msg += f"\n⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        return msg
