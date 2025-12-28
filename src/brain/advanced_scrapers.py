"""
DEMIR AI - Advanced Market Scrapers
API gerektirmeden gelişmiş piyasa verilerini web scraping ile çeker.

PHASE 36-37: Ultra Intelligence
1. Fear & Greed Index (alternative.me)
2. Token Unlock Calendar (defillama)
3. Liquidation Heatmap (coinglass)
4. Economic Calendar (investing.com)
5. Exchange Listings (binance announcements)
6. TradingView Signals (technicals summary)
7. CME Gap Tracker (tradingview/CME data)
8. DeFi TVL Monitor (defillama)
9. Stablecoin Flow (USDT/USDC minting)
"""
import logging
import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("ADVANCED_SCRAPERS")


@dataclass
class MarketEvent:
    """Piyasa olayı"""
    event_type: str    # FEAR_GREED, TOKEN_UNLOCK, LIQUIDATION, ECONOMIC, LISTING, TV_SIGNAL, CME_GAP, DEFI_TVL, STABLECOIN
    title: str
    detail: str
    impact: str        # HIGH, MEDIUM, LOW
    direction: str     # BULLISH, BEARISH, NEUTRAL
    timestamp: datetime
    coins: List[str]
    action: str


class AdvancedMarketScrapers:
    """
    Gelişmiş Piyasa Tarayıcıları
    
    Web scraping ile şunları takip eder:
    - Fear & Greed Index
    - Token Unlock takvimi
    - Likidasyon seviyeleri
    - Ekonomik takvim
    - Borsa duyuruları
    - TradingView sinyalleri
    - CME Gap Tracker
    - DeFi TVL Monitor
    - Stablecoin Flow
    """
    
    # Request headers to avoid blocks
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    def __init__(self):
        self.last_fetch = {}
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.cme_friday_close = None  # Track CME Friday close price
        
    # =========================================
    # 1. FEAR & GREED INDEX
    # =========================================
    def get_fear_greed_index(self) -> Dict:
        """
        Kripto Fear & Greed Index çek.
        
        Returns:
            {'value': 75, 'classification': 'Greed', 'direction': 'BULLISH/BEARISH'}
        """
        cache_key = 'fear_greed'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            # Alternative.me free API
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=10, headers=self.HEADERS)
            
            if response.status_code == 200:
                data = response.json()
                fng = data.get('data', [{}])[0]
                
                value = int(fng.get('value', 50))
                classification = fng.get('value_classification', 'Neutral')
                
                # Determine direction
                if value <= 25:
                    direction = 'BULLISH'  # Extreme Fear = Buy opportunity
                    action = "🟢 Extreme Fear = Alım fırsatı! Tarihsel olarak dip bölgesi."
                elif value >= 75:
                    direction = 'BEARISH'  # Extreme Greed = Sell risk
                    action = "🔴 Extreme Greed = Dikkat! Düzeltme riski yüksek."
                else:
                    direction = 'NEUTRAL'
                    action = "Nötr bölge - Normal piyasa koşulları."
                
                result = {
                    'value': value,
                    'classification': classification,
                    'direction': direction,
                    'action': action,
                    'timestamp': datetime.now()
                }
                
                self._set_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
        
        return {'value': 50, 'classification': 'Neutral', 'direction': 'NEUTRAL', 'action': 'Veri alınamadı'}
    
    # =========================================
    # 2. TOKEN UNLOCK CALENDAR
    # =========================================
    def get_token_unlocks(self, days_ahead: int = 7) -> List[MarketEvent]:
        """
        Yaklaşan token unlock'ları çek.
        
        Returns:
            List of upcoming token unlocks that may cause selling pressure
        """
        cache_key = 'token_unlocks'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        events = []
        
        try:
            # DefiLlama unlocks API (free)
            url = "https://api.llama.fi/unlocks"
            response = requests.get(url, timeout=15, headers=self.HEADERS)
            
            if response.status_code == 200:
                data = response.json()
                
                now = datetime.now()
                cutoff = now + timedelta(days=days_ahead)
                
                for unlock in data[:20]:  # Check top 20
                    try:
                        unlock_date = datetime.fromtimestamp(unlock.get('timestamp', 0))
                        
                        if now <= unlock_date <= cutoff:
                            coin = unlock.get('symbol', 'UNKNOWN')
                            usd_value = unlock.get('value', 0)
                            
                            if usd_value > 10_000_000:  # $10M+ unlock
                                impact = 'HIGH' if usd_value > 100_000_000 else 'MEDIUM'
                                
                                events.append(MarketEvent(
                                    event_type='TOKEN_UNLOCK',
                                    title=f"🔓 {coin} Token Unlock",
                                    detail=f"${usd_value/1e6:.1f}M unlock - {unlock_date.strftime('%d %b')}",
                                    impact=impact,
                                    direction='BEARISH',
                                    timestamp=unlock_date,
                                    coins=[coin],
                                    action=f"⚠️ {coin} satış baskısı riski - Unlock öncesi dikkatli ol!"
                                ))
                    except:
                        continue
                        
            self._set_cache(cache_key, events)
            
        except Exception as e:
            logger.warning(f"Token unlock fetch failed: {e}")
        
        return events
    
    # =========================================
    # 3. LIQUIDATION LEVELS (Coinglass)
    # =========================================
    def get_liquidation_levels(self, symbol: str = 'BTC') -> Dict:
        """
        Likidasyon seviyelerini çek.
        
        Fiyat bu seviyelere çekilir (liquidity grab).
        """
        cache_key = f'liquidations_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            # Coinglass has API but requires key
            # Alternative: scrape from page or use approximate calculation
            
            # Fallback: Calculate approximate liquidation levels from price
            # In real implementation, would scrape coinglass.com
            
            result = {
                'symbol': symbol,
                'long_liquidations': [],  # Price levels with clustered long liquidations
                'short_liquidations': [],  # Price levels with clustered short liquidations
                'note': 'Premium API gerekli - Coinglass verileri için API key ekleyin',
                'timestamp': datetime.now()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.warning(f"Liquidation data fetch failed: {e}")
        
        return {}
    
    # =========================================
    # 4. ECONOMIC CALENDAR
    # =========================================
    def get_economic_events(self, days_ahead: int = 3) -> List[MarketEvent]:
        """
        Önemli ekonomik olayları çek (Fed, CPI, etc).
        
        Crypto piyasasını etkileyen makro olaylar.
        """
        cache_key = 'economic_events'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        events = []
        
        try:
            # Known important events (static schedule - would be scraped in production)
            # Fed meetings: ~every 6 weeks
            # CPI release: 2nd week of each month
            
            # Use free forexfactory or similar
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            response = requests.get(url, timeout=10, headers=self.HEADERS)
            
            if response.status_code == 200:
                data = response.json()
                
                high_impact_keywords = ['CPI', 'NFP', 'FOMC', 'Fed', 'Interest Rate', 'GDP', 'Unemployment']
                
                for event in data[:30]:
                    title = event.get('title', '')
                    impact = event.get('impact', '')
                    
                    if impact == 'High' or any(kw in title for kw in high_impact_keywords):
                        country = event.get('country', 'US')
                        date_str = event.get('date', '')
                        
                        try:
                            event_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
                        except:
                            event_date = datetime.now() + timedelta(days=1)
                        
                        events.append(MarketEvent(
                            event_type='ECONOMIC',
                            title=f"📅 {title}",
                            detail=f"{country} - {event_date.strftime('%d %b %H:%M')}",
                            impact='HIGH',
                            direction='NEUTRAL',  # Can go either way
                            timestamp=event_date,
                            coins=['BTC', 'ETH'],  # All crypto affected
                            action=f"⚠️ Yüksek volatilite bekleniyor - Pozisyon boyutunu düşür!"
                        ))
            
            self._set_cache(cache_key, events)
            
        except Exception as e:
            logger.debug(f"Economic calendar fetch failed: {e}")
        
        return events
    
    # =========================================
    # 5. EXCHANGE LISTINGS (Binance)
    # =========================================
    def get_new_listings(self) -> List[MarketEvent]:
        """
        Yeni borsa listinglerini çek.
        
        Yeni listing = Pump potansiyeli (dikkatli!)
        """
        cache_key = 'new_listings'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        events = []
        
        try:
            # Binance announcements API
            url = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&pageNo=1&pageSize=20"
            response = requests.get(url, timeout=10, headers=self.HEADERS)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', {}).get('catalogs', [{}])[0].get('articles', [])
                
                listing_keywords = ['Will List', 'Lists', 'New Listing', 'Adds', 'Trading Starts']
                
                for article in articles[:10]:
                    title = article.get('title', '')
                    
                    if any(kw.lower() in title.lower() for kw in listing_keywords):
                        # Extract coin symbol from title
                        coins = self._extract_coins_from_text(title)
                        
                        release_date = article.get('releaseDate', 0)
                        try:
                            event_date = datetime.fromtimestamp(release_date / 1000)
                        except:
                            event_date = datetime.now()
                        
                        events.append(MarketEvent(
                            event_type='LISTING',
                            title=f"🚀 {title[:60]}...",
                            detail=f"Binance - {event_date.strftime('%d %b')}",
                            impact='HIGH',
                            direction='BULLISH',
                            timestamp=event_date,
                            coins=coins,
                            action=f"🎯 Yeni listing! Pump potansiyeli var ama dikkatli ol - dump da olabilir!"
                        ))
            
            self._set_cache(cache_key, events)
            
        except Exception as e:
            logger.debug(f"Exchange listings fetch failed: {e}")
        
        return events
    
    # =========================================
    # 6. TRADINGVIEW TECHNICALS
    # =========================================
    def get_tradingview_signals(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        TradingView teknik analiz özetini çek.
        
        Oscillators + Moving Averages = Overall signal
        """
        cache_key = f'tv_signals_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            # TradingView has a widget API for technicals
            # We'll use their technical analysis summary
            
            # Map symbol to TradingView format
            tv_symbol = f"BINANCE:{symbol}"
            
            # TradingView technicals endpoint (undocumented but works)
            url = f"https://scanner.tradingview.com/crypto/scan"
            
            payload = {
                "symbols": {"tickers": [tv_symbol], "query": {"types": []}},
                "columns": [
                    "Recommend.All", "Recommend.MA", "Recommend.Other",
                    "RSI", "Mom", "MACD.macd", "Stoch.K",
                    "ADX", "CCI20", "AO"
                ]
            }
            
            response = requests.post(url, json=payload, timeout=10, headers={
                **self.HEADERS,
                'Content-Type': 'application/json'
            })
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('data'):
                    values = data['data'][0].get('d', [])
                    
                    # Parse signals (-1 to 1 scale)
                    overall = values[0] if len(values) > 0 else 0
                    ma_signal = values[1] if len(values) > 1 else 0
                    osc_signal = values[2] if len(values) > 2 else 0
                    rsi = values[3] if len(values) > 3 else -1  # -1 = veri yok
                    
                    # Convert to action
                    if overall > 0.5:
                        action = 'STRONG_BUY'
                        direction = 'BULLISH'
                        emoji = '🟢🟢'
                    elif overall > 0.1:
                        action = 'BUY'
                        direction = 'BULLISH'
                        emoji = '🟢'
                    elif overall < -0.5:
                        action = 'STRONG_SELL'
                        direction = 'BEARISH'
                        emoji = '🔴🔴'
                    elif overall < -0.1:
                        action = 'SELL'
                        direction = 'BEARISH'
                        emoji = '🔴'
                    else:
                        action = 'NEUTRAL'
                        direction = 'NEUTRAL'
                        emoji = '⚪'
                    
                    result = {
                        'symbol': symbol,
                        'overall': overall,
                        'ma_signal': ma_signal,
                        'oscillator_signal': osc_signal,
                        'rsi': rsi,
                        'action': action,
                        'direction': direction,
                        'emoji': emoji,
                        'summary': f"{emoji} TradingView: {action} (RSI: {rsi:.0f})",
                        'timestamp': datetime.now()
                    }
                    
                    self._set_cache(cache_key, result)
                    return result
                    
        except Exception as e:
            logger.warning(f"TradingView signals fetch failed: {e}")
        
        return {
            'symbol': symbol,
            'action': 'NEUTRAL',
            'direction': 'NEUTRAL',
            'summary': '⚪ TradingView: Veri alınamadı',
            'timestamp': datetime.now()
        }
    
    def get_realtime_price(self, symbol: str) -> float:
        """
        TradingView scanner üzerinden canlı fiyat çek (API Fallback).
        API key gerektirmez, public endpoint.
        """
        try:
            url = "https://scanner.tradingview.com/crypto/scan"
            tv_symbol = f"BINANCE:{symbol}"
            
            payload = {
                "symbols": {"tickers": [tv_symbol], "query": {"types": []}},
                "columns": ["close"]
            }
            
            response = requests.post(url, json=payload, timeout=5, headers=self.HEADERS)
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    return float(data['data'][0]['d'][0])
        except Exception as e:
            logger.debug(f"TV price fetch failed: {e}")
        return 0.0
    
    # =========================================
    # AGGREGATE ALL DATA
    # =========================================
    def get_all_market_intelligence(self) -> Dict:
        """Tüm market intelligence verilerini topla"""
        
        # Fear & Greed
        fng = self.get_fear_greed_index()
        
        # Token Unlocks
        unlocks = self.get_token_unlocks(days_ahead=7)
        
        # Economic Events
        economic = self.get_economic_events(days_ahead=3)
        
        # Exchange Listings
        listings = self.get_new_listings()
        
        # TradingView Signals for major coins
        tv_btc = self.get_tradingview_signals('BTCUSDT')
        tv_eth = self.get_tradingview_signals('ETHUSDT')
        
        # Combine all events
        all_events = unlocks + economic + listings
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x.timestamp)
        
        return {
            'fear_greed': fng,
            'upcoming_events': all_events[:10],  # Top 10 events
            'tradingview': {
                'BTC': tv_btc,
                'ETH': tv_eth
            },
            'summary': self._generate_summary(fng, all_events, tv_btc),
            'timestamp': datetime.now()
        }
    
    def _generate_summary(self, fng: Dict, events: List, tv_btc: Dict) -> str:
        """Market intelligence özeti oluştur"""
        
        summary = ""
        
        # Fear & Greed Summary
        fng_value = fng.get('value', 50)
        if fng_value <= 25:
            summary += "🟢 Extreme Fear - Tarihsel alım bölgesi! "
        elif fng_value >= 75:
            summary += "🔴 Extreme Greed - Dikkat, düzeltme riski! "
        
        # High impact events
        high_impact = [e for e in events if e.impact == 'HIGH']
        if high_impact:
            summary += f"⚠️ {len(high_impact)} önemli olay yaklaşıyor. "
        
        # TradingView
        tv_action = tv_btc.get('action', 'NEUTRAL')
        if 'BUY' in tv_action:
            summary += "📊 TradingView: Teknik olarak alım sinyali. "
        elif 'SELL' in tv_action:
            summary += "📊 TradingView: Teknik olarak satış sinyali. "
        
        return summary if summary else "Piyasa normal koşullarda."
    
    # =========================================
    # FORMAT FOR TELEGRAM
    # =========================================
    def format_for_telegram(self) -> str:
        """Telegram için market intelligence raporu"""
        
        intel = self.get_all_market_intelligence()
        
        msg = "🌐 *Gelişmiş Piyasa İstihbaratı*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # Fear & Greed
        fng = intel.get('fear_greed', {})
        fng_value = fng.get('value', 50)
        fng_class = fng.get('classification', 'Neutral')
        fng_emoji = "🟢" if fng_value <= 30 else "🔴" if fng_value >= 70 else "🟡"
        
        msg += f"😱 *Fear & Greed:* {fng_emoji} {fng_value}/100 ({fng_class})\n"
        msg += f"_{fng.get('action', '')}_\n\n"
        
        # TradingView Signals
        tv = intel.get('tradingview', {})
        msg += f"📊 *TradingView Sinyalleri:*\n"
        
        for coin, tv_data in tv.items():
            msg += f"• {coin}: {tv_data.get('summary', 'N/A')}\n"
        msg += "\n"
        
        # Upcoming Events
        events = intel.get('upcoming_events', [])
        if events:
            msg += f"📅 *Yaklaşan Olaylar ({len(events)}):*\n"
            for event in events[:5]:  # Top 5
                emoji = "🔓" if event.event_type == 'TOKEN_UNLOCK' else "📅" if event.event_type == 'ECONOMIC' else "🚀"
                msg += f"• {emoji} {event.title}\n"
                msg += f"  _{event.detail}_\n"
        
        # Summary
        summary = intel.get('summary', '')
        if summary:
            msg += f"\n💡 *Özet:* {summary}\n"
        
        msg += f"\n⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        return msg
    
    # =========================================
    # HELPERS
    # =========================================
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
    
    def _extract_coins_from_text(self, text: str) -> List[str]:
        """Extract coin symbols from text"""
        coins = []
        
        # Common coin patterns
        patterns = [
            r'\b([A-Z]{2,5})/USDT\b',
            r'\b([A-Z]{2,5})\s*\(',
            r'Lists?\s+([A-Z]{2,5})\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            coins.extend(matches)
        
        return list(set(coins))
    
    # =========================================
    # 7. CME GAP TRACKER
    # =========================================
    def get_cme_gap(self) -> Dict:
        """
        Bitcoin CME Gap'i tespit et.
        
        CME (Chicago Mercantile Exchange) hafta sonları kapalı.
        Hafta sonu BTC fiyatı değişirse, Pazartesi açılışında gap oluşur.
        BTC genellikle bu gap'i kapatmak için o seviyeye döner.
        
        Returns:
            {'has_gap': True, 'gap_price': 104000, 'current_price': 106000, 'gap_percent': -1.9}
        """
        cache_key = 'cme_gap'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            # Get current BTC price
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            response = requests.get(url, timeout=5)
            current_price = float(response.json()['price']) if response.status_code == 200 else 0
            
            # CME closes Friday 5PM EST, opens Sunday 6PM EST
            # We need to track Friday close and compare to current
            
            now = datetime.now()
            day_of_week = now.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
            
            # Get historical klines to find Friday close (approximate)
            kline_url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=4h&limit=50"
            kline_response = requests.get(kline_url, timeout=10)
            
            if kline_response.status_code == 200:
                klines = kline_response.json()
                
                # Find Friday close (CME closes around 22:00 UTC Friday)
                friday_close = None
                for k in reversed(klines):
                    kline_time = datetime.fromtimestamp(k[0] / 1000)
                    if kline_time.weekday() == 4 and kline_time.hour >= 20:  # Friday evening
                        friday_close = float(k[4])  # Close price
                        break
                
                if friday_close and current_price:
                    gap_percent = ((current_price - friday_close) / friday_close) * 100
                    
                    # Only report if gap is significant (>1%)
                    if abs(gap_percent) > 1:
                        direction = 'BULLISH' if gap_percent < 0 else 'BEARISH'
                        action = f"Gap'i kapatmak için fiyat ${friday_close:,.0f}'a dönebilir!" if abs(gap_percent) > 2 else "Küçük gap - izle."
                        
                        result = {
                            'has_gap': True,
                            'gap_price': friday_close,
                            'current_price': current_price,
                            'gap_percent': gap_percent,
                            'direction': direction,
                            'action': action,
                            'timestamp': datetime.now()
                        }
                    else:
                        result = {
                            'has_gap': False,
                            'gap_price': friday_close,
                            'current_price': current_price,
                            'gap_percent': gap_percent,
                            'direction': 'NEUTRAL',
                            'action': 'CME gap yok veya kapatıldı.',
                            'timestamp': datetime.now()
                        }
                    
                    self._set_cache(cache_key, result)
                    return result
                    
        except Exception as e:
            logger.warning(f"CME gap fetch failed: {e}")
        
        return {
            'has_gap': False,
            'gap_price': 0,
            'current_price': 0,
            'gap_percent': 0,
            'direction': 'NEUTRAL',
            'action': 'CME gap verisi alınamadı',
            'timestamp': datetime.now()
        }
    
    # =========================================
    # 8. DEFI TVL MONITOR
    # =========================================
    def get_defi_tvl(self) -> Dict:
        """
        DeFi Total Value Locked (TVL) takibi.
        
        TVL düşüşü = Risk-off sentiment, bearish
        TVL artışı = Risk-on sentiment, bullish
        
        Returns:
            {'total_tvl': 150B, 'change_24h': -2.5%, 'direction': 'BEARISH'}
        """
        cache_key = 'defi_tvl'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            # DefiLlama TVL API (free)
            url = "https://api.llama.fi/v2/historicalChainTvl"
            response = requests.get(url, timeout=15, headers=self.HEADERS)
            
            if response.status_code == 200:
                data = response.json()
                
                if len(data) >= 2:
                    current_tvl = data[-1].get('tvl', 0)
                    yesterday_tvl = data[-2].get('tvl', 0)
                    
                    if yesterday_tvl > 0:
                        change_24h = ((current_tvl - yesterday_tvl) / yesterday_tvl) * 100
                        
                        # Determine direction
                        if change_24h > 3:
                            direction = 'BULLISH'
                            action = "🟢 DeFi TVL artıyor! Risk-on sentiment, bullish."
                        elif change_24h < -3:
                            direction = 'BEARISH'
                            action = "🔴 DeFi TVL düşüyor! Risk-off sentiment, dikkatli ol."
                        else:
                            direction = 'NEUTRAL'
                            action = "DeFi TVL stabil."
                        
                        result = {
                            'total_tvl': current_tvl,
                            'total_tvl_formatted': f"${current_tvl/1e9:.1f}B",
                            'change_24h': change_24h,
                            'direction': direction,
                            'action': action,
                            'timestamp': datetime.now()
                        }
                        
                        self._set_cache(cache_key, result)
                        return result
                        
        except Exception as e:
            logger.warning(f"DeFi TVL fetch failed: {e}")
        
        return {
            'total_tvl': 0,
            'total_tvl_formatted': 'N/A',
            'change_24h': 0,
            'direction': 'NEUTRAL',
            'action': 'DeFi TVL verisi alınamadı',
            'timestamp': datetime.now()
        }
    
    # =========================================
    # 9. STABLECOIN FLOW (USDT/USDC Minting)
    # =========================================
    def get_stablecoin_flow(self) -> Dict:
        """
        Stablecoin mint/burn takibi.
        
        Yeni stablecoin mint = Alım hazırlığı (bullish)
        Stablecoin burn = Çıkış yapılıyor (bearish)
        
        Returns:
            {'usdt_supply': 120B, 'change_7d': +500M, 'direction': 'BULLISH'}
        """
        cache_key = 'stablecoin_flow'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            # DefiLlama stablecoins API
            url = "https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1"  # USDT
            response = requests.get(url, timeout=15, headers=self.HEADERS)
            
            if response.status_code == 200:
                data = response.json()
                
                if len(data) >= 7:
                    current_supply = data[-1].get('totalCirculating', {}).get('peggedUSD', 0)
                    week_ago_supply = data[-7].get('totalCirculating', {}).get('peggedUSD', 0)
                    
                    if week_ago_supply > 0:
                        change_7d = current_supply - week_ago_supply
                        change_percent = (change_7d / week_ago_supply) * 100
                        
                        # Determine direction
                        if change_7d > 500_000_000:  # $500M+ mint
                            direction = 'BULLISH'
                            action = f"🟢 ${change_7d/1e6:.0f}M USDT mint edildi! Alım hazırlığı."
                        elif change_7d < -500_000_000:  # $500M+ burn
                            direction = 'BEARISH'
                            action = f"🔴 ${abs(change_7d)/1e6:.0f}M USDT yakıldı! Çıkış sinyali."
                        else:
                            direction = 'NEUTRAL'
                            action = "Stablecoin akışı normal."
                        
                        result = {
                            'usdt_supply': current_supply,
                            'usdt_supply_formatted': f"${current_supply/1e9:.1f}B",
                            'change_7d': change_7d,
                            'change_7d_formatted': f"${change_7d/1e6:+.0f}M",
                            'change_percent': change_percent,
                            'direction': direction,
                            'action': action,
                            'timestamp': datetime.now()
                        }
                        
                        self._set_cache(cache_key, result)
                        return result
                        
        except Exception as e:
            logger.warning(f"Stablecoin flow fetch failed: {e}")
        
        return {
            'usdt_supply': 0,
            'usdt_supply_formatted': 'N/A',
            'change_7d': 0,
            'change_7d_formatted': 'N/A',
            'change_percent': 0,
            'direction': 'NEUTRAL',
            'action': 'Stablecoin verisi alınamadı',
            'timestamp': datetime.now()
        }

