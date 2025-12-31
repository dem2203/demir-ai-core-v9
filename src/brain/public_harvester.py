# -*- coding: utf-8 -*-
"""
DEMIR AI - PUBLIC DATA HARVESTER (API-LESS / FREE)
==================================================
Bu modül, ücretli API key gerektirmeyen veya 'hidden public' endpointleri
kullanarak veri toplar. Playwright kullanmaz (Railway uyumlu).

KAPSAM:
1. DefiLlama (Macro On-Chain, TVL, Stablecoins)
2. Alternative.me (Fear & Greed Index)
3. News RSS (The Block, Cointelegraph)
4. Coinglass (Public API Endpoint attempts)
"""
import logging
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import random
import time

logger = logging.getLogger("PUBLIC_HARVESTER")

class PublicHarvester:
    """
    Ücretsiz ve Public veri toplayıcı.
    Anti-Bot korumalarına takılmamak için 'requests' ve 'headers' manipülasyonu kullanır.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        })
        self.cache = {}
        self.cache_expiry = {}

    def _get_cached_or_fetch(self, key: str, fetch_func, ttl_seconds: int = 300):
        """Basit in-memory cache mekanizması."""
        now = datetime.now()
        if key in self.cache and key in self.cache_expiry:
            if now < self.cache_expiry[key]:
                return self.cache[key]
        
        try:
            data = fetch_func()
            if data:
                self.cache[key] = data
                self.cache_expiry[key] = now + timedelta(seconds=ttl_seconds)
                return data
        except Exception as e:
            logger.error(f"Harvester fetch error [{key}]: {e}")
            # Cache varsa eski veriyi dön, yoksa None
            return self.cache.get(key)
        
        return None

    # ==========================================================
    # 1. MACRO ON-CHAIN (DefiLlama) - Item 5
    # ==========================================================
    def fetch_defillama_macro(self) -> Dict:
        """DefiLlama üzerinden global TVL ve Stablecoin verileri."""
        
        def _fetch():
            # 1. Global TVL
            tvl_resp = self.session.get("https://api.llama.fi/v2/chains")
            tvl_data = tvl_resp.json()
            
            total_tvl = sum(c.get('tvl', 0) for c in tvl_data)
            
            # 2. Stablecoins
            stable_resp = self.session.get("https://stablecoins.llama.fi/stablecoins?includePrices=true")
            stable_data = stable_resp.json()
            
            stables = stable_data.get('peggedAssets', [])
            total_stable_mcap = 0
            usdt_mcap = 0
            usdc_mcap = 0
            
            for s in stables:
                mcap = s.get('circulating', {}).get('peggedUSD', 0)
                total_stable_mcap += mcap
                if s.get('symbol') == 'USDT':
                    usdt_mcap = mcap
                elif s.get('symbol') == 'USDC':
                    usdc_mcap = mcap
            
            return {
                "total_tvl": total_tvl,
                "stablecoin_mcap": total_stable_mcap,
                "usdt_dominance": (usdt_mcap / total_stable_mcap * 100) if total_stable_mcap > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }

        return self._get_cached_or_fetch("defillama", _fetch, ttl_seconds=3600) # 1 saat cache

    # ==========================================================
    # 2. SENTIMENT & NEWS (RSS Feeds) - Item 6
    # ==========================================================
    def fetch_news_sentiment(self) -> Dict:
        """RSS Feedleri üzerinden son haberleri ve basit sentimenti çeker."""
        
        rss_urls = [
            "https://cointelegraph.com/rss",
            "https://decrypt.co/feed",
            "https://theblock.co/rss"
        ]
        
        def _fetch():
            headlines = []
            
            for url in rss_urls:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:3]: # Her kaynaktan son 3 haber
                        headlines.append({
                            "title": entry.title,
                            "link": entry.link,
                            "published": entry.get('published', '')
                        })
                except:
                    continue
            
            # Basit Keyword Sentiment
            bullish_keywords = ['surge', 'soar', 'record', 'high', 'bull', 'adoption', 'approve', 'gain']
            bearish_keywords = ['crash', 'drop', 'ban', 'stuck', 'bear', 'hack', 'fraud', 'lawsuit', 'sec']
            
            score = 0
            for h in headlines:
                title_lower = h['title'].lower()
                if any(k in title_lower for k in bullish_keywords):
                    score += 1
                if any(k in title_lower for k in bearish_keywords):
                    score -= 1
            
            sentiment = "NEUTRAL"
            if score > 2: sentiment = "BULLISH"
            elif score < -2: sentiment = "BEARISH"
            
            return {
                "sentiment": sentiment,
                "score": score,
                "headlines": headlines[:5],
                "source_count": len(rss_urls)
            }

        return self._get_cached_or_fetch("news_rss", _fetch, ttl_seconds=1800) # 30 dk cache

    # ==========================================================
    # 3. FEAR & GREED INDEX
    # ==========================================================
    def fetch_fear_greed(self) -> Dict:
        """Alternative.me public API."""
        
        def _fetch():
            resp = self.session.get("https://api.alternative.me/fng/")
            data = resp.json()
            item = data['data'][0]
            return {
                "value": int(item['value']),
                "classification": item['value_classification']
            }
            
        return self._get_cached_or_fetch("fng", _fetch, ttl_seconds=3600)

    # ==========================================================
    # 4. COINGLASS (Hidden Public Endpoint) - Item 3
    # ==========================================================
    def fetch_coinglass_oi(self, symbol="BTC") -> Dict:
        """
        CoinGlass public endpoint denemesi (Open Interest).
        Not: Bu endpointler sık sık değişebilir veya bloklanabilir.
        """
        def _fetch():
            # Coinglass public API - genellikle çalışır ama headers lazım
            url = f"https://fapi.coinglass.com/api/openInterest/v3/chart?symbol={symbol}&interval=0"
            
            try:
                # Rastgele gecikme (anti-bot)
                time.sleep(random.uniform(0.5, 1.5))
                resp = self.session.get(url, timeout=5)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('success'):
                        data_list = data.get('data', {}).get('dataMap', [])
                        # Son veri noktasını al
                        if data_list:
                             # Veri yapısını parse et (basitleştirilmiş)
                            last_point = list(data_list.values())[-1][-1] 
                            return {
                                "open_interest": last_point,
                                "source": "coinglass_public_api"
                            }
            except Exception as e:
                logger.warning(f"Coinglass fetch failed: {e}")
                
            return None

        # Şimdilik devre dışı bırakıyorum çünkü çok sıkı koruma var, boş veri dönerse akışı bozmasın
        # return self._get_cached_or_fetch("cg_oi", _fetch, ttl_seconds=300)
        return {"open_interest": None, "note": "Endpoint protected"}

    # ==========================================================
    # 5. OPTIONS DATA (Item 1 Proxy)
    # ==========================================================
    def fetch_options_sentiment(self, symbol="BTC") -> Dict:
        """
        CoinGlass veya benzeri kaynaklardan Put/Call Ratio tahmini.
        API keysiz options verisi çok zordur, bu metot public özetleri tarar.
        """
        def _fetch():
            # Fallback: Coinglass public options summary page JSON (Mobile API emulation)
            url = f"https://fapi.coinglass.com/api/options/info?symbol={symbol}"
            
            try:
                time.sleep(random.uniform(0.5, 1.5))
                resp = self.session.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('success'):
                         # Basic Options Info
                         info = data.get('data', {})
                         # P/C Ratio isn't always directly here, but we check
                         return {
                             "source": "coinglass_public_api",
                             "volume_24h": info.get('volUsd'),
                             "open_interest": info.get('oiUsd')
                         }
            except:
                pass
            
            # Alternative: Deribit Public API (Official & Free!)
            # Deribit option verileri serbesttir.
            try:
                # Get BTC volatility index or general options summary
                d_url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={symbol}&kind=option"
                resp = self.session.get(d_url, timeout=5)
                if resp.status_code == 200:
                    opts = resp.json().get('result', [])
                    # Calculate simple Put/Call Volume Ratio from active options
                    puts_vol = 0
                    calls_vol = 0
                    for opt in opts:
                        vol = opt.get('volume', 0)
                        instr = opt.get('instrument_name', '')
                        if '-P' in instr:
                            puts_vol += vol
                        elif '-C' in instr:
                            calls_vol += vol
                    
                    pc_ratio = puts_vol / calls_vol if calls_vol > 0 else 1.0
                    
                    sentiment = "NEUTRAL"
                    if pc_ratio > 1.2: sentiment = "BEARISH" # Hedging/Fear high
                    elif pc_ratio < 0.7: sentiment = "BULLISH" # Speculation high
                    
                    return {
                        "source": "deribit_api",
                        "put_volume": puts_vol,
                        "call_volume": calls_vol,
                        "pc_ratio_volume": pc_ratio,
                        "sentiment": sentiment
                    }
            except Exception as e:
                logger.error(f"Deribit options fetch failed: {e}")
                
            return None

        return self._get_cached_or_fetch("options_sentiment", _fetch, ttl_seconds=600)
