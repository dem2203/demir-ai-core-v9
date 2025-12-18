# -*- coding: utf-8 -*-
"""
DEMIR AI - Google Trends Scraper
Bitcoin arama hacmi analizi.

PHASE 66: Retail Sentiment via Search Volume
- PyTrends API (Google Trends)
- Bitcoin, Crypto arama hacmi
- %30+ artış = Retail FOMO uyarısı
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger("GOOGLE_TRENDS")


class GoogleTrendsScraper:
    """
    Google Trends Scraper for Bitcoin Search Volume
    
    Uses pytrends API to fetch search interest data.
    High search volume often indicates retail FOMO (buy signal reversal).
    """
    
    def __init__(self):
        self.pytrends = None
        self._init_pytrends()
    
    def _init_pytrends(self):
        """Initialize pytrends connection."""
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl='en-US', tz=360)
            logger.info("✅ PyTrends initialized")
        except ImportError:
            logger.warning("⚠️ pytrends not installed. Run: pip install pytrends")
            self.pytrends = None
        except Exception as e:
            logger.warning(f"PyTrends init failed: {e}")
            self.pytrends = None
    
    def get_bitcoin_interest(self, days: int = 7) -> Dict:
        """
        Get Bitcoin search interest over time.
        
        Returns:
            {
                'current_interest': 75,  # 0-100 scale
                'avg_7d': 65,
                'change_pct': 15.4,  # vs 7-day average
                'is_fomo': False,  # True if +30%
                'sentiment': 'NEUTRAL',
                'confidence': 50
            }
        """
        if not self.pytrends:
            return self._empty_result()
        
        try:
            # Build payload
            self.pytrends.build_payload(
                kw_list=['Bitcoin'],
                cat=0,
                timeframe=f'now {days}-d',
                geo='',
                gprop=''
            )
            
            # Get interest over time
            data = self.pytrends.interest_over_time()
            
            if data.empty:
                return self._empty_result()
            
            # Calculate metrics
            values = data['Bitcoin'].tolist()
            current = values[-1] if values else 50
            avg = sum(values) / len(values) if values else 50
            
            change_pct = ((current / avg) - 1) * 100 if avg > 0 else 0
            is_fomo = change_pct > 30
            
            # Determine sentiment
            # High search interest often means retail FOMO = potential top
            if is_fomo:
                sentiment = 'BEARISH'  # Contrarian
                confidence = min(70, 50 + abs(change_pct) * 0.5)
            elif change_pct > 15:
                sentiment = 'NEUTRAL'
                confidence = 45
            elif change_pct < -20:
                sentiment = 'BULLISH'  # Contrarian - low interest = accumulation
                confidence = min(65, 50 + abs(change_pct) * 0.4)
            else:
                sentiment = 'NEUTRAL'
                confidence = 40
            
            return {
                'current_interest': current,
                'avg_7d': avg,
                'change_pct': change_pct,
                'is_fomo': is_fomo,
                'sentiment': sentiment,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Google Trends fetch failed: {e}")
            return self._empty_result()
    
    def get_related_queries(self) -> Dict:
        """Get top related queries for Bitcoin."""
        if not self.pytrends:
            return {'queries': [], 'available': False}
        
        try:
            self.pytrends.build_payload(kw_list=['Bitcoin'])
            related = self.pytrends.related_queries()
            
            top = related.get('Bitcoin', {}).get('top', None)
            rising = related.get('Bitcoin', {}).get('rising', None)
            
            queries = []
            if top is not None and not top.empty:
                queries.extend(top['query'].head(5).tolist())
            if rising is not None and not rising.empty:
                queries.extend(rising['query'].head(5).tolist())
            
            return {
                'queries': queries[:10],
                'available': True
            }
        except Exception as e:
            logger.warning(f"Related queries failed: {e}")
            return {'queries': [], 'available': False}
    
    def _empty_result(self) -> Dict:
        """Return empty result."""
        return {
            'current_interest': 0,
            'avg_7d': 0,
            'change_pct': 0,
            'is_fomo': False,
            'sentiment': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def get_bitcoin_trends() -> Dict:
    """Get Bitcoin trends."""
    scraper = GoogleTrendsScraper()
    return scraper.get_bitcoin_interest()
