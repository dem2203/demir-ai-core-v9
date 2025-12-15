"""
SENTIMENT ANALYZER - Fear & Greed Index
========================================
Phase 32: Predictive Intelligence

Fetches market sentiment data to provide early warnings.
Free API Source: alternative.me
"""

import logging
import aiohttp
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger("SENTIMENT")

class SentimentAnalyzer:
    """
    Market sentiment analysis using Fear & Greed Index.
    
    Index Values:
    0-24: Extreme Fear (Potential buy signal)
    25-49: Fear
    50-74: Greed
    75-100: Extreme Greed (Potential sell signal)
    """
    
    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    
    def __init__(self):
        self.cached_data = {}
        self.last_fetch_time = None
        self.cache_duration = 3600  # 1 hour cache
    
    async def get_fear_greed(self) -> Dict:
        """
        Fetch current Fear & Greed Index.
        
        Returns:
            {
                'fear_greed_index': 25,
                'fear_greed_label': 'Extreme Fear',
                'timestamp': '...'
            }
        """
        try:
            # Check cache
            if self.last_fetch_time:
                elapsed = (datetime.now() - self.last_fetch_time).total_seconds()
                if elapsed < self.cache_duration and self.cached_data:
                    return self.cached_data
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.FEAR_GREED_URL, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('data'):
                            fg_data = data['data'][0]
                            
                            result = {
                                'fear_greed_index': int(fg_data.get('value', 50)),
                                'fear_greed_label': fg_data.get('value_classification', 'Neutral'),
                                'timestamp': fg_data.get('timestamp', ''),
                                'time_until_update': fg_data.get('time_until_update', '')
                            }
                            
                            self.cached_data = result
                            self.last_fetch_time = datetime.now()
                            
                            logger.info(f"😰 Fear & Greed: {result['fear_greed_index']} ({result['fear_greed_label']})")
                            return result
            
            return self._empty_result()
            
        except Exception as e:
            logger.error(f"Fear & Greed fetch failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        """Return neutral value if fetch fails."""
        return {
            'fear_greed_index': 50,
            'fear_greed_label': 'Neutral',
            'timestamp': '',
            'time_until_update': ''
        }
    
    def get_sentiment_signal(self, fg_index: int) -> str:
        """
        Convert Fear & Greed to trading signal.
        """
        if fg_index <= 20:
            return "EXTREME_FEAR_BUY"
        elif fg_index <= 40:
            return "FEAR_CAUTION"
        elif fg_index >= 80:
            return "EXTREME_GREED_SELL"
        elif fg_index >= 60:
            return "GREED_CAUTION"
        else:
            return "NEUTRAL"


# Quick test
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        analyzer = SentimentAnalyzer()
        result = await analyzer.get_fear_greed()
        print(f"\nFear & Greed Index: {result['fear_greed_index']}")
        print(f"Classification: {result['fear_greed_label']}")
        print(f"Signal: {analyzer.get_sentiment_signal(result['fear_greed_index'])}")
    
    asyncio.run(test())
