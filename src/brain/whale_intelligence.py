# -*- coding: utf-8 -*-
"""
DEMIR AI - Whale Intelligence (API-based)
Büyük trade ve pozisyon analizi - Playwright gerektirmez.

PHASE 53: Fixed Whale Intel
- Binance Large Trades API
- Open Interest analizi
- Long/Short ratio (gerçek)
- Funding rate sentiment
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("WHALE_INTEL")


class WhaleIntelligence:
    """
    Whale Intelligence - API Based (No Playwright)
    
    Veri Kaynakları:
    1. Binance Futures - Open Interest
    2. Binance Futures - Long/Short Ratio
    3. Binance Futures - Top Trader Positions
    4. Binance Futures - Funding Rate
    
    Tüm veriler GERÇEK ve API'dan geliyor.
    """
    
    # Binance Futures API
    BINANCE_FUTURES = "https://fapi.binance.com"
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 60  # 60 saniye cache
    
    def get_open_interest(self, symbol: str = 'BTCUSDT') -> Dict:
        """Open Interest verisi."""
        try:
            resp = requests.get(
                f"{self.BINANCE_FUTURES}/fapi/v1/openInterest",
                params={'symbol': symbol},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'symbol': symbol,
                    'open_interest': float(data.get('openInterest', 0)),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.warning(f"OI fetch error: {e}")
        return {'open_interest': 0}
    
    def get_long_short_ratio(self, symbol: str = 'BTCUSDT', period: str = '1h') -> Dict:
        """Top trader long/short ratio."""
        try:
            # Global L/S ratio
            resp = requests.get(
                f"{self.BINANCE_FUTURES}/futures/data/globalLongShortAccountRatio",
                params={'symbol': symbol, 'period': period, 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    ratio = float(data[0].get('longShortRatio', 1))
                    long_pct = (ratio / (1 + ratio)) * 100
                    short_pct = 100 - long_pct
                    
                    return {
                        'symbol': symbol,
                        'long_short_ratio': ratio,
                        'long_pct': round(long_pct, 1),
                        'short_pct': round(short_pct, 1),
                        'bias': 'LONG' if ratio > 1.1 else 'SHORT' if ratio < 0.9 else 'NEUTRAL',
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            logger.warning(f"L/S ratio error: {e}")
        return {'long_short_ratio': 1, 'bias': 'NEUTRAL', 'long_pct': 50, 'short_pct': 50}
    
    def get_top_trader_positions(self, symbol: str = 'BTCUSDT') -> Dict:
        """Top trader pozisyon oranları."""
        try:
            resp = requests.get(
                f"{self.BINANCE_FUTURES}/futures/data/topLongShortPositionRatio",
                params={'symbol': symbol, 'period': '1h', 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    ratio = float(data[0].get('longShortRatio', 1))
                    return {
                        'top_trader_ratio': ratio,
                        'top_bias': 'LONG' if ratio > 1.15 else 'SHORT' if ratio < 0.85 else 'NEUTRAL',
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            logger.warning(f"Top trader error: {e}")
        return {'top_trader_ratio': 1, 'top_bias': 'NEUTRAL'}
    
    def get_funding_rate(self, symbol: str = 'BTCUSDT') -> Dict:
        """Funding rate - market sentiment göstergesi."""
        try:
            resp = requests.get(
                f"{self.BINANCE_FUTURES}/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    rate = float(data[0].get('fundingRate', 0))
                    # Funding rate > 0.01% = aşırı long, < -0.01% = aşırı short
                    rate_pct = rate * 100
                    
                    if rate_pct > 0.05:
                        sentiment = 'EXTREME_LONG'
                        bias = 'SHORT'  # Counter-trade
                    elif rate_pct > 0.01:
                        sentiment = 'BULLISH'
                        bias = 'NEUTRAL'
                    elif rate_pct < -0.05:
                        sentiment = 'EXTREME_SHORT'
                        bias = 'LONG'  # Counter-trade
                    elif rate_pct < -0.01:
                        sentiment = 'BEARISH'
                        bias = 'NEUTRAL'
                    else:
                        sentiment = 'NEUTRAL'
                        bias = 'NEUTRAL'
                    
                    return {
                        'funding_rate': rate,
                        'funding_rate_pct': round(rate_pct, 4),
                        'sentiment': sentiment,
                        'counter_bias': bias,
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            logger.warning(f"Funding rate error: {e}")
        return {'funding_rate': 0, 'sentiment': 'NEUTRAL', 'counter_bias': 'NEUTRAL'}
    
    def get_full_whale_analysis(self, symbol: str = 'BTCUSDT') -> Dict:
        """Tam whale analizi."""
        oi = self.get_open_interest(symbol)
        ls = self.get_long_short_ratio(symbol)
        top = self.get_top_trader_positions(symbol)
        funding = self.get_funding_rate(symbol)
        
        # Kombine sinyal
        signals = []
        if ls.get('bias') == 'LONG':
            signals.append('LONG')
        elif ls.get('bias') == 'SHORT':
            signals.append('SHORT')
        
        if top.get('top_bias') == 'LONG':
            signals.append('LONG')
        elif top.get('top_bias') == 'SHORT':
            signals.append('SHORT')
        
        if funding.get('counter_bias') != 'NEUTRAL':
            signals.append(funding['counter_bias'])
        
        # Final bias
        long_count = signals.count('LONG')
        short_count = signals.count('SHORT')
        
        if long_count > short_count:
            whale_bias = 'LONG'
            confidence = 40 + (long_count * 10)
        elif short_count > long_count:
            whale_bias = 'SHORT'
            confidence = 40 + (short_count * 10)
        else:
            whale_bias = 'NEUTRAL'
            confidence = 30
        
        return {
            'available': True,
            'symbol': symbol,
            'whale_bias': whale_bias,
            'confidence': min(70, confidence),
            'open_interest': oi.get('open_interest', 0),
            'long_short_ratio': ls.get('long_short_ratio', 1),
            'long_pct': ls.get('long_pct', 50),
            'short_pct': ls.get('short_pct', 50),
            'top_trader_ratio': top.get('top_trader_ratio', 1),
            'top_bias': top.get('top_bias', 'NEUTRAL'),
            'funding_rate_pct': funding.get('funding_rate_pct', 0),
            'funding_sentiment': funding.get('sentiment', 'NEUTRAL'),
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_signal_for_orchestrator(self, symbol: str = 'BTCUSDT') -> Dict:
        """SignalOrchestrator için sinyal."""
        analysis = self.get_full_whale_analysis(symbol)
        
        return {
            'direction': analysis['whale_bias'],
            'confidence': analysis['confidence'],
            'reason': f"L/S: {analysis['long_short_ratio']:.2f}, Funding: {analysis['funding_rate_pct']:.3f}%"
        }


# Convenience functions
def get_whale_signal(symbol: str = 'BTCUSDT') -> Dict:
    """Hızlı whale sinyali."""
    intel = WhaleIntelligence()
    return intel.get_signal_for_orchestrator(symbol)


def get_whale_analysis(symbol: str = 'BTCUSDT') -> Dict:
    """Tam whale analizi."""
    intel = WhaleIntelligence()
    return intel.get_full_whale_analysis(symbol)
