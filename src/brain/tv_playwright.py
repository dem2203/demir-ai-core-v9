# -*- coding: utf-8 -*-
"""
DEMIR AI - TradingView Playwright Scraper
Playwright ile gerçek zamanlı TradingView teknik göstergeleri.

PHASE 52: Advanced Browser Scraping
- RSI, MACD, Bollinger Bands
- Volume Profile
- Support/Resistance
- TradingView widget verileri
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import re

logger = logging.getLogger("TV_PLAYWRIGHT")


class TradingViewPlaywright:
    """
    TradingView Playwright Scraper
    
    Gerçek zamanlı teknik göstergeler:
    - RSI (14)
    - MACD
    - Bollinger Bands
    - Volume analizi
    - Trend yönü
    
    Kaynak: TradingView Mini Widget veya Scanner API
    """
    
    # TradingView Scanner API (public, no auth required)
    SCANNER_API = "https://scanner.tradingview.com/crypto/scan"
    
    # Symbol mappings
    SYMBOLS = {
        'BTCUSDT': 'BINANCE:BTCUSDT',
        'ETHUSDT': 'BINANCE:ETHUSDT',
        'SOLUSDT': 'BINANCE:SOLUSDT',
        'LTCUSDT': 'BINANCE:LTCUSDT'
    }
    
    # Technical columns we want
    COLUMNS = [
        "close",
        "change",
        "volume",
        "RSI",
        "MACD.macd",
        "MACD.signal",
        "BB.upper",
        "BB.lower",
        "Recommend.All",  # Technical rating: -1 to 1
        "Recommend.MA",
        "Recommend.Other",
        "ADX",
        "ATR",
        "Volatility.D"
    ]
    
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 60  # 60 saniye cache
        self.last_fetch: Dict[str, datetime] = {}
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Cache geçerli mi kontrol et."""
        if symbol not in self.last_fetch:
            return False
        return (datetime.now() - self.last_fetch[symbol]).seconds < self.cache_ttl
    
    def get_indicators(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        TradingView'dan teknik göstergeleri çek.
        
        Returns:
            {
                'rsi': 65.3,
                'macd': {'macd': 123.4, 'signal': 100.2, 'histogram': 23.2},
                'bollinger': {'upper': 88000, 'lower': 84000, 'position': 0.6},
                'recommendation': 'BUY' / 'SELL' / 'NEUTRAL',
                'adx': 25.3,
                'volatility': 2.1
            }
        """
        # Check cache
        if self._is_cache_valid(symbol) and symbol in self.cache:
            return self.cache[symbol]
        
        try:
            import requests
            
            tv_symbol = self.SYMBOLS.get(symbol, f"BINANCE:{symbol}")
            
            # TradingView Scanner API request
            payload = {
                "symbols": {"tickers": [tv_symbol], "query": {"types": []}},
                "columns": self.COLUMNS
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.post(
                self.SCANNER_API,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"TradingView API error: {response.status_code}")
                return self._empty_response("API error")
            
            data = response.json()
            
            if not data.get('data') or len(data['data']) == 0:
                return self._empty_response("No data returned")
            
            # Parse response
            values = data['data'][0]['d']
            
            # Map columns to values
            result = dict(zip(self.COLUMNS, values))
            
            # Calculate derived values
            price = result.get('close', 0)
            bb_upper = result.get('BB.upper', 0)
            bb_lower = result.get('BB.lower', 0)
            
            # Bollinger position (0-1, where 0.5 is middle)
            bb_range = bb_upper - bb_lower if bb_upper and bb_lower else 1
            bb_position = (price - bb_lower) / bb_range if bb_range > 0 else 0.5
            
            # Technical recommendation interpretation
            rec_all = result.get('Recommend.All', 0) or 0
            rsi = result.get('RSI', 50) or 50
            adx = result.get('ADX', 20) or 20
            
            if rec_all > 0.3:
                recommendation = 'STRONG_BUY'
                direction = 'LONG'
            elif rec_all > 0.1:
                recommendation = 'BUY'
                direction = 'LONG'
            elif rec_all < -0.3:
                recommendation = 'STRONG_SELL'
                direction = 'SHORT'
            elif rec_all < -0.1:
                recommendation = 'SELL'
                direction = 'SHORT'
            else:
                recommendation = 'NEUTRAL'
                direction = 'NEUTRAL'
            
            # BOOSTED Confidence calculation
            # Base: recommendation strength (0-50)
            base_confidence = abs(rec_all) * 100
            
            # RSI boost: extreme values add confidence
            rsi_boost = 0
            if rsi < 30 or rsi > 70:
                rsi_boost = 15  # Oversold/overbought zones
            elif rsi < 35 or rsi > 65:
                rsi_boost = 10
            
            # ADX boost: strong trend adds confidence
            adx_boost = 0
            if adx > 25:
                adx_boost = 15  # Strong trend
            elif adx > 20:
                adx_boost = 10
            
            # Bollinger boost: extreme positions
            bb_boost = 0
            if bb_position < 0.2 or bb_position > 0.8:
                bb_boost = 10
            
            # Total confidence
            confidence = base_confidence + rsi_boost + adx_boost + bb_boost
            confidence = max(40, min(90, confidence))  # Clamp to 40-90
            
            parsed = {
                'available': True,
                'symbol': symbol,
                'price': price,
                'change_pct': result.get('change', 0),
                'volume': result.get('volume', 0),
                'rsi': result.get('RSI', 50),
                'macd': {
                    'macd': result.get('MACD.macd', 0),
                    'signal': result.get('MACD.signal', 0),
                    'histogram': (result.get('MACD.macd', 0) or 0) - (result.get('MACD.signal', 0) or 0)
                },
                'bollinger': {
                    'upper': bb_upper,
                    'lower': bb_lower,
                    'position': round(bb_position, 2)
                },
                'recommendation': recommendation,
                'recommendation_value': rec_all,
                'direction': direction,
                'confidence': round(confidence, 1),
                'adx': result.get('ADX', 0),
                'atr': result.get('ATR', 0),
                'volatility': result.get('Volatility.D', 0),
                'source': 'tradingview_scanner',
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[symbol] = parsed
            self.last_fetch[symbol] = datetime.now()
            
            logger.info(f"✅ TradingView: {symbol} RSI={parsed['rsi']:.1f} Rec={recommendation}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"TradingView fetch error: {e}")
            return self._empty_response(str(e))
    
    def _empty_response(self, reason: str) -> Dict:
        """Empty response when data unavailable."""
        return {
            'available': False,
            'reason': reason,
            'rsi': 0,
            'macd': {'macd': 0, 'signal': 0, 'histogram': 0},
            'bollinger': {'upper': 0, 'lower': 0, 'position': 0},
            'recommendation': 'UNAVAILABLE',
            'direction': 'NEUTRAL',
            'confidence': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_signal_for_orchestrator(self, symbol: str = 'BTCUSDT') -> Dict:
        """SignalOrchestrator için formatted sinyal."""
        data = self.get_indicators(symbol)
        
        if not data.get('available'):
            return {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'reason': data.get('reason', 'Data unavailable')
            }
        
        return {
            'direction': data['direction'],
            'confidence': data['confidence'],
            'reason': f"RSI: {data['rsi']:.0f}, Rec: {data['recommendation']}"
        }
    
    def get_multi_symbol_analysis(self, symbols: List[str] = None) -> Dict:
        """Çoklu sembol analizi."""
        if symbols is None:
            symbols = list(self.SYMBOLS.keys())
        
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_indicators(symbol)
        
        return results


# Convenience functions
def get_tradingview_signal(symbol: str = 'BTCUSDT') -> Dict:
    """Hızlı TradingView sinyali."""
    scraper = TradingViewPlaywright()
    return scraper.get_signal_for_orchestrator(symbol)


def get_tradingview_indicators(symbol: str = 'BTCUSDT') -> Dict:
    """Tam gösterge verisi."""
    scraper = TradingViewPlaywright()
    return scraper.get_indicators(symbol)
