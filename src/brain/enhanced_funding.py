# -*- coding: utf-8 -*-
"""
DEMIR AI - Enhanced Funding Rate Tracker
Multi-exchange funding rate comparison

PHASE 45: High-Value Scraper
- Track funding rates across Binance, Bybit, OKX
- Detect divergences (manipulation signals)
- Historical pattern matching
- Extreme funding = contrarian signal
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("ENHANCED_FUNDING")


@dataclass
class FundingRate:
    """Funding rate data"""
    exchange: str
    symbol: str
    rate: float  # As percentage (e0.01 = 0.01%)
    next_funding_time: datetime
    timestamp: datetime


class EnhancedFundingTracker:
    """
    Enhanced Funding Rate Tracker
    
    Tracks funding rates across multiple exchanges:
    - Binance Futures
    - Bybit
    - OKX
    
    Detects:
    - Extreme funding (>0.1% = overheated)
    - Cross-exchange divergence (manipulation)
    - Funding rate reversals (trend change)
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.last_fetch = {}
    
    def get_multi_exchange_funding(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Get funding rates from all exchanges.
        
        Returns:
            {
                'binance': FundingRate,
                'bybit': FundingRate,
                'okx': FundingRate,
                'average': float,
                'divergence': float,
                'signal': str,
                'summary': str
            }
        """
        cache_key = f'funding_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        funding_data = {}
        
   # Fetch from each exchange
        funding_data['binance'] = self._fetch_binance_funding(symbol)
        funding_data['bybit'] = self._fetch_bybit_funding(symbol)
        funding_data['okx'] = self._fetch_okx_funding(symbol)
        
        # Calculate metrics
        rates = [f.rate for f in funding_data.values() if f is not None]
        
        if not rates:
            return self._empty_result(symbol)
        
        average_rate = sum(rates) / len(rates)
        max_rate = max(rates)
        min_rate = min(rates)
        divergence = max_rate - min_rate
        
        # Determine signal
        signal, summary = self._analyze_funding(average_rate, divergence, rates)
        
        result = {
            **funding_data,
            'average': average_rate,
            'max': max_rate,
            'min': min_rate,
            'divergence': divergence,
            'signal': signal,
            'summary': summary,
            'timestamp': datetime.now()
        }
        
        self._set_cache(cache_key, result)
        logger.info(f"Multi-exchange funding: {average_rate:.4f}% (divergence: {divergence:.4f}%)")
        
        return result
    
    def format_for_telegram(self, symbol: str = 'BTCUSDT') -> str:
        """Telegram formatı"""
        data = self.get_multi_exchange_funding(symbol)
        
        msg = "💰 *Enhanced Funding Rate*\\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\\n\\n"
        
        # Exchange rates
        for exchange in ['binance', 'bybit', 'okx']:
            fr = data.get(exchange)
            if fr:
                rate_pct = fr.rate * 100
                emoji = "🔴" if fr.rate > 0.1 else "🟡" if fr.rate > 0.05 else "🟢"
                msg += f"{exchange.capitalize()}: {emoji} {rate_pct:.4f}%\\n"
        
        msg += f"\\n**Average:** {data['average']*100:.4f}%\\n"
        msg += f"**Divergence:** {data['divergence']*100:.4f}%\\n\\n"
        
        # Signal
        signal = data['signal']
        if signal == 'EXTREME_LONG':
            msg += "🔴 **EXTREME LONG BIAS**\\n"
        elif signal == 'EXTREME_SHORT':
            msg += "🟢 **EXTREME SHORT BIAS**\\n"
        elif signal == 'DIVERGENCE':
            msg += "⚠️ **CROSS-EXCHANGE DIVERGENCE**\\n"
        else:
            msg += "↔️ **NORMAL FUNDING**\\n"
        
        msg += f"\\n💡 _{data['summary']}_\\n"
        msg += f"\\n⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        
        return msg
    
    # =========================================
    # PRIVATE EXCHANGE FETCHERS
    # =========================================
    
    def _fetch_binance_funding(self, symbol: str) -> Optional[FundingRate]:
        """Fetch Binance funding rate"""
        try:
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                return FundingRate(
                    exchange='binance',
                    symbol=symbol,
                    rate=float(data.get('lastFundingRate', 0)),
                    next_funding_time=datetime.fromtimestamp(data.get('nextFundingTime', 0) / 1000),
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Binance funding failed: {e}")
            return None
    
    def _fetch_bybit_funding(self, symbol: str) -> Optional[FundingRate]:
        """Fetch Bybit funding rate"""
        try:
            # Bybit uses different symbol format
            bybit_symbol = symbol.replace('USDT', '')
            
            url = "https://api.bybit.com/v2/public/tickers"
            params = {'symbol': f"{bybit_symbol}USDT"}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('result'):
                    result = data['result'][0] if isinstance(data['result'], list) else data['result']
                    
                    return FundingRate(
                        exchange='bybit',
                        symbol=symbol,
                        rate=float(result.get('funding_rate', 0)),
                        next_funding_time=datetime.now() + timedelta(hours=8),  # Bybit funds every 8h
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"Bybit funding failed: {e}")
            return None
    
    def _fetch_okx_funding(self, symbol: str) -> Optional[FundingRate]:
        """Fetch OKX funding rate"""
        try:
            # OKX uses swap format
            okx_symbol = f"{symbol[:-4]}-{symbol[-4:]}-SWAP"  # BTC-USDT-SWAP
            
            url = "https://www.okx.com/api/v5/public/funding-rate"
            params = {'instId': okx_symbol}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('data'):
                    result = data['data'][0]
                    
                    return FundingRate(
                        exchange='okx',
                        symbol=symbol,
                        rate=float(result.get('fundingRate', 0)),
                        next_funding_time=datetime.fromtimestamp(int(result.get('nextFundingTime', 0)) / 1000),
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"OKX funding failed: {e}")
            return None
    
    # =========================================
    # ANALYSIS
    # =========================================
    
    def _analyze_funding(self, avg_rate: float, divergence: float, all_rates: List[float]) -> tuple:
        """
        Analyze funding rates and generate signal.
        
        Returns: (signal, summary)
        """
        # Extreme thresholds
        EXTREME_LONG_THRESHOLD = 0.0010  # 0.10%
        EXTREME_SHORT_THRESHOLD = -0.0005  # -0.05%
        DIVERGENCE_THRESHOLD = 0.0005  # 0.05% difference
        
        # Check for extreme funding
        if avg_rate >= EXTREME_LONG_THRESHOLD:
            return ('EXTREME_LONG', 
                    f"⚠️ Extreme long bias ({avg_rate*100:.3f}%). Market overheated - CONTRARIAN SHORT signal!")
        
        elif avg_rate <= EXTREME_SHORT_THRESHOLD:
            return ('EXTREME_SHORT',
                    f"✅ Extreme short bias ({avg_rate*100:.3f}%). Market oversold - CONTRARIAN LONG signal!")
        
        # Check for divergence
        elif divergence >= DIVERGENCE_THRESHOLD:
            return ('DIVERGENCE',
                    f"🔍 Cross-exchange divergence ({divergence*100:.3f}%). Possible manipulation or arbitrage.")
        
        # Normal funding
        else:
            if avg_rate > 0:
                return ('NORMAL_LONG',
                        f"↔️ Moderate long bias ({avg_rate*100:.3f}%). Normal bullish sentiment.")
            elif avg_rate < 0:
                return ('NORMAL_SHORT',
                        f"↔️ Moderate short bias ({avg_rate*100:.3f}%). Normal bearish sentiment.")
            else:
                return ('NEUTRAL',
                        "↔️ Balanced funding. No directional bias.")
    
    def _empty_result(self, symbol: str) -> Dict:
        """Return empty result"""
        return {
            'binance': None,
            'bybit': None,
            'okx': None,
            'average': 0,
            'max': 0,
            'min': 0,
            'divergence': 0,
            'signal': 'NEUTRAL',
            'summary': 'Funding data unavailable',
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
