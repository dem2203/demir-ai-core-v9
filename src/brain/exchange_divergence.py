# -*- coding: utf-8 -*-
"""
DEMIR AI - Cross-Exchange Price Divergence
Borsa arası fiyat farkı tespit - Arbitraj ve hareket sinyali.

PHASE 75: Cross-Exchange Divergence
- Coinbase vs Binance fiyat karşılaştırması
- Coinbase Premium = Kurumsal talep
- >0.3% fark = Hareket habercisi
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger("EXCHANGE_DIVERGENCE")


class ExchangeDivergenceDetector:
    """
    Cross-Exchange Price Divergence Detector
    
    Borsa arası fiyat farkı tespit eder.
    
    Nasıl Çalışır:
    1. Binance ve Coinbase fiyatlarını al
    2. Fark hesapla (Coinbase Premium)
    3. Premium > 0 = Kurumsal alım (LONG)
    4. Premium < 0 = Kurumsal satış (SHORT)
    """
    
    DIVERGENCE_THRESHOLD = 0.3  # %0.3 fark = anlamlı
    STRONG_DIVERGENCE = 0.5  # %0.5 fark = güçlü sinyal
    
    def __init__(self):
        logger.info("✅ Exchange Divergence Detector initialized")
    
    def detect_divergence(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Borsa arası fiyat farkı tespit et.
        
        Returns:
            {
                'binance_price': 87150.0,
                'coinbase_price': 87300.0,
                'premium_pct': 0.17,
                'divergence_type': 'COINBASE_PREMIUM'/'BINANCE_PREMIUM',
                'direction': 'LONG'/'SHORT',
                'confidence': 55
            }
        """
        try:
            # Binance fiyatı
            binance = requests.get(
                f"https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            
            if binance.status_code != 200:
                return self._empty_result()
            
            binance_price = float(binance.json()['price'])
            
            # Coinbase fiyatı
            cb_symbol = symbol.replace('USDT', '-USD')
            coinbase = requests.get(
                f"https://api.coinbase.com/v2/prices/{cb_symbol}/spot",
                timeout=5
            )
            
            if coinbase.status_code != 200:
                # Coinbase API failed, try alternative
                return self._try_alternative_exchanges(symbol, binance_price)
            
            coinbase_price = float(coinbase.json()['data']['amount'])
            
            # Premium hesapla
            premium_pct = ((coinbase_price / binance_price) - 1) * 100
            
            # Yön ve güven
            if abs(premium_pct) >= self.STRONG_DIVERGENCE:
                if premium_pct > 0:
                    direction = 'LONG'
                    divergence_type = 'COINBASE_PREMIUM'
                else:
                    direction = 'SHORT'
                    divergence_type = 'BINANCE_PREMIUM'
                confidence = min(75, 55 + abs(premium_pct) * 20)
            elif abs(premium_pct) >= self.DIVERGENCE_THRESHOLD:
                if premium_pct > 0:
                    direction = 'LONG'
                    divergence_type = 'COINBASE_PREMIUM'
                else:
                    direction = 'SHORT'
                    divergence_type = 'BINANCE_PREMIUM'
                confidence = min(60, 45 + abs(premium_pct) * 15)
            else:
                direction = 'NEUTRAL'
                divergence_type = 'ALIGNED'
                confidence = 40
            
            return {
                'binance_price': binance_price,
                'coinbase_price': coinbase_price,
                'premium_pct': round(premium_pct, 3),
                'divergence_type': divergence_type,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Exchange divergence detection failed: {e}")
            return self._empty_result()
    
    def _try_alternative_exchanges(self, symbol: str, binance_price: float) -> Dict:
        """Coinbase yerine Kraken dene."""
        try:
            # Kraken fiyatı
            kraken_symbol = 'XXBTZUSD' if 'BTC' in symbol else 'XETHZUSD'
            kraken = requests.get(
                f"https://api.kraken.com/0/public/Ticker",
                params={'pair': kraken_symbol},
                timeout=5
            )
            
            if kraken.status_code == 200:
                data = kraken.json()
                if 'result' in data:
                    key = list(data['result'].keys())[0]
                    kraken_price = float(data['result'][key]['c'][0])
                    
                    premium_pct = ((kraken_price / binance_price) - 1) * 100
                    
                    if abs(premium_pct) >= self.DIVERGENCE_THRESHOLD:
                        direction = 'LONG' if premium_pct > 0 else 'SHORT'
                        divergence_type = 'KRAKEN_PREMIUM' if premium_pct > 0 else 'BINANCE_PREMIUM'
                        confidence = min(60, 45 + abs(premium_pct) * 15)
                    else:
                        direction = 'NEUTRAL'
                        divergence_type = 'ALIGNED'
                        confidence = 40
                    
                    return {
                        'binance_price': binance_price,
                        'coinbase_price': kraken_price,  # Actually Kraken
                        'premium_pct': round(premium_pct, 3),
                        'divergence_type': divergence_type,
                        'direction': direction,
                        'confidence': confidence,
                        'available': True
                    }
        except Exception as e:
            logger.warning(f"Alternative exchange check failed: {e}")
        
        return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'binance_price': 0,
            'coinbase_price': 0,
            'premium_pct': 0,
            'divergence_type': 'UNKNOWN',
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def detect_exchange_divergence(symbol: str = 'BTCUSDT') -> Dict:
    """Quick exchange divergence detection."""
    detector = ExchangeDivergenceDetector()
    return detector.detect_divergence(symbol)
