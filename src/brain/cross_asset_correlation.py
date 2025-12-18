# -*- coding: utf-8 -*-
"""
DEMIR AI - Cross-Asset Correlation Analyzer
Varlıklar arası korelasyon - BTC.D, ETH/BTC divergence.

PHASE 88: Cross-Asset Correlation
- BTC Dominance takibi
- ETH/BTC oranı
- Altcoin season tespiti
"""
import logging
import requests
from datetime import datetime
from typing import Dict

logger = logging.getLogger("CROSS_ASSET")


class CrossAssetCorrelation:
    """
    Varlıklar Arası Korelasyon Analizi
    
    BTC.D, ETH/BTC gibi oranları izler ve divergence tespit eder.
    """
    
    BINANCE = "https://api.binance.com/api/v3"
    COINGECKO = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        logger.info("✅ Cross-Asset Correlation initialized")
    
    def analyze_correlations(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Çapraz varlık korelasyonu analiz et.
        
        Returns:
            {
                'btc_dominance': 58.5,
                'btc_dom_trend': 'INCREASING',
                'eth_btc_ratio': 0.035,
                'eth_btc_trend': 'DECREASING',
                'altcoin_season': False,
                'direction': 'LONG',
                'confidence': 60
            }
        """
        try:
            # BTC Dominance (CoinGecko)
            btc_dom = self._get_btc_dominance()
            btc_dom_trend = self._get_dominance_trend()
            
            # ETH/BTC ratio
            eth_btc = self._get_eth_btc_ratio()
            
            # BTC ve ETH fiyat değişimi
            btc_change = self._get_24h_change('BTCUSDT')
            eth_change = self._get_24h_change('ETHUSDT')
            
            # Altcoin season: ETH outperforms BTC + BTC.D düşüyor
            altcoin_season = eth_change > btc_change and btc_dom_trend == 'DECREASING'
            
            # Yön belirleme
            if btc_dom_trend == 'INCREASING' and btc_change > 0:
                direction = 'LONG'  # BTC güçleniyor
                confidence = 60
            elif btc_dom_trend == 'DECREASING' and altcoin_season:
                direction = 'LONG'  # Altcoin rallisi, BTC de genellikle yukarı
                confidence = 55
            elif btc_dom_trend == 'INCREASING' and btc_change < 0:
                direction = 'SHORT'  # Risk-off, BTC'ye kaçış ama düşüyor
                confidence = 55
            else:
                direction = 'NEUTRAL'
                confidence = 45
            
            # ETH/BTC divergence
            if eth_change > btc_change + 2:  # ETH %2+ outperform
                eth_btc_trend = 'ETH_OUTPERFORMING'
            elif btc_change > eth_change + 2:  # BTC %2+ outperform
                eth_btc_trend = 'BTC_OUTPERFORMING'
            else:
                eth_btc_trend = 'ALIGNED'
            
            return {
                'btc_dominance': btc_dom,
                'btc_dom_trend': btc_dom_trend,
                'eth_btc_ratio': eth_btc,
                'eth_btc_trend': eth_btc_trend,
                'btc_24h_change': btc_change,
                'eth_24h_change': eth_change,
                'altcoin_season': altcoin_season,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Cross-asset correlation failed: {e}")
            return self._empty_result()
    
    def _get_btc_dominance(self) -> float:
        """BTC dominance al."""
        try:
            resp = requests.get(
                f"{self.COINGECKO}/global",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get('data', {}).get('market_cap_percentage', {}).get('btc', 50)
        except:
            pass
        return 50.0
    
    def _get_dominance_trend(self) -> str:
        """BTC dominance trendi (basit tahmin)."""
        # Gerçek trend için geçmiş veri gerekir, şimdilik 24h change kullan
        try:
            resp = requests.get(
                f"{self.BINANCE}/ticker/24hr",
                params={'symbol': 'BTCUSDT'},
                timeout=5
            )
            if resp.status_code == 200:
                change = float(resp.json().get('priceChangePercent', 0))
                if change > 1:
                    return 'INCREASING'
                elif change < -1:
                    return 'DECREASING'
        except:
            pass
        return 'STABLE'
    
    def _get_eth_btc_ratio(self) -> float:
        """ETH/BTC oranı."""
        try:
            resp = requests.get(
                f"{self.BINANCE}/ticker/price",
                params={'symbol': 'ETHBTC'},
                timeout=5
            )
            if resp.status_code == 200:
                return float(resp.json().get('price', 0.035))
        except:
            pass
        return 0.035
    
    def _get_24h_change(self, symbol: str) -> float:
        """24 saatlik değişim."""
        try:
            resp = requests.get(
                f"{self.BINANCE}/ticker/24hr",
                params={'symbol': symbol},
                timeout=5
            )
            if resp.status_code == 200:
                return float(resp.json().get('priceChangePercent', 0))
        except:
            pass
        return 0.0
    
    def _empty_result(self) -> Dict:
        return {
            'btc_dominance': 50,
            'btc_dom_trend': 'UNKNOWN',
            'eth_btc_ratio': 0.035,
            'eth_btc_trend': 'UNKNOWN',
            'altcoin_season': False,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def analyze_correlations() -> Dict:
    """Quick correlation analysis."""
    analyzer = CrossAssetCorrelation()
    return analyzer.analyze_correlations()
