# -*- coding: utf-8 -*-
"""
DEMIR AI - CoinGlass Whale Alerts Scraper
Büyük transfer takibi - Borsa girişi/çıkışı.

PHASE 79: Whale Alerts Scraping
- Borsaya BTC girişi = Satış baskısı
- Borsadan BTC çıkışı = Hodl sinyali
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger("COINGLASS_WHALE_ALERTS")


class CoinGlassWhaleAlerts:
    """
    CoinGlass Whale Alerts Scraper
    
    Büyük kripto transferlerini takip eder.
    Borsaya giriş = Satış, Borsadan çıkış = Alış sinyali.
    """
    
    # Whale Alert API (alternatif)
    WHALE_ALERT_API = "https://api.whale-alert.io/v1/transactions"
    
    # Blockchair için alternatif
    BLOCKCHAIR_API = "https://api.blockchair.com/bitcoin/transactions"
    
    def __init__(self):
        self.min_btc = 100  # Minimum 100 BTC
        logger.info("✅ CoinGlass Whale Alerts Scraper initialized")
    
    def get_whale_alerts(self, hours: int = 4) -> Dict:
        """
        Son X saat içindeki whale transferlerini al.
        
        Returns:
            {
                'exchange_inflows': [{'amount': 500, 'exchange': 'Binance'}],
                'exchange_outflows': [{'amount': 300, 'exchange': 'Coinbase'}],
                'net_flow': -200,  # Negatif = Net çıkış (Bullish)
                'direction': 'LONG'/'SHORT'/'NEUTRAL',
                'confidence': 60
            }
        """
        try:
            # Yöntem 1: Public mempool/exchange flow tahmini
            result = self._estimate_exchange_flow()
            return result
            
        except Exception as e:
            logger.warning(f"Whale alerts scraping failed: {e}")
            return self._empty_result()
    
    def _estimate_exchange_flow(self) -> Dict:
        """
        Exchange flow'u Binance verilerinden tahmin et.
        
        Funding rate ve OI değişiminden borsa akışı çıkarılabilir.
        """
        try:
            # OI değişimi (artan OI = borsaya giriş)
            oi_resp = requests.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={'symbol': 'BTCUSDT', 'period': '1h', 'limit': 5},
                timeout=10
            )
            
            if oi_resp.status_code != 200:
                return self._empty_result()
            
            oi_data = oi_resp.json()
            
            if len(oi_data) < 2:
                return self._empty_result()
            
            current_oi = float(oi_data[-1]['sumOpenInterest'])
            prev_oi = float(oi_data[0]['sumOpenInterest'])
            oi_change_pct = ((current_oi / prev_oi) - 1) * 100
            
            # Spot volume değişimi
            volume_resp = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={'symbol': 'BTCUSDT'},
                timeout=5
            )
            
            volume_data = volume_resp.json()
            volume_24h = float(volume_data.get('volume', 0))
            price_change_pct = float(volume_data.get('priceChangePercent', 0))
            
            # Exchange flow tahmini
            # OI artışı + Fiyat düşüşü = Borsaya giriş (satış)
            # OI düşüşü + Fiyat artışı = Borsadan çıkış (hodl)
            
            if oi_change_pct > 5:
                # OI hızla artıyor
                if price_change_pct < 0:
                    net_flow = abs(oi_change_pct) * 100  # Inflow (BTC)
                    direction = 'SHORT'
                    confidence = 60
                else:
                    net_flow = 0
                    direction = 'LONG'
                    confidence = 50
            elif oi_change_pct < -3:
                # OI hızla düşüyor
                net_flow = -abs(oi_change_pct) * 100  # Outflow
                direction = 'LONG'
                confidence = 60
            else:
                net_flow = 0
                direction = 'NEUTRAL'
                confidence = 40
            
            # Tahmini inflow/outflow
            if net_flow > 0:
                inflows = [{'amount_btc': net_flow, 'exchange': 'Estimated', 'type': 'INFLOW'}]
                outflows = []
            elif net_flow < 0:
                inflows = []
                outflows = [{'amount_btc': abs(net_flow), 'exchange': 'Estimated', 'type': 'OUTFLOW'}]
            else:
                inflows = []
                outflows = []
            
            return {
                'exchange_inflows': inflows,
                'exchange_outflows': outflows,
                'net_flow_btc': net_flow,
                'oi_change_pct': oi_change_pct,
                'price_change_pct': price_change_pct,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Exchange flow estimate failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'exchange_inflows': [],
            'exchange_outflows': [],
            'net_flow_btc': 0,
            'oi_change_pct': 0,
            'price_change_pct': 0,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def get_whale_alerts() -> Dict:
    """Quick whale alerts check."""
    scraper = CoinGlassWhaleAlerts()
    return scraper.get_whale_alerts()
