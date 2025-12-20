"""
DEMIR AI - ON-CHAIN INTELLIGENCE MODULE
Glassnode alternatifi: Binance + Public APIs

Whale hareketleri, Exchange flow, MVRV simülasyonu
"""

import logging
import aiohttp
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("ON_CHAIN_INTEL")

class OnChainIntelligence:
    """
    ON-CHAIN VERİ ZEKASI
    
    Ücretsiz Kaynaklar:
    1. Binance Large Trades (Whale Detector)
    2. Exchange Flow (Deposit/Withdraw patterns)
    3. Whale Alert API (Ücretsiz tier)
    4. Blockchain.info (BTC specific)
    """
    
    # Whale eşikleri (USD)
    WHALE_THRESHOLD = {
        'BTC/USDT': 1_000_000,
        'ETH/USDT': 500_000,
        'default': 250_000
    }
    
    def __init__(self):
        self.session = None
        self.whale_cache = []
        self.last_update = None
        
    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def detect_whale_trades(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Binance Whale Tracker Entegrasyonu (Zero-Mock).
        Gerçek zamanlı WebSocket verisi kullanılır.
        """
        try:
            from src.brain.whale_tracker import get_whale_tracker
            tracker = get_whale_tracker()
            
            # Eğer tracker çalışmıyorsa başlat (Lazy Start)
            if not tracker.running:
                await tracker.start()
                # İlk veri için kısa bekleme
                await asyncio.sleep(2)
            
            # WebSocket'ten anlık özet al
            whale_summary = tracker.get_whale_summary()
            
            if not whale_summary['available']:
                return {'whale_count': 0, 'whale_volume': 0, 'direction': 'neutral', 'reason': 'Tracker warming up'}
            
            # Veriyi OnChainIntel formatına çevir
            # WhaleTracker net_flow_usd ve imbalance veriyor
            
            net_flow = whale_summary.get('net_flow_usd', 0)
            
            direction = "NEUTRAL"
            if net_flow > 1_000_000: direction = "ACCUMULATION"
            elif net_flow < -1_000_000: direction = "DISTRIBUTION"
            
            result = {
                'whale_buys': whale_summary.get('whale_trade_count', 0), # Simplified count
                'whale_sells': 0, # Tracker aggregates net flow mostly
                'whale_buy_volume': max(0, net_flow) if net_flow > 0 else 0,
                'whale_sell_volume': abs(net_flow) if net_flow < 0 else 0,
                'total_whale_volume': abs(net_flow),
                'direction': direction,
                'net_flow_usd': net_flow,
                'imbalance_ratio': whale_summary.get('imbalance_ratio', 1.0),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🐋 Whale Activity (WS): {direction} | Net Flow: ${net_flow:,.0f}")
            return result
            
        except Exception as e:
            logger.error(f"Whale detection error: {e}")
            return {'whale_count': 0, 'direction': 'neutral', 'error': str(e)}
    
    async def get_exchange_netflow(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Exchange'e giren/çıkan coin akışı.
        Binance order book depth'ten türetilir.
        
        - Yüksek bid depth = Alıcılar hazır (bullish)
        - Yüksek ask depth = Satıcılar hazır (bearish)
        """
        try:
            session = await self._get_session()
            
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=500"
            
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {'netflow': 0, 'signal': 'neutral'}
                
                data = await resp.json()
            
            # Bid (alım) derinliği
            bid_volume = sum(float(bid[1]) for bid in data['bids'][:100])
            bid_value = sum(float(bid[0]) * float(bid[1]) for bid in data['bids'][:100])
            
            # Ask (satım) derinliği
            ask_volume = sum(float(ask[1]) for ask in data['asks'][:100])
            ask_value = sum(float(ask[0]) * float(ask[1]) for ask in data['asks'][:100])
            
            # Net flow (pozitif = alım baskısı)
            netflow_ratio = bid_value / max(ask_value, 1)
            
            if netflow_ratio > 1.2:
                signal = "INFLOW"  # Paralar borsaya geliyor (potansiyel satış)
            elif netflow_ratio < 0.8:
                signal = "OUTFLOW"  # Paralar borsadan çıkıyor (hodl)
            else:
                signal = "BALANCED"
            
            result = {
                'bid_depth': bid_value,
                'ask_depth': ask_value,
                'netflow_ratio': netflow_ratio,
                'signal': signal,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            }
            
            logger.info(f"💰 Exchange Flow: {signal} | Bid/Ask Ratio: {netflow_ratio:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Exchange flow error: {e}")
            return {'netflow_ratio': 1.0, 'signal': 'neutral'}
    
    async def calculate_mvrv_proxy(self, symbol: str = "BTCUSDT") -> Dict:
        """
        MVRV Proxy: Kısa vadeli holder P/L durumu.
        Gerçek MVRV yerine 30 günlük ortalamaya göre hesaplama.
        """
        try:
            session = await self._get_session()
            
            # Son 30 günlük veri
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=30"
            
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {'mvrv_proxy': 1.0, 'zone': 'neutral'}
                
                klines = await resp.json()
            
            # Ortalama maliyet tahmini (30 günlük VWAP benzeri)
            total_volume = 0
            total_value = 0
            
            for k in klines:
                typical_price = (float(k[2]) + float(k[3]) + float(k[4])) / 3  # HLC avg
                volume = float(k[5])
                total_value += typical_price * volume
                total_volume += volume
            
            avg_cost = total_value / max(total_volume, 1)
            current_price = float(klines[-1][4])  # Son kapanış
            
            # MVRV Proxy = Current Price / Average Cost
            mvrv = current_price / avg_cost
            
            # Zone belirleme
            if mvrv > 1.5:
                zone = "OVERVALUED"  # Çok kârlı, düşüş riski
                signal = "CAUTION"
            elif mvrv > 1.1:
                zone = "PROFITABLE"  # Sağlıklı kâr
                signal = "NEUTRAL"
            elif mvrv > 0.9:
                zone = "NEUTRAL"
                signal = "NEUTRAL"
            elif mvrv > 0.7:
                zone = "UNDERVALUED"  # Fırsat bölgesi
                signal = "OPPORTUNITY"
            else:
                zone = "CAPITULATION"  # Panik satışı, dip sinyali
                signal = "STRONG_BUY"
            
            result = {
                'mvrv_proxy': mvrv,
                'avg_cost_30d': avg_cost,
                'current_price': current_price,
                'zone': zone,
                'signal': signal,
                'profit_pct': (mvrv - 1) * 100
            }
            
            logger.info(f"📊 MVRV Proxy: {mvrv:.2f} | Zone: {zone} | Avg Cost: ${avg_cost:,.0f}")
            return result
            
        except Exception as e:
            logger.error(f"MVRV calculation error: {e}")
            return {'mvrv_proxy': 1.0, 'zone': 'neutral'}
    
    async def get_full_onchain_analysis(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Tüm on-chain metriklerini birleştir.
        """
        whale_data, flow_data, mvrv_data = await asyncio.gather(
            self.detect_whale_trades(symbol),
            self.get_exchange_netflow(symbol),
            self.calculate_mvrv_proxy(symbol)
        )
        
        # Composite Score (-100 to +100)
        score = 0
        
        # Whale yönü
        if whale_data.get('direction') == 'ACCUMULATION':
            score += 30
        elif whale_data.get('direction') == 'DISTRIBUTION':
            score -= 30
        
        # Exchange flow
        if flow_data.get('signal') == 'OUTFLOW':
            score += 20  # Borsadan çıkış = bullish
        elif flow_data.get('signal') == 'INFLOW':
            score -= 20
        
        # MVRV
        mvrv = mvrv_data.get('mvrv_proxy', 1.0)
        if mvrv < 0.8:
            score += 30  # Undervalued
        elif mvrv > 1.4:
            score -= 30  # Overvalued
        
        # Final signal
        if score >= 40:
            final_signal = "STRONG_BUY"
        elif score >= 20:
            final_signal = "BUY"
        elif score <= -40:
            final_signal = "STRONG_SELL"
        elif score <= -20:
            final_signal = "SELL"
        else:
            final_signal = "NEUTRAL"
        
        return {
            'whale': whale_data,
            'exchange_flow': flow_data,
            'mvrv': mvrv_data,
            'composite_score': score,
            'signal': final_signal,
            'timestamp': datetime.now().isoformat()
        }
    
    async def close(self):
        if self.session:
            await self.session.close()
