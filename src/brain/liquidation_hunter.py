"""
DEMIR AI - LIQUIDATION HUNTER
Tasfiye haritası ve likidite hedefleri

Fiyatın "çekileceği" seviyeleri tahmin eder.
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("LIQUIDATION_HUNTER")

class LiquidationHunter:
    """
    LİKİDASYON AVCISI
    
    Özellikler:
    1. Liquidation Levels (Long/Short tasfiye seviyeleri)
    2. Open Interest Changes
    3. Funding Rate Extremes
    4. Long/Short Ratio Analysis
    
    Piyasa yapıcılar likiditeyi avlar - biz de onları takip ederiz.
    """
    
    # Leverage dağılımı tahmini (gerçek veriye erişim olmadan)
    LEVERAGE_DISTRIBUTION = [2, 3, 5, 10, 20, 25, 50, 75, 100, 125]
    
    def __init__(self):
        self.session = None
        
    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def calculate_liquidation_levels(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Tasfiye seviyelerini hesapla.
        
        Long Liq = Entry * (1 - 1/leverage)
        Short Liq = Entry * (1 + 1/leverage)
        """
        try:
            session = await self._get_session()
            
            # Güncel fiyat
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            async with session.get(url) as resp:
                data = await resp.json()
                current_price = float(data['price'])
            
            # Son 24 saat high/low (pozisyon giriş tahminleri)
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            async with session.get(url) as resp:
                stats = await resp.json()
                high_24h = float(stats['highPrice'])
                low_24h = float(stats['lowPrice'])
                vwap = float(stats['weightedAvgPrice'])
            
            long_liquidations = []
            short_liquidations = []
            
            # Her leverage için tasfiye seviyeleri
            for lev in self.LEVERAGE_DISTRIBUTION:
                # Long pozisyonlar için tasfiye (fiyat düşerse)
                # Entry point olarak VWAP ve low kullanıyoruz
                for entry in [vwap, low_24h, current_price * 0.98]:
                    liq_price = entry * (1 - 1/lev + 0.005)  # +0.5% maintenance margin
                    if liq_price < current_price:
                        long_liquidations.append({
                            'price': liq_price,
                            'leverage': lev,
                            'distance_pct': ((current_price - liq_price) / current_price) * 100
                        })
                
                # Short pozisyonlar için tasfiye (fiyat yükselirse)
                for entry in [vwap, high_24h, current_price * 1.02]:
                    liq_price = entry * (1 + 1/lev - 0.005)
                    if liq_price > current_price:
                        short_liquidations.append({
                            'price': liq_price,
                            'leverage': lev,
                            'distance_pct': ((liq_price - current_price) / current_price) * 100
                        })
            
            # En yakın seviyeleri bul
            long_liquidations.sort(key=lambda x: x['distance_pct'])
            short_liquidations.sort(key=lambda x: x['distance_pct'])
            
            # Cluster analizi - aynı bölgedeki likidasyonları grupla
            def cluster_levels(levels: List, tolerance: float = 0.5):
                if not levels:
                    return []
                clusters = []
                current_cluster = [levels[0]]
                
                for level in levels[1:]:
                    if abs(level['distance_pct'] - current_cluster[-1]['distance_pct']) < tolerance:
                        current_cluster.append(level)
                    else:
                        avg_price = sum(l['price'] for l in current_cluster) / len(current_cluster)
                        clusters.append({
                            'price': avg_price,
                            'count': len(current_cluster),
                            'intensity': len(current_cluster) * 10,  # 0-100 scale
                            'distance_pct': current_cluster[0]['distance_pct']
                        })
                        current_cluster = [level]
                
                # Son cluster
                if current_cluster:
                    avg_price = sum(l['price'] for l in current_cluster) / len(current_cluster)
                    clusters.append({
                        'price': avg_price,
                        'count': len(current_cluster),
                        'intensity': len(current_cluster) * 10,
                        'distance_pct': current_cluster[0]['distance_pct']
                    })
                
                return clusters[:5]  # En yakın 5 cluster
            
            long_clusters = cluster_levels(long_liquidations)
            short_clusters = cluster_levels(short_liquidations)
            
            # Magnet effect - en güçlü çekim noktası
            all_clusters = long_clusters + short_clusters
            if all_clusters:
                strongest = max(all_clusters, key=lambda x: x['intensity'])
                magnet_price = strongest['price']
                magnet_direction = "DOWN" if magnet_price < current_price else "UP"
            else:
                magnet_price = current_price
                magnet_direction = "NEUTRAL"
            
            result = {
                'current_price': current_price,
                'long_liquidation_clusters': long_clusters,
                'short_liquidation_clusters': short_clusters,
                'nearest_long_liq': long_clusters[0] if long_clusters else None,
                'nearest_short_liq': short_clusters[0] if short_clusters else None,
                'magnet_price': magnet_price,
                'magnet_direction': magnet_direction,
                'high_24h': high_24h,
                'low_24h': low_24h
            }
            
            logger.info(f"🎯 Liq Hunter: Magnet at ${magnet_price:,.0f} ({magnet_direction})")
            return result
            
        except Exception as e:
            logger.error(f"Liquidation calculation error: {e}")
            return {'error': str(e)}
    
    async def get_funding_extremes(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Funding rate aşırılıkları.
        Aşırı pozitif = Çok long, short fırsatı
        Aşırı negatif = Çok short, long fırsatı
        """
        try:
            session = await self._get_session()
            
            # Binance Futures funding rate
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=10"
            
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {'funding_rate': 0, 'signal': 'neutral'}
                
                data = await resp.json()
            
            if not data:
                return {'funding_rate': 0, 'signal': 'neutral'}
            
            current_funding = float(data[-1]['fundingRate']) * 100  # Percentage
            
            # Son 10 funding'in ortalaması
            avg_funding = sum(float(d['fundingRate']) for d in data) / len(data) * 100
            
            # Extreme detection
            if current_funding > 0.1:  # > 0.1% = Very bullish crowd
                signal = "EXTREME_LONG"  # Contrarian: Short fırsatı
                risk = "HIGH"
            elif current_funding > 0.05:
                signal = "BULLISH_BIAS"
                risk = "MEDIUM"
            elif current_funding < -0.1:  # < -0.1% = Very bearish crowd
                signal = "EXTREME_SHORT"  # Contrarian: Long fırsatı
                risk = "HIGH"
            elif current_funding < -0.05:
                signal = "BEARISH_BIAS"
                risk = "MEDIUM"
            else:
                signal = "NEUTRAL"
                risk = "LOW"
            
            result = {
                'current_funding': current_funding,
                'avg_funding_10': avg_funding,
                'signal': signal,
                'risk': risk,
                'contrarian_play': "SHORT" if current_funding > 0.05 else "LONG" if current_funding < -0.05 else "NONE"
            }
            
            logger.info(f"💵 Funding: {current_funding:.4f}% | Signal: {signal}")
            return result
            
        except Exception as e:
            logger.error(f"Funding rate error: {e}")
            return {'funding_rate': 0, 'signal': 'neutral'}
    
    async def get_long_short_ratio(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Long/Short oranı.
        Kalabalığın tersi genelde doğru.
        """
        try:
            session = await self._get_session()
            
            # Binance Futures Long/Short Ratio
            url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=10"
            
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {'ratio': 1.0, 'signal': 'neutral'}
                
                data = await resp.json()
            
            if not data:
                return {'ratio': 1.0, 'signal': 'neutral'}
            
            current_ratio = float(data[-1]['longShortRatio'])
            
            # Trend
            if len(data) >= 3:
                prev_ratio = float(data[-3]['longShortRatio'])
                trend = "INCREASING" if current_ratio > prev_ratio else "DECREASING"
            else:
                trend = "STABLE"
            
            # Extreme detection
            if current_ratio > 2.0:  # %66+ long
                signal = "EXTREME_LONG"
                contrarian = "SHORT"
            elif current_ratio > 1.5:
                signal = "BULLISH_CROWD"
                contrarian = "CAUTION_LONG"
            elif current_ratio < 0.5:  # %66+ short
                signal = "EXTREME_SHORT"
                contrarian = "LONG"
            elif current_ratio < 0.7:
                signal = "BEARISH_CROWD"
                contrarian = "CAUTION_SHORT"
            else:
                signal = "BALANCED"
                contrarian = "NONE"
            
            result = {
                'long_short_ratio': current_ratio,
                'long_pct': (current_ratio / (1 + current_ratio)) * 100,
                'short_pct': (1 / (1 + current_ratio)) * 100,
                'signal': signal,
                'contrarian_play': contrarian,
                'trend': trend
            }
            
            logger.info(f"⚖️ L/S Ratio: {current_ratio:.2f} | {result['long_pct']:.1f}% Long | Signal: {signal}")
            return result
            
        except Exception as e:
            logger.error(f"Long/Short ratio error: {e}")
            return {'ratio': 1.0, 'signal': 'neutral'}
    
    async def get_open_interest_change(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Open Interest değişimi.
        OI artışı + fiyat artışı = Güçlü trend
        OI düşüşü + fiyat artışı = Zayıf rally
        """
        try:
            session = await self._get_session()
            
            url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=1h&limit=24"
            
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {'oi_change': 0, 'signal': 'neutral'}
                
                data = await resp.json()
            
            if len(data) < 2:
                return {'oi_change': 0, 'signal': 'neutral'}
            
            current_oi = float(data[-1]['sumOpenInterestValue'])
            prev_oi = float(data[0]['sumOpenInterestValue'])
            
            oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100
            
            # Signal based on OI change magnitude
            if oi_change_pct > 10:
                signal = "STRONG_INFLOW"
                interpretation = "Yeni pozisyonlar açılıyor - trend güçleniyor"
            elif oi_change_pct > 5:
                signal = "MODERATE_INFLOW"
                interpretation = "Pozisyon artışı"
            elif oi_change_pct < -10:
                signal = "STRONG_OUTFLOW"
                interpretation = "Pozisyonlar kapatılıyor - trend zayıflıyor"
            elif oi_change_pct < -5:
                signal = "MODERATE_OUTFLOW"
                interpretation = "Pozisyon azalışı"
            else:
                signal = "STABLE"
                interpretation = "Stabil OI"
            
            result = {
                'current_oi': current_oi,
                'oi_change_24h': oi_change_pct,
                'signal': signal,
                'interpretation': interpretation
            }
            
            logger.info(f"📈 OI Change: {oi_change_pct:.1f}% | {signal}")
            return result
            
        except Exception as e:
            logger.error(f"Open Interest error: {e}")
            return {'oi_change': 0, 'signal': 'neutral'}
    
    async def get_full_liquidation_analysis(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Tüm likidite analizini birleştir.
        """
        liq_levels, funding, ls_ratio, oi = await asyncio.gather(
            self.calculate_liquidation_levels(symbol),
            self.get_funding_extremes(symbol),
            self.get_long_short_ratio(symbol),
            self.get_open_interest_change(symbol)
        )
        
        # Composite Signal
        score = 0
        
        # Funding contrarian
        if funding.get('signal') == 'EXTREME_LONG':
            score -= 20  # Çok long = bearish
        elif funding.get('signal') == 'EXTREME_SHORT':
            score += 20  # Çok short = bullish
        
        # L/S Ratio contrarian
        if ls_ratio.get('signal') == 'EXTREME_LONG':
            score -= 15
        elif ls_ratio.get('signal') == 'EXTREME_SHORT':
            score += 15
        
        # OI signal
        if 'INFLOW' in oi.get('signal', ''):
            score += 10  # Yeni para = bullish
        elif 'OUTFLOW' in oi.get('signal', ''):
            score -= 10
        
        # Magnet direction
        if liq_levels.get('magnet_direction') == 'UP':
            score += 5
        elif liq_levels.get('magnet_direction') == 'DOWN':
            score -= 5
        
        # Final signal
        if score >= 25:
            final_signal = "BULLISH"
        elif score <= -25:
            final_signal = "BEARISH"
        else:
            final_signal = "NEUTRAL"
        
        return {
            'liquidation_levels': liq_levels,
            'funding': funding,
            'long_short_ratio': ls_ratio,
            'open_interest': oi,
            'composite_score': score,
            'signal': final_signal,
            'timestamp': datetime.now().isoformat()
        }
    
    async def close(self):
        if self.session:
            await self.session.close()
