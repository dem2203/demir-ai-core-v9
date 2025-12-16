"""
PREDICTIVE ANALYZER - Leading Indicator Signals
================================================
Phase 32.5: Gerçek Önceden Uyarı + Trade Sinyalleri

Öncü Göstergeler (Ücretsiz API):
1. BTC Mempool Whale Detection (Blockchain.info)
2. Funding Rate Extreme (Binance)
3. OI Divergence (Binance)
4. Liquidation Risk (Binance)

Çıktı: Entry, Stop Loss, Take Profit ile sinyal
"""

import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("PREDICTIVE")

class PredictiveAnalyzer:
    """
    Önceden Uyarı Sistemi - Hareket Olmadan Önce Tespit
    
    Her sinyal için:
    - Direction: LONG veya SHORT
    - Entry: Giriş fiyatı
    - Stop Loss: Risk limiti
    - Take Profit: Hedef
    - Confidence: Güven seviyesi
    - Time Horizon: Tahmini süre
    """
    
    def __init__(self):
        self.mempool_url = "https://blockchain.info/unconfirmed-transactions?format=json"
        self.binance_url = "https://fapi.binance.com"
        self.last_check = {}
        self.cache = {}
    
    async def analyze_predictive_signals(self, symbol: str = "BTCUSDT", current_price: float = 0) -> Dict:
        """
        Tüm öncü göstergeleri analiz et ve sinyal üret.
        
        Returns:
            {
                'has_signal': True/False,
                'signal_type': 'PREDICTIVE_LONG' / 'PREDICTIVE_SHORT',
                'direction': 'LONG' / 'SHORT',
                'entry': 95000,
                'stop_loss': 94000,
                'take_profit_1': 97000,
                'take_profit_2': 99000,
                'confidence': 75,
                'reasons': ['Funding extreme', 'OI divergence'],
                'time_horizon': '2-6 saat'
            }
        """
        try:
            signals = []
            reasons = []
            
            async with aiohttp.ClientSession() as session:
                # 1. Funding Rate Check
                funding_signal = await self._check_funding_rate(session, symbol, current_price)
                if funding_signal:
                    signals.append(funding_signal)
                    reasons.append(funding_signal['reason'])
                
                # 2. OI Divergence Check
                oi_signal = await self._check_oi_divergence(session, symbol, current_price)
                if oi_signal:
                    signals.append(oi_signal)
                    reasons.append(oi_signal['reason'])
                
                # 3. Long/Short Ratio Extreme
                ls_signal = await self._check_ls_extreme(session, symbol, current_price)
                if ls_signal:
                    signals.append(ls_signal)
                    reasons.append(ls_signal['reason'])
                
                # 4. BTC Mempool (only for BTC)
                if 'BTC' in symbol:
                    mempool_signal = await self._check_mempool_whales(session, current_price)
                    if mempool_signal:
                        signals.append(mempool_signal)
                        reasons.append(mempool_signal['reason'])
            
            # Combine signals
            if not signals:
                return {'has_signal': False}
            
            # Determine overall direction
            long_votes = sum(1 for s in signals if s.get('direction') == 'LONG')
            short_votes = sum(1 for s in signals if s.get('direction') == 'SHORT')
            
            if long_votes == short_votes:
                return {'has_signal': False, 'reason': 'Conflicting signals'}
            
            direction = 'LONG' if long_votes > short_votes else 'SHORT'
            confidence = max(s.get('confidence', 50) for s in signals)
            
            # Calculate Entry/SL/TP based on direction
            if direction == 'LONG':
                entry = current_price * 0.998  # Slightly below current
                stop_loss = current_price * 0.97  # 3% below
                tp1 = current_price * 1.02  # 2% above
                tp2 = current_price * 1.04  # 4% above
            else:
                entry = current_price * 1.002  # Slightly above current
                stop_loss = current_price * 1.03  # 3% above
                tp1 = current_price * 0.98  # 2% below
                tp2 = current_price * 0.96  # 4% below
            
            result = {
                'has_signal': True,
                'signal_type': f'PREDICTIVE_{direction}',
                'direction': direction,
                'entry': round(entry, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit_1': round(tp1, 2),
                'take_profit_2': round(tp2, 2),
                'confidence': confidence,
                'reasons': reasons,
                'time_horizon': '2-6 saat',
                'signal_count': len(signals),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🔮 PREDICTIVE: {direction} signal with {len(signals)} confirmations")
            return result
            
        except Exception as e:
            logger.error(f"Predictive analysis failed: {e}")
            return {'has_signal': False, 'error': str(e)}
    
    async def _check_funding_rate(self, session, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Funding Rate Extreme = Pozisyon dengesizliği
        
        > 0.05% = Çok fazla long → SHORT sinyali
        < -0.03% = Çok fazla short → LONG sinyali
        """
        try:
            url = f"{self.binance_url}/fapi/v1/premiumIndex?symbol={symbol}"
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    funding_rate = float(data.get('lastFundingRate', 0))
                    fr_pct = funding_rate * 100
                    
                    if fr_pct >= 0.05:
                        return {
                            'direction': 'SHORT',
                            'reason': f'Funding Rate %{fr_pct:.3f} (Aşırı Long)',
                            'confidence': min(70 + int(fr_pct * 100), 90)
                        }
                    elif fr_pct <= -0.03:
                        return {
                            'direction': 'LONG',
                            'reason': f'Funding Rate %{fr_pct:.3f} (Aşırı Short)',
                            'confidence': min(65 + int(abs(fr_pct) * 100), 85)
                        }
            return None
        except Exception as e:
            logger.debug(f"Funding rate check failed: {e}")
            return None
    
    async def _check_oi_divergence(self, session, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Open Interest Divergence = Pozisyon birikimi
        
        OI ↑ + Fiyat ↓ = Short birikimi → LONG squeeze potansiyeli
        OI ↑ + Fiyat ↑ = Long birikimi → SHORT likidasyon riski
        """
        try:
            # Get current OI
            url = f"{self.binance_url}/fapi/v1/openInterest?symbol={symbol}"
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    current_oi = float(data.get('openInterest', 0))
                    
                    # Compare with cached OI (if exists)
                    cache_key = f"oi_{symbol}"
                    if cache_key in self.cache:
                        prev_oi = self.cache[cache_key]['oi']
                        prev_price = self.cache[cache_key]['price']
                        
                        oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100 if prev_oi > 0 else 0
                        price_change_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
                        
                        # Divergence detection
                        if oi_change_pct > 5 and price_change_pct < -1:
                            # OI up, price down = shorts accumulating
                            return {
                                'direction': 'LONG',
                                'reason': f'OI Divergence: OI +{oi_change_pct:.1f}% ama Fiyat {price_change_pct:.1f}%',
                                'confidence': 65
                            }
                        elif oi_change_pct > 5 and price_change_pct > 1:
                            # OI up, price up = longs accumulating
                            return {
                                'direction': 'SHORT',
                                'reason': f'OI Divergence: OI +{oi_change_pct:.1f}% ve Fiyat +{price_change_pct:.1f}% (aşırı long)',
                                'confidence': 60
                            }
                    
                    # Update cache
                    self.cache[cache_key] = {'oi': current_oi, 'price': current_price, 'time': datetime.now()}
            
            return None
        except Exception as e:
            logger.debug(f"OI divergence check failed: {e}")
            return None
    
    async def _check_ls_extreme(self, session, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Long/Short Ratio Extreme = Crowd positioning
        
        > 2.5 = Herkes long → Düzeltme yakın
        < 0.4 = Herkes short → Squeeze yakın
        """
        try:
            url = f"{self.binance_url}/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        ls_ratio = float(data[0].get('longShortRatio', 1))
                        
                        if ls_ratio >= 2.5:
                            return {
                                'direction': 'SHORT',
                                'reason': f'L/S Ratio {ls_ratio:.2f} (Herkes Long → Düzeltme riski)',
                                'confidence': 70
                            }
                        elif ls_ratio <= 0.4:
                            return {
                                'direction': 'LONG',
                                'reason': f'L/S Ratio {ls_ratio:.2f} (Herkes Short → Squeeze potansiyeli)',
                                'confidence': 70
                            }
            return None
        except Exception as e:
            logger.debug(f"L/S ratio check failed: {e}")
            return None
    
    async def _check_mempool_whales(self, session, current_price: float) -> Optional[Dict]:
        """
        BTC Mempool Whale Detection
        
        Büyük bekleyen işlemler (>100 BTC) = Hareket yaklaşıyor
        """
        try:
            async with session.get(self.mempool_url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    txs = data.get('txs', [])
                    
                    whale_count = 0
                    total_btc = 0
                    
                    for tx in txs[:100]:  # Check first 100 pending txs
                        inputs = tx.get('inputs', [])
                        total_satoshi = sum(i.get('prev_out', {}).get('value', 0) for i in inputs)
                        btc_value = total_satoshi / 100_000_000
                        
                        if btc_value > 50:  # >50 BTC is whale
                            whale_count += 1
                            total_btc += btc_value
                    
                    if whale_count >= 3 or total_btc > 500:
                        # Large pending volume = expect volatility
                        return {
                            'direction': 'SHORT',  # Large pending usually = selling
                            'reason': f'Mempool: {whale_count} whale TX ({total_btc:.0f} BTC bekliyor)',
                            'confidence': 55
                        }
            return None
        except Exception as e:
            logger.debug(f"Mempool check failed: {e}")
            return None
    
    def format_predictive_signal(self, signal: Dict) -> str:
        """
        Telegram formatında predictive sinyal
        """
        if not signal.get('has_signal'):
            return ""
        
        direction = signal['direction']
        emoji = "🟢" if direction == "LONG" else "🔴"
        
        msg = f"{emoji} **ÖNCEDEN UYARI SİNYALİ**\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"📍 **Yön:** {direction}\n"
        msg += f"💰 **Giriş:** ${signal['entry']:,.0f}\n"
        msg += f"🛡️ **Stop Loss:** ${signal['stop_loss']:,.0f}\n"
        msg += f"🎯 **TP1:** ${signal['take_profit_1']:,.0f}\n"
        msg += f"🎯 **TP2:** ${signal['take_profit_2']:,.0f}\n"
        msg += f"📊 **Güven:** %{signal['confidence']}\n"
        msg += f"⏰ **Süre:** {signal['time_horizon']}\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"**Nedenler:**\n"
        for reason in signal.get('reasons', []):
            msg += f"• {reason}\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"_⚠️ Bu bir TAHMİNDİR, garanti değildir!_"
        
        return msg


# Test
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        analyzer = PredictiveAnalyzer()
        signal = await analyzer.analyze_predictive_signals("BTCUSDT", 104000)
        print(analyzer.format_predictive_signal(signal))
    
    asyncio.run(test())
