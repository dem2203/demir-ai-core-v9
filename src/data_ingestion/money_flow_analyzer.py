"""
MARKET MONEY FLOW ANALYZER (Phase 30 v2)
=========================================
Mikabot-tarzı para akışı analizi - TAM FORMAT.

Her coin için:
    - Flow %: Toplam akış içindeki pay
    - 15m %: 15 dakikalık alıcı yüzdesi
    - Mts: Momentum değeri (0-1 arası)
    - Oklar: Her timeframe için yön (5m, 15m, 1h, 4h, 12h, 1d)

Veri Kaynağı: Binance Futures Klines API
"""

import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import aiohttp

logger = logging.getLogger("MONEY_FLOW")

class MoneyFlowAnalyzer:
    """
    Mikabot-tarzı detaylı para akışı analizi.
    Multi-timeframe, momentum ve coin bazlı analiz.
    """
    
    # Timeframes for arrows (in order: 5m, 15m, 1h, 4h, 12h, 1d)
    TIMEFRAMES = ['5m', '15m', '1h', '4h', '12h', '1d']
    
    # Klines limit per timeframe
    KLINES_LIMIT = {
        '5m': 12,   # 1 hour of data
        '15m': 4,   # 1 hour of data
        '1h': 1,    # Last hour
        '4h': 1,    # Last 4 hours
        '12h': 1,   # Last 12 hours
        '1d': 1     # Last day
    }
    
    # Top coins to analyze
    TARGET_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT',
        'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LTCUSDT'
    ]
    
    BASE_URL = "https://fapi.binance.com"
    
    def __init__(self):
        self.last_report_time = None
        self.cached_data = {}
    
    async def get_market_money_flow(self) -> Dict:
        """
        Mikabot formatında piyasa para akışı raporu.
        
        Returns:
            {
                'market_buyer_pct': 86.1,
                'market_15m_pct': 56.6,
                'buying_power': '1.8X',
                'timeframe_flows': {'15m': 56.6, '1h': 47.3, ...},
                'coin_details': [
                    {'symbol': 'BTC', 'flow_pct': 22.0, 'buyer_15m': 64, 'mts': 0.8, 'arrows': '🔺🔺🔺🔻🔻🔻'},
                    ...
                ]
            }
        """
        try:
            coin_details = []
            total_volume = 0
            total_buy_volume = 0
            
            async with aiohttp.ClientSession() as session:
                # Fetch data for each coin
                for symbol in self.TARGET_SYMBOLS:
                    coin_data = await self._fetch_coin_multi_tf(session, symbol)
                    if coin_data:
                        coin_details.append(coin_data)
                        total_volume += coin_data['volume_24h']
                        total_buy_volume += coin_data['buy_volume_24h']
            
            if not coin_details or total_volume == 0:
                return self._empty_report()
            
            # Calculate market-wide metrics
            market_buyer_pct = (total_buy_volume / total_volume) * 100 if total_volume > 0 else 50
            
            # Calculate flow percentage for each coin
            for coin in coin_details:
                coin['flow_pct'] = round((coin['volume_24h'] / total_volume) * 100, 1) if total_volume > 0 else 0
            
            # Sort by flow percentage (highest first)
            coin_details.sort(key=lambda x: x['flow_pct'], reverse=True)
            
            # Calculate timeframe averages
            timeframe_flows = {}
            for tf in self.TIMEFRAMES:
                tf_values = [c['tf_buyers'].get(tf, 50) for c in coin_details if c['tf_buyers'].get(tf)]
                timeframe_flows[tf] = round(sum(tf_values) / len(tf_values), 1) if tf_values else 50.0
            
            # Buying power calculation (like Mikabot)
            buying_power = round((market_buyer_pct / 50) - 1, 1)
            buying_power_str = f"+{buying_power}X" if buying_power >= 0 else f"{buying_power}X"
            
            result = {
                'market_buyer_pct': round(market_buyer_pct, 1),
                'market_15m_pct': timeframe_flows.get('15m', 50.0),
                'buying_power': buying_power_str,
                'timeframe_flows': timeframe_flows,
                'coin_details': coin_details[:10],  # Top 10 coins
                'total_volume': total_volume,
                'timestamp': datetime.now().isoformat()
            }
            
            self.cached_data = result
            logger.info(f"💰 Market Flow: {market_buyer_pct:.1f}% | 15m: {timeframe_flows.get('15m', 50):.1f}% | Power: {buying_power_str}")
            
            return result
            
        except Exception as e:
            logger.error(f"Money flow analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._empty_report()
    
    async def _fetch_coin_multi_tf(self, session: aiohttp.ClientSession, symbol: str) -> Optional[Dict]:
        """
        Fetch multi-timeframe data for a single coin.
        Returns buyer % and arrows for each timeframe.
        """
        try:
            tf_buyers = {}
            arrows = []
            volume_24h = 0
            buy_volume_24h = 0
            
            for tf in self.TIMEFRAMES:
                limit = self.KLINES_LIMIT.get(tf, 1)
                url = f"{self.BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={tf}&limit={limit}"
                
                try:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            klines = await response.json()
                            
                            if klines:
                                # Aggregate klines data
                                total_vol = sum(float(k[7]) for k in klines)  # Quote volume
                                taker_buy_vol = sum(float(k[10]) for k in klines)  # Taker buy quote volume
                                
                                if total_vol > 0:
                                    buyer_pct = (taker_buy_vol / total_vol) * 100
                                    tf_buyers[tf] = round(buyer_pct, 1)
                                    
                                    # Arrow based on buyer %
                                    if buyer_pct >= 50:
                                        arrows.append('🟢')
                                    else:
                                        arrows.append('🔴')
                                else:
                                    tf_buyers[tf] = 50.0
                                    arrows.append('➖')
                                
                                # Use 1d data for 24h volume
                                if tf == '1d' and klines:
                                    volume_24h = float(klines[-1][7])
                                    buy_volume_24h = float(klines[-1][10])
                                    
                except asyncio.TimeoutError:
                    tf_buyers[tf] = 50.0
                    arrows.append('➖')
                except Exception as e:
                    logger.debug(f"Error fetching {symbol} {tf}: {e}")
                    tf_buyers[tf] = 50.0
                    arrows.append('➖')
                
                await asyncio.sleep(0.02)  # Rate limit protection
            
            # Calculate momentum (based on 15m vs 1d difference)
            mts_15m = tf_buyers.get('15m', 50)
            mts_1d = tf_buyers.get('1d', 50)
            momentum = round(abs(mts_15m - 50) / 50, 1)  # 0-1 range
            
            return {
                'symbol': symbol.replace('USDT', ''),
                'tf_buyers': tf_buyers,
                'arrows': ''.join(arrows),
                'buyer_15m': tf_buyers.get('15m', 50),
                'buyer_1h': tf_buyers.get('1h', 50),
                'mts': momentum,
                'volume_24h': volume_24h,
                'buy_volume_24h': buy_volume_24h,
                'flow_pct': 0  # Will be calculated later
            }
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} data: {e}")
            return None
    
    def _empty_report(self) -> Dict:
        """Return empty report structure."""
        return {
            'market_buyer_pct': 50.0,
            'market_15m_pct': 50.0,
            'buying_power': '0X',
            'timeframe_flows': {tf: 50.0 for tf in self.TIMEFRAMES},
            'coin_details': [],
            'total_volume': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def format_for_telegram(self, data: Dict = None) -> str:
        """
        Mikabot-tarzı Telegram mesajı formatla.
        
        Output:
        📊 Marketteki Tüm Coinlere Olan Nakit Girişi Raporu.
        Kısa Vadeli Market Alım Gücü: 1.8X
        Marketteki Hacim Payı: %86.1
        ━━━━━━━━━━━━━━━━━━━━
        15m=> %56.6 🔺
        1h=> %47.3 🔻
        ...
        ━━━━━━━━━━━━━━━━━━━━
        En çok nakit girişi olanlar.
        ZEC Nakit: %22.0 15m:%64 Mts:0.8 🔺🔺🔺🔻🔻🔻
        BTC Nakit: %19.6 15m:%65 Mts:0.8 🔺🔺🔻🔻🔻🔻
        ...
        """
        if data is None:
            data = self.cached_data
        
        if not data or not data.get('coin_details'):
            return "⚠️ Veri yok"
        
        # Header
        msg = "📊 **Marketteki Tüm Coinlere Olan Nakit Girişi Raporu.**\n"
        msg += f"_Kısa Vadeli Market Alım Gücü:_ **{data.get('buying_power', '0X')}**\n"
        msg += f"_Marketteki Hacim Payı:_ **%{data.get('market_buyer_pct', 50):.1f}**\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        
        # Timeframe analysis
        tf_flows = data.get('timeframe_flows', {})
        for tf in ['15m', '1h', '4h', '12h', '1d']:
            pct = tf_flows.get(tf, 50)
            arrow = "🔺" if pct >= 50 else "🔻"
            msg += f"{tf}=> **%{pct:.1f}** {arrow}\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        
        # Coin details
        msg += "**En çok nakit girişi olanlar.**\n"
        msg += "_(Sonunda 🔺 olanlar sağlıklıdır)_\n"
        msg += "_Nakitin nereye aktığını gösterir._\n\n"
        
        for coin in data.get('coin_details', [])[:5]:  # Top 5
            symbol = coin.get('symbol', '???')
            flow_pct = coin.get('flow_pct', 0)
            buyer_15m = coin.get('buyer_15m', 50)
            mts = coin.get('mts', 0)
            arrows = coin.get('arrows', '➖➖➖➖➖➖')
            
            msg += f"**{symbol}** Nakit: %{flow_pct:.1f} 15m:%{buyer_15m:.0f} Mts:{mts} {arrows}\n"
        
        msg += f"\n━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"_Güncelleme: {datetime.now().strftime('%H:%M')}_"
        
        return msg


# Test function
async def test_money_flow():
    """Test the money flow analyzer."""
    analyzer = MoneyFlowAnalyzer()
    data = await analyzer.get_market_money_flow()
    print(analyzer.format_for_telegram(data))


if __name__ == "__main__":
    asyncio.run(test_money_flow())
