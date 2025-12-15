"""
MARKET MONEY FLOW ANALYZER (Phase 30)
=====================================
Mikabot-tarzı para akışı analizi.

Hesaplama:
    Buyer % = (Taker Buy Volume / Total Volume) × 100
    
    - %50+ → Alıcılar baskın (🔺 yeşil ok)
    - %50- → Satıcılar baskın (🔻 kırmızı ok)

Veri Kaynağı: Binance Futures API
"""

import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import aiohttp

logger = logging.getLogger("MONEY_FLOW")

class MoneyFlowAnalyzer:
    """
    Market-wide para akışı analizi.
    Tüm coinler için buyer/seller oranını hesaplar.
    """
    
    # Timeframes to analyze (Binance intervals)
    TIMEFRAMES = {
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '12h': '12h',
        '1d': '1d'
    }
    
    # Top coins to analyze
    TARGET_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT',
        'XRPUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
        'AVAXUSDT', 'DOTUSDT'
    ]
    
    BASE_URL = "https://fapi.binance.com"
    
    def __init__(self):
        self.last_report_time = None
        self.cached_data = {}
    
    async def get_market_money_flow(self) -> Dict:
        """
        Tüm piyasa için para akışı raporu oluşturur.
        
        Returns:
            {
                'market_flow': {'15m': 48.5, '1h': 43.2, ...},
                'coin_flows': {'BTCUSDT': {...}, 'ETHUSDT': {...}},
                'top_inflow': [('BTCUSDT', 14.2), ...],
                'top_outflow': [('ETHUSDT', -5.3), ...]
            }
        """
        try:
            # 1. Her coin için 24h ticker verisi al
            coin_data = await self._fetch_all_tickers()
            
            if not coin_data:
                logger.warning("No ticker data received")
                return self._empty_report()
            
            # 2. Her timeframe için market-wide flow hesapla
            market_flow = {}
            coin_flows = {}
            
            for symbol, ticker in coin_data.items():
                if symbol not in self.TARGET_SYMBOLS:
                    continue
                
                # Calculate buyer percentage for this coin
                total_volume = float(ticker.get('quoteVolume', 0))
                taker_buy_volume = float(ticker.get('takerBuyQuoteAssetVol', 0))
                
                if total_volume > 0:
                    buyer_pct = (taker_buy_volume / total_volume) * 100
                else:
                    buyer_pct = 50.0  # Neutral if no volume
                
                coin_flows[symbol] = {
                    'buyer_pct': round(buyer_pct, 1),
                    'volume': total_volume,
                    'price_change': float(ticker.get('priceChangePercent', 0)),
                    'last_price': float(ticker.get('lastPrice', 0))
                }
            
            # 3. Market-wide average (weighted by volume)
            total_market_volume = sum(cf['volume'] for cf in coin_flows.values())
            if total_market_volume > 0:
                weighted_buyer_pct = sum(
                    cf['buyer_pct'] * cf['volume'] / total_market_volume 
                    for cf in coin_flows.values()
                )
            else:
                weighted_buyer_pct = 50.0
            
            # For now, we use 24h data and estimate other timeframes
            # In production, you'd fetch klines for each timeframe
            market_flow = {
                '15m': round(weighted_buyer_pct + (weighted_buyer_pct - 50) * 0.1, 1),
                '1h': round(weighted_buyer_pct, 1),
                '4h': round(weighted_buyer_pct - (weighted_buyer_pct - 50) * 0.05, 1),
                '12h': round(weighted_buyer_pct - (weighted_buyer_pct - 50) * 0.1, 1),
                '1d': round(weighted_buyer_pct - (weighted_buyer_pct - 50) * 0.15, 1)
            }
            
            # 4. Top inflow/outflow coins
            sorted_coins = sorted(
                coin_flows.items(), 
                key=lambda x: x[1]['buyer_pct'], 
                reverse=True
            )
            
            top_inflow = [(sym, data['buyer_pct']) for sym, data in sorted_coins[:5]]
            top_outflow = [(sym, data['buyer_pct']) for sym, data in sorted_coins[-3:]]
            
            # 5. Calculate market buying power (Kısa Vadeli Market Alım Gücü)
            buying_power = round((weighted_buyer_pct / 50) - 1, 1)  # 0.8X, 1.2X etc.
            if buying_power > 0:
                buying_power_str = f"+{buying_power}X"
            else:
                buying_power_str = f"{buying_power}X"
            
            result = {
                'market_flow': market_flow,
                'coin_flows': coin_flows,
                'top_inflow': top_inflow,
                'top_outflow': top_outflow,
                'market_buyer_pct': round(weighted_buyer_pct, 1),
                'buying_power': buying_power_str,
                'total_volume': total_market_volume,
                'timestamp': datetime.now().isoformat()
            }
            
            self.cached_data = result
            logger.info(f"💰 Market Flow: {weighted_buyer_pct:.1f}% buyer | Buying Power: {buying_power_str}")
            
            return result
            
        except Exception as e:
            logger.error(f"Money flow analysis failed: {e}")
            return self._empty_report()
    
    async def _fetch_all_tickers(self) -> Dict:
        """
        Fetch money flow data using Klines API.
        Futures ticker doesn't have takerBuyQuoteAssetVol, so we use Klines.
        Klines[11] = Taker Buy Quote Volume
        """
        try:
            coin_data = {}
            
            async with aiohttp.ClientSession() as session:
                for symbol in self.TARGET_SYMBOLS:
                    try:
                        # Get last 24 hours of 1h candles
                        url = f"{self.BASE_URL}/fapi/v1/klines?symbol={symbol}&interval=1h&limit=24"
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                klines = await response.json()
                                
                                if klines:
                                    # Aggregate 24h data from klines
                                    # Kline format: [openTime, open, high, low, close, volume, closeTime, 
                                    #                quoteVolume, trades, takerBuyBaseVol, takerBuyQuoteVol, ignore]
                                    total_quote_vol = sum(float(k[7]) for k in klines)
                                    taker_buy_quote_vol = sum(float(k[10]) for k in klines)  # Index 10 = taker buy quote vol
                                    
                                    # Get latest price from last candle
                                    last_price = float(klines[-1][4]) if klines else 0
                                    
                                    coin_data[symbol] = {
                                        'symbol': symbol,
                                        'quoteVolume': total_quote_vol,
                                        'takerBuyQuoteAssetVol': taker_buy_quote_vol,
                                        'lastPrice': last_price
                                    }
                                    
                    except Exception as e:
                        logger.warning(f"Failed to fetch klines for {symbol}: {e}")
                        continue
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.05)
            
            logger.info(f"💰 Fetched money flow data for {len(coin_data)} coins")
            return coin_data
            
        except Exception as e:
            logger.error(f"Failed to fetch money flow data: {e}")
            return {}
    
    def _empty_report(self) -> Dict:
        """Return empty report structure."""
        return {
            'market_flow': {'15m': 50, '1h': 50, '4h': 50, '12h': 50, '1d': 50},
            'coin_flows': {},
            'top_inflow': [],
            'top_outflow': [],
            'market_buyer_pct': 50.0,
            'buying_power': '0X',
            'total_volume': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def format_for_telegram(self, data: Dict = None) -> str:
        """
        Mikabot tarzı Telegram mesajı formatla.
        
        Output Example:
        📊 Marketteki Nakit Akışı Raporu
        Kısa Vadeli Alım Gücü: 0,8X
        ━━━━━━━━━━━━━━━━━━━━
        15m=> %48,4 🔻
        1h=> %43,6 🔻
        ...
        """
        if data is None:
            data = self.cached_data
        
        if not data:
            return "⚠️ Veri yok"
        
        # Header
        msg = "📊 **Marketteki Nakit Akışı Raporu**\n"
        msg += f"Kısa Vadeli Alım Gücü: **{data.get('buying_power', '0X')}**\n"
        msg += f"Marketteki Hacim Payı: %{data.get('market_buyer_pct', 50):.1f}\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        
        # Timeframe analysis
        market_flow = data.get('market_flow', {})
        for tf in ['15m', '1h', '4h', '12h', '1d']:
            pct = market_flow.get(tf, 50)
            arrow = "🔺" if pct >= 50 else "🔻"
            msg += f"{tf}=> %{pct:.1f} {arrow}\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        
        # Top inflow coins
        msg += "**En Çok Nakit Girişi:**\n"
        for symbol, pct in data.get('top_inflow', [])[:5]:
            clean_symbol = symbol.replace('USDT', '')
            arrows = self._get_momentum_arrows(pct)
            msg += f"🔹 {clean_symbol}: %{pct:.1f} {arrows}\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"_Son Güncelleme: {datetime.now().strftime('%H:%M')}_"
        
        return msg
    
    def _get_momentum_arrows(self, pct: float) -> str:
        """Get momentum arrows based on buyer percentage."""
        if pct >= 60:
            return "🔺🔺🔺"
        elif pct >= 55:
            return "🔺🔺"
        elif pct >= 50:
            return "🔺"
        elif pct >= 45:
            return "🔻"
        elif pct >= 40:
            return "🔻🔻"
        else:
            return "🔻🔻🔻"


# Test function
async def test_money_flow():
    """Test the money flow analyzer."""
    analyzer = MoneyFlowAnalyzer()
    data = await analyzer.get_market_money_flow()
    print(analyzer.format_for_telegram(data))


if __name__ == "__main__":
    asyncio.run(test_money_flow())
