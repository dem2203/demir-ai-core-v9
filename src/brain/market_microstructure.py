import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("MARKET_MICROSTRUCTURE")

class MarketMicrostructure:
    """
    PROFESSIONAL Crypto Market Analysis
    - Order Book Imbalance
    - Liquidation Zones
    - Funding Rates
    - Volume Profile
    - CVD (Cumulative Volume Delta)
    
    ALL REAL DATA - NO MOCKS!
    """
    
    def __init__(self, binance_api):
        self.binance = binance_api
        
    async def analyze_orderbook_imbalance(self, symbol: str) -> Dict:
        """
        Order Book Imbalance Detection
        
        Whale walls, bid/ask imbalance = directional pressure
        """
        try:
            if not self.binance.exchange:
                await self.binance.connect()
            
            ticker = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            
            # Fetch deep order book (1000 levels)
            orderbook = await self.binance.exchange.fetch_order_book(ticker, limit=1000)
            
            bids = pd.DataFrame(orderbook['bids'], columns=['price', 'size'])
            asks = pd.DataFrame(orderbook['asks'], columns=['price', 'size'])
            
            # Calculate total liquidity
            total_bid_volume = bids['size'].sum()
            total_ask_volume = asks['size'].sum()
            
            # Imbalance ratio
            imbalance_ratio = total_bid_volume / (total_bid_volume + total_ask_volume)
            
            # Detect whale walls (top 1% of orders)
            bid_threshold = bids['size'].quantile(0.99)
            ask_threshold = asks['size'].quantile(0.99)
            
            whale_bids = bids[bids['size'] >= bid_threshold]
            whale_asks = asks[asks['size'] >= ask_threshold]
            
            # FIXED SCALING! Was 20x (too weak), now 100x
            if imbalance_ratio > 0.55:
                signal = "BULLISH"
                strength = min(int((imbalance_ratio - 0.5) * 100), 10)  # 100x scaling!
                reason = f"Order book shows {imbalance_ratio*100:.1f}% buy pressure"
            elif imbalance_ratio < 0.45:
                signal = "BEARISH"
                strength = min(int((0.5 - imbalance_ratio) * 100), 10)  # 100x scaling!
                reason = f"Order book shows {(1-imbalance_ratio)*100:.1f}% sell pressure"
            else:
                signal = "NEUTRAL"
                strength = 5
                reason = "Balanced order book"
            
            # Add whale wall info
            if len(whale_bids) > len(whale_asks) * 2:
                reason += f" | {len(whale_bids)} whale buy walls detected"
            elif len(whale_asks) > len(whale_bids) * 2:
                reason += f" | {len(whale_asks)} whale sell walls detected"
            
            logger.info(f"📊 Order Book: {signal} | Imbalance: {imbalance_ratio:.2%}")
            
            return {
                'signal': signal,
                'strength': int(strength),
                'imbalance_ratio': imbalance_ratio,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'whale_bids': len(whale_bids),
                'whale_asks': len(whale_asks),
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Order book analysis failed: {e}")
            return {'signal': 'ERROR', 'strength': 0, 'reason': str(e)}
    
    async def analyze_funding_rate(self, symbol: str) -> Dict:
        """
        Funding Rate Analysis
        
        High positive = longs paying shorts = potential short
        High negative = shorts paying longs = potential long
        """
        try:
            if not self.binance.exchange:
                await self.binance.connect()
            
            ticker = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            
            # Fetch funding rate (Futures only)
            funding_info = await self.binance.exchange.fetch_funding_rate(ticker)
            
            funding_rate = funding_info['fundingRate']
            funding_rate_pct = funding_rate * 100
            
            # Annualized funding (funding happens 3x/day on Binance)
            annual_funding = funding_rate * 3 * 365 * 100
            
            # LOWER THRESHOLDS based on real market test!
            if annual_funding > 10:  # Was 15%, now 10%
                signal = "BEARISH"
                strength = min(int(annual_funding / 1.2), 10)
                reason = f"High positive funding ({annual_funding:.1f}% APR) - longs overleveraged"
            elif annual_funding < -10:  # Was -15%, now -10%
                signal = "BULLISH"
                strength = min(int(abs(annual_funding) / 1.2), 10)
                reason = f"Negative funding ({annual_funding:.1f}% APR) - shorts overleveraged"
            elif annual_funding > 3:  # Was 5%, now 3%
                signal = "BEARISH"
                strength = 6
                reason = f"Elevated funding rate ({annual_funding:.1f}% APR) - long bias"
            elif annual_funding < -3:  # Was -5%, now -3%
                signal = "BULLISH"
                strength = 6
                reason = f"Negative funding rate ({annual_funding:.1f}% APR) - short bias"
            else:
                signal = "NEUTRAL"
                strength = 5
                reason = f"Balanced funding rate ({annual_funding:.1f}% APR)"
            
            logger.info(f"💰 Funding: {annual_funding:.1f}% APR → {signal}")
            
            return {
                'signal': signal,
                'strength': strength,
                'funding_rate': funding_rate,
                'annual_funding_pct': annual_funding,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Funding rate analysis failed: {e}")
            return {'signal': 'ERROR', 'strength': 0, 'reason': str(e)}
    
    async def analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """
        Volume Profile Analysis
        
        POC (Point of Control) = highest volume price level
        Value Area = 70% of volume
        """
        try:
            if df.empty or len(df) < 20:
                return {'signal': 'NEUTRAL', 'strength': 5, 'reason': 'Insufficient data'}
            
            # Create price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            num_bins = 50
            
            price_bins = np.linspace(price_min, price_max, num_bins)
            volume_at_price = np.zeros(num_bins - 1)
            
            # Distribute volume across price levels
            for idx, row in df.iterrows():
                # Simple distribution: assign volume to bins within candle range
                candle_low_idx = np.searchsorted(price_bins, row['low'])
                candle_high_idx = np.searchsorted(price_bins, row['high'])
                
                if candle_low_idx < candle_high_idx:
                    volume_per_bin = row['volume'] / (candle_high_idx - candle_low_idx)
                    volume_at_price[candle_low_idx:candle_high_idx] += volume_per_bin
            
            # Find POC (Point of Control)
            poc_idx = np.argmax(volume_at_price)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            
            # Current price
            current_price = df['close'].iloc[-1]
            
            # LOWER THRESHOLDS based on real market test!
            distance_from_poc = (current_price - poc_price) / poc_price
            
            if abs(distance_from_poc) < 0.003:  # Within 0.3% of POC (was 0.5%)
                signal = "NEUTRAL"
                strength = 7
                reason = f"Price at POC (${poc_price:.2f}) - high volume node"
            elif current_price > poc_price * 1.010:  # 1.0% above POC (was 1.5%)
                signal = "BEARISH"
                strength = 7
                reason = f"Price {distance_from_poc*100:.1f}% above POC (${poc_price:.2f}) - potential reversion"
            elif current_price < poc_price * 0.990:  # 1.0% below POC (was 1.5%)
                signal = "BULLISH"
                strength = 7
                reason = f"Price {abs(distance_from_poc)*100:.1f}% below POC (${poc_price:.2f}) - potential bounce"
            else:
                signal = "NEUTRAL"
                strength = 5
                reason = f"Price near POC (${poc_price:.2f})"
            
            logger.info(f"📊 Volume Profile: POC=${poc_price:.2f} | Current=${current_price:.2f}")
            
            return {
                'signal': signal,
                'strength': strength,
                'poc_price': poc_price,
                'current_price': current_price,
                'distance_from_poc_pct': distance_from_poc * 100,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Volume profile analysis failed: {e}")
            return {'signal': 'ERROR', 'strength': 0, 'reason': str(e)}
    
    async def analyze_cvd(self, df: pd.DataFrame) -> Dict:
        """
        Cumulative Volume Delta (CVD)
        
        Buy volume vs Sell volume accumulation
        Divergence with price = strong signal
        """
        try:
            if df.empty or len(df) < 20:
                return {'signal': 'NEUTRAL', 'strength': 5, 'reason': 'Insufficient data'}
            
            # Estimate buy/sell volume from candle direction
            # Up candle (close > open) = more buying
            # Down candle (close < open) = more selling
            
            df = df.copy()
            df['delta'] = np.where(
                df['close'] > df['open'],
                df['volume'],  # Buy volume
                -df['volume']  # Sell volume
            )
            
            # Cumulative sum
            df['cvd'] = df['delta'].cumsum()
            
            # CVD trend
            cvd_current = df['cvd'].iloc[-1]
            cvd_20_ago = df['cvd'].iloc[-20] if len(df) >= 20 else df['cvd'].iloc[0]
            cvd_change = cvd_current - cvd_20_ago
            
            # Price trend
            price_current = df['close'].iloc[-1]
            price_20_ago = df['close'].iloc[-20] if len(df) >= 20 else df['close'].iloc[0]
            price_change_pct = (price_current - price_20_ago) / price_20_ago * 100
            
            # Detect divergence
            if cvd_change > 0 and price_change_pct < -1:  # CVD up, price down
                signal = "BULLISH"
                strength = 8
                reason = "Bullish divergence: CVD climbing while price falls (hidden buying)"
            elif cvd_change < 0 and price_change_pct > 1:  # CVD down, price up
                signal = "BEARISH"
                strength = 8
                reason = "Bearish divergence: CVD falling while price rises (hidden selling)"
            elif cvd_change > 0 and price_change_pct > 0:  # Both up
                signal = "BULLISH"
                strength = 7
                reason = "CVD confirms uptrend (buying pressure)"
            elif cvd_change < 0 and price_change_pct < 0:  # Both down
                signal = "BEARISH"
                strength = 7
                reason = "CVD confirms downtrend (selling pressure)"
            else:
                signal = "NEUTRAL"
                strength = 5
                reason = "CVD neutral"
            
            logger.info(f"📈 CVD: {signal} | Change: {cvd_change:,.0f}")
            
            return {
                'signal': signal,
                'strength': strength,
                'cvd_current': cvd_current,
                'cvd_change': cvd_change,
                'price_change_pct': price_change_pct,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"CVD analysis failed: {e}")
            return {'signal': 'ERROR', 'strength': 0, 'reason': str(e)}
