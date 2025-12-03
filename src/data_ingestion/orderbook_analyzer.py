import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import ccxt.async_support as ccxt
from src.config.settings import Config

logger = logging.getLogger("ORDERBOOK_ANALYZER")

class OrderBookAnalyzer:
    """
    DEMIR AI V20.0 - ORDER BOOK DEPTH ANALYZER
    
    Detects "Whale Walls" - large buy/sell orders that act as support/resistance.
    Uses Binance Level 2 Order Book data.
    """
    
    WHALE_THRESHOLD_USD = 1_000_000  # $1M+ orders are "whale walls"
    DEPTH_LIMIT = 100  # How many price levels to fetch
    
    def __init__(self):
        self.exchange = None
        self.exchange_config = {
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        }
    
    async def connect(self):
        try:
            self.exchange = ccxt.binance(self.exchange_config)
            await self.exchange.load_markets()
            logger.info("OrderBook Analyzer Connected.")
        except Exception as e:
            logger.error(f"OrderBook Connection Failed: {e}")
            self.exchange = None
    
    async def fetch_order_book(self, symbol: str) -> Optional[Dict]:
        """
        Fetches Level 2 Order Book (bids and asks with volumes).
        """
        if not self.exchange:
            await self.connect()
        if not self.exchange:
            return None
        
        try:
            # Fetch order book with depth limit
            orderbook = await self.exchange.fetch_order_book(symbol, limit=self.DEPTH_LIMIT)
            return orderbook
        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            return None
    
    def detect_whale_walls(self, orderbook: Dict, current_price: float) -> Dict:
        """
        Analyzes order book to find significant support/resistance levels.
        
        Returns:
            {
                'whale_support': price level with massive buy wall,
                'whale_resistance': price level with massive sell wall,
                'total_bid_liquidity': total USD in buy orders,
                'total_ask_liquidity': total USD in sell orders,
                'walls_detected': list of significant walls
            }
        """
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return {}
        
        bids = orderbook['bids']  # [[price, amount], ...]
        asks = orderbook['asks']
        
        # Calculate liquidity at each level
        bid_walls = []
        ask_walls = []
        
        total_bid_liquidity = 0
        total_ask_liquidity = 0
        
        # Analyze Bids (Support Levels)
        for price, amount in bids:
            usd_value = price * amount
            total_bid_liquidity += usd_value
            
            if usd_value >= self.WHALE_THRESHOLD_USD:
                bid_walls.append({
                    'price': price,
                    'amount': amount,
                    'usd_value': usd_value,
                    'distance_pct': ((price - current_price) / current_price) * 100
                })
        
        # Analyze Asks (Resistance Levels)
        for price, amount in asks:
            usd_value = price * amount
            total_ask_liquidity += usd_value
            
            if usd_value >= self.WHALE_THRESHOLD_USD:
                ask_walls.append({
                    'price': price,
                    'amount': amount,
                    'usd_value': usd_value,
                    'distance_pct': ((price - current_price) / current_price) * 100
                })
        
        # Find the strongest walls (closest to current price)
        whale_support = None
        whale_resistance = None
        
        if bid_walls:
            # Strongest support = largest wall closest to price
            bid_walls.sort(key=lambda x: abs(x['distance_pct']))
            whale_support = bid_walls[0]['price']
        
        if ask_walls:
            # Strongest resistance = largest wall closest to price
            ask_walls.sort(key=lambda x: abs(x['distance_pct']))
            whale_resistance = ask_walls[0]['price']
        
        return {
            'whale_support': whale_support,
            'whale_resistance': whale_resistance,
            'total_bid_liquidity': total_bid_liquidity,
            'total_ask_liquidity': total_ask_liquidity,
            'bid_walls': bid_walls[:3],  # Top 3 support walls
            'ask_walls': ask_walls[:3],  # Top 3 resistance walls
            'orderbook_imbalance': (total_bid_liquidity - total_ask_liquidity) / (total_bid_liquidity + total_ask_liquidity) if (total_bid_liquidity + total_ask_liquidity) > 0 else 0
        }
    
    async def analyze_orderbook(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Main method: Fetches order book and analyzes it.
        """
        orderbook = await self.fetch_order_book(symbol)
        if not orderbook:
            return None
        
        analysis = self.detect_whale_walls(orderbook, current_price)
        
        if analysis.get('whale_support'):
            logger.info(f"🐋 WHALE SUPPORT detected at ${analysis['whale_support']:,.2f} ({analysis['bid_walls'][0]['usd_value']/1e6:.2f}M)")
        if analysis.get('whale_resistance'):
            logger.info(f"🐋 WHALE RESISTANCE detected at ${analysis['whale_resistance']:,.2f} ({analysis['ask_walls'][0]['usd_value']/1e6:.2f}M)")
        
        # Order Book Imbalance Interpretation
        imbalance = analysis.get('orderbook_imbalance', 0)
        if imbalance > 0.2:
            logger.info(f"📊 Order Book: BULLISH (Buy pressure {imbalance*100:.1f}%)")
        elif imbalance < -0.2:
            logger.info(f"📊 Order Book: BEARISH (Sell pressure {abs(imbalance)*100:.1f}%)")
        
        return analysis
    
    def generate_liquidity_heatmap(self, orderbook: Dict, current_price: float, price_range_pct: float = 0.05) -> Dict:
        """
        Generates liquidity heatmap data for visualization.
        
        Args:
            orderbook: Order book dict
            current_price: Current market price
            price_range_pct: +/- percentage range to visualize (default 5%)
        
        Returns:
            {
                'price_levels': [99000, 99100, 99200, ...],
                'bid_volumes': [10.5, 8.2, ...],
                'ask_volumes': [0, 0, 9.3, ...]
            }
        """
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return {}
        
        # Define price range
        min_price = current_price * (1 - price_range_pct)
        max_price = current_price * (1 + price_range_pct)
        
        # Create price bins ($100 increments)
        price_increment = 100
        price_levels = np.arange(min_price, max_price, price_increment)
        
        bid_volumes = np.zeros(len(price_levels))
        ask_volumes = np.zeros(len(price_levels))
        
        # Aggregate bids
        for price, amount in orderbook['bids']:
            if min_price <= price <= max_price:
                idx = int((price - min_price) / price_increment)
                if 0 <= idx < len(bid_volumes):
                    bid_volumes[idx] += amount
        
        # Aggregate asks
        for price, amount in orderbook['asks']:
            if min_price <= price <= max_price:
                idx = int((price - min_price) / price_increment)
                if 0 <= idx < len(ask_volumes):
                    ask_volumes[idx] += amount
        
        return {
            'price_levels': price_levels.tolist(),
            'bid_volumes': bid_volumes.tolist(),
            'ask_volumes': ask_volumes.tolist()
        }
    
    async def close(self):
        if self.exchange:
            await self.exchange.close()
