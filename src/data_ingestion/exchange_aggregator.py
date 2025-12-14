"""
Exchange Aggregator - Multi-Exchange Data Integration
Phase 29.3

Aggregates data from Binance, Bybit, and Coinbase for:
- Cross-exchange price comparison
- Price divergence detection
- Best price discovery
- Arbitrage opportunity alerts
"""

import asyncio
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("EXCHANGE_AGGREGATOR")


@dataclass
class ExchangePrice:
    """Price data from an exchange"""
    exchange: str
    price: float
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    volume_24h: float = 0.0


class ExchangeAggregator:
    """
    Aggregates market data from multiple exchanges.
    
    Provides:
    - Cross-exchange price comparison
    - Price divergence detection
    - Best bid/ask discovery
    """
    
    DIVERGENCE_THRESHOLD_PCT = 0.15  # 0.15% price difference is significant
    
    def __init__(self):
        self.binance = None
        self.bybit = None
        self.coinbase = None
        self._initialized = False
        self._price_cache: Dict[str, List[ExchangePrice]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration_seconds = 5  # Cache for 5 seconds
    
    async def initialize(self):
        """Initialize all exchange connectors"""
        if self._initialized:
            return
        
        try:
            from src.data_ingestion.connectors.binance_connector import BinanceConnector
            from src.data_ingestion.connectors.bybit_connector import BybitConnector
            from src.data_ingestion.connectors.coinbase_connector import CoinbaseConnector
            
            self.binance = BinanceConnector()
            self.bybit = BybitConnector()
            self.coinbase = CoinbaseConnector()
            
            # Connect all
            await asyncio.gather(
                self.binance.connect(),
                self.bybit.connect(),
                self.coinbase.connect(),
                return_exceptions=True
            )
            
            self._initialized = True
            logger.info("✅ ExchangeAggregator initialized with all exchanges")
        except Exception as e:
            logger.error(f"ExchangeAggregator initialization error: {e}")
    
    async def get_cross_exchange_prices(self, symbol: str) -> Dict:
        """
        Get current price from all exchanges.
        
        Returns:
            {
                'binance': {'price': 97450.0, 'deviation': 0.0},
                'bybit': {'price': 97480.0, 'deviation': 0.03},
                'coinbase': {'price': 97420.0, 'deviation': -0.03},
                'reference_price': 97450.0,
                'max_divergence': 0.06,
                'timestamp': '2025-12-14 21:00:00'
            }
        """
        if not self._initialized:
            await self.initialize()
        
        prices = {}
        reference_price = 0.0
        
        # Fetch from Binance (primary reference)
        try:
            if self.binance and self.binance.exchange:
                ticker = await self.binance.exchange.fetch_ticker(symbol)
                reference_price = float(ticker['last'])
                prices['binance'] = {
                    'price': reference_price,
                    'deviation': 0.0,
                    'bid': float(ticker.get('bid', 0) or 0),
                    'ask': float(ticker.get('ask', 0) or 0),
                    'volume_24h': float(ticker.get('quoteVolume', 0) or 0)
                }
        except Exception as e:
            logger.warning(f"Binance price fetch failed: {e}")
        
        # Fetch from Bybit
        try:
            if self.bybit and self.bybit.exchange:
                # Bybit uses different symbol format for futures
                bybit_symbol = symbol
                ticker = await self.bybit.exchange.fetch_ticker(bybit_symbol)
                bybit_price = float(ticker['last'])
                deviation = ((bybit_price - reference_price) / reference_price * 100) if reference_price > 0 else 0
                prices['bybit'] = {
                    'price': bybit_price,
                    'deviation': round(deviation, 3),
                    'bid': float(ticker.get('bid', 0) or 0),
                    'ask': float(ticker.get('ask', 0) or 0),
                    'funding_rate': 0.0  # Will be fetched separately if needed
                }
        except Exception as e:
            logger.warning(f"Bybit price fetch failed: {e}")
        
        # Fetch from Coinbase
        try:
            if self.coinbase and self.coinbase.exchange:
                # Coinbase uses BTC-USD format sometimes
                coinbase_symbol = symbol.replace('/USDT', '/USD')
                ticker = await self.coinbase.exchange.fetch_ticker(coinbase_symbol)
                coinbase_price = float(ticker['last'])
                deviation = ((coinbase_price - reference_price) / reference_price * 100) if reference_price > 0 else 0
                prices['coinbase'] = {
                    'price': coinbase_price,
                    'deviation': round(deviation, 3),
                    'bid': float(ticker.get('bid', 0) or 0),
                    'ask': float(ticker.get('ask', 0) or 0)
                }
        except Exception as e:
            logger.warning(f"Coinbase price fetch failed: {e}")
        
        # Calculate max divergence
        all_prices = [p['price'] for p in prices.values() if p.get('price', 0) > 0]
        max_divergence = 0.0
        if len(all_prices) >= 2:
            max_divergence = (max(all_prices) - min(all_prices)) / min(all_prices) * 100
        
        return {
            **prices,
            'reference_price': reference_price,
            'max_divergence': round(max_divergence, 3),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'exchanges_online': len(prices)
        }
    
    async def detect_price_divergence(self, symbol: str) -> Optional[Dict]:
        """
        Detect significant price divergence between exchanges.
        
        Returns alert if divergence > threshold.
        """
        prices = await self.get_cross_exchange_prices(symbol)
        
        if prices['max_divergence'] >= self.DIVERGENCE_THRESHOLD_PCT:
            # Find best buy and sell exchanges
            exchange_prices = []
            for ex in ['binance', 'bybit', 'coinbase']:
                if ex in prices and prices[ex].get('price', 0) > 0:
                    exchange_prices.append((ex, prices[ex]['price']))
            
            if len(exchange_prices) >= 2:
                exchange_prices.sort(key=lambda x: x[1])
                best_buy = exchange_prices[0]  # Lowest price
                best_sell = exchange_prices[-1]  # Highest price
                
                logger.warning(f"⚡ PRICE DIVERGENCE DETECTED: {symbol} - {prices['max_divergence']:.2f}%")
                logger.info(f"   Buy on {best_buy[0]} @ ${best_buy[1]:,.2f}")
                logger.info(f"   Sell on {best_sell[0]} @ ${best_sell[1]:,.2f}")
                
                return {
                    'symbol': symbol,
                    'divergence_pct': prices['max_divergence'],
                    'best_buy_exchange': best_buy[0],
                    'best_buy_price': best_buy[1],
                    'best_sell_exchange': best_sell[0],
                    'best_sell_price': best_sell[1],
                    'potential_profit_pct': prices['max_divergence'] - 0.1,  # Minus fees estimate
                    'timestamp': prices['timestamp']
                }
        
        return None
    
    async def get_aggregated_orderbook_imbalance(self, symbol: str) -> Dict:
        """
        Get buy/sell pressure from multiple exchanges.
        """
        total_bid_volume = 0.0
        total_ask_volume = 0.0
        exchange_imbalances = {}
        
        # Binance orderbook
        try:
            if self.binance and self.binance.exchange:
                ob = await self.binance.exchange.fetch_order_book(symbol, limit=20)
                bid_vol = sum([p * a for p, a in ob['bids'][:20]])
                ask_vol = sum([p * a for p, a in ob['asks'][:20]])
                total_bid_volume += bid_vol
                total_ask_volume += ask_vol
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
                exchange_imbalances['binance'] = round(imbalance * 100, 2)
        except Exception as e:
            logger.debug(f"Binance orderbook error: {e}")
        
        # Bybit orderbook
        try:
            if self.bybit and self.bybit.exchange:
                ob = await self.bybit.exchange.fetch_order_book(symbol, limit=20)
                bid_vol = sum([p * a for p, a in ob['bids'][:20]])
                ask_vol = sum([p * a for p, a in ob['asks'][:20]])
                total_bid_volume += bid_vol
                total_ask_volume += ask_vol
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
                exchange_imbalances['bybit'] = round(imbalance * 100, 2)
        except Exception as e:
            logger.debug(f"Bybit orderbook error: {e}")
        
        # Overall imbalance
        total_imbalance = 0.0
        if total_bid_volume + total_ask_volume > 0:
            total_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) * 100
        
        bias = "NEUTRAL"
        if total_imbalance > 10:
            bias = "BULLISH"
        elif total_imbalance < -10:
            bias = "BEARISH"
        
        return {
            'total_bid_volume_usd': total_bid_volume,
            'total_ask_volume_usd': total_ask_volume,
            'total_imbalance_pct': round(total_imbalance, 2),
            'exchange_imbalances': exchange_imbalances,
            'bias': bias
        }
    
    def format_for_telegram(self, cross_exchange_data: Dict) -> str:
        """
        Format cross-exchange data for Telegram message.
        """
        lines = ["📈 **CROSS-EXCHANGE:**"]
        
        for ex in ['binance', 'bybit', 'coinbase']:
            if ex in cross_exchange_data:
                data = cross_exchange_data[ex]
                price = data.get('price', 0)
                deviation = data.get('deviation', 0)
                
                if price > 0:
                    dev_str = f"+{deviation:.2f}%" if deviation > 0 else f"{deviation:.2f}%"
                    dev_emoji = "🟢" if deviation > 0 else "🔴" if deviation < 0 else "⚪"
                    ex_name = ex.capitalize()
                    lines.append(f"   {dev_emoji} {ex_name}: ${price:,.2f} ({dev_str})")
        
        if cross_exchange_data.get('max_divergence', 0) >= self.DIVERGENCE_THRESHOLD_PCT:
            lines.append(f"   ⚡ **Divergence Alert: {cross_exchange_data['max_divergence']:.2f}%**")
        
        return "\n".join(lines)
    
    async def close(self):
        """Close all exchange connections"""
        close_tasks = []
        if self.binance:
            close_tasks.append(self.binance.close())
        if self.bybit:
            close_tasks.append(self.bybit.close())
        if self.coinbase:
            close_tasks.append(self.coinbase.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self._initialized = False
        logger.info("ExchangeAggregator closed")


# Singleton instance
_aggregator_instance: Optional[ExchangeAggregator] = None


async def get_exchange_aggregator() -> ExchangeAggregator:
    """Get or create the singleton ExchangeAggregator instance"""
    global _aggregator_instance
    if _aggregator_instance is None:
        _aggregator_instance = ExchangeAggregator()
        await _aggregator_instance.initialize()
    return _aggregator_instance


# Quick test
if __name__ == "__main__":
    async def test():
        aggregator = await get_exchange_aggregator()
        
        print("Testing Cross-Exchange Prices...")
        prices = await aggregator.get_cross_exchange_prices("BTC/USDT")
        print(f"Prices: {prices}")
        
        print("\nTelegram Format:")
        print(aggregator.format_for_telegram(prices))
        
        print("\nTesting Orderbook Imbalance...")
        imbalance = await aggregator.get_aggregated_orderbook_imbalance("BTC/USDT")
        print(f"Imbalance: {imbalance}")
        
        await aggregator.close()
    
    asyncio.run(test())
