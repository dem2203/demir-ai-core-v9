import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import ccxt.async_support as ccxt
from src.config.settings import Config

logger = logging.getLogger("TRADE_FLOW")

class TradeFlowAnalyzer:
    """
    REAL-TIME TRADE FLOW ANALYZER (Gerçek Zamanlı İşlem Akışı Analiz Edilmesi)
    
    Analyzes executed trades (not just orderbook) to detect:
    - Whale market orders (large aggressive buys/sells)
    - Buy/Sell volume imbalance
    - Trade clustering (institutional accumulation)
    - Order flow momentum shifts
    
    Complementary to OrderBookAnalyzer (which tracks passive orders).
    This tracks AGGRESSIVE orders (market orders that execute immediately).
    """
    
    # Thresholds
    WHALE_TRADE_USD = 500_000  # $500K+ single trade = whale
    LARGE_TRADE_USD = 100_000  # $100K+ = significant
    TRADE_LOOKBACK_SECONDS = 300  # Analyze last 5 minutes
    
    def __init__(self):
        self.exchange = None
        self.exchange_config = {
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        }
        self.trade_history_cache = {}  # Cache recent trades
    
    async def connect(self):
        """Initialize exchange connection"""
        try:
            self.exchange = ccxt.binance(self.exchange_config)
            await self.exchange.load_markets()
            logger.info("✅ Trade Flow Analyzer Connected")
        except Exception as e:
            logger.error(f"Trade Flow connection failed: {e}")
            self.exchange = None
    
    async def fetch_recent_trades(self, symbol: str, limit: int = 500) -> Optional[List[Dict]]:
        """
        Fetch recent executed trades from exchange
        
        Returns:
            List of trade dicts: [{'price': 94000, 'amount': 1.5, 'side': 'buy', 'timestamp': ...}, ...]
        """
        if not self.exchange:
            await self.connect()
        if not self.exchange:
            return None
        
        try:
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            # Cache for re-use
            self.trade_history_cache[symbol] = trades
            return trades
        except Exception as e:
            logger.error(f"Failed to fetch trades for {symbol}: {e}")
            return None
    
    def analyze_trade_flow(self, trades: List[Dict], current_price: float) -> Dict:
        """
        Analyze recent trade flow to detect whale activity and momentum
        
        Args:
            trades: List of recent trades
            current_price: Current market price
            
        Returns:
            Comprehensive trade flow analysis
        """
        if not trades or len(trades) == 0:
            return self._empty_analysis()
        
        # Filter to lookback window
        cutoff_time = datetime.now().timestamp() * 1000 - (self.TRADE_LOOKBACK_SECONDS * 1000)
        recent_trades = [t for t in trades if t['timestamp'] >= cutoff_time]
        
        if len(recent_trades) < 10:
            logger.warning("Insufficient recent trades for analysis")
            return self._empty_analysis()
        
        # 1. Detect whale trades
        whale_analysis = self._detect_whale_trades(recent_trades, current_price)
        
        # 2. Calculate buy/sell volume imbalance
        volume_analysis = self._calculate_volume_imbalance(recent_trades)
        
        # 3. Detect trade clustering (accumulation/distribution)
        clustering_analysis = self._detect_trade_clustering(recent_trades, current_price)
        
        # 4. Calculate order flow momentum
        momentum_analysis = self._calculate_flow_momentum(recent_trades)
        
        # 5. Synthesize into actionable signal
        flow_signal = self._synthesize_flow_signal(
            whale_analysis,
            volume_analysis,
            clustering_analysis,
            momentum_analysis
        )
        
        return {
            'whale_trades': whale_analysis,
            'volume_imbalance': volume_analysis,
            'clustering': clustering_analysis,
            'momentum': momentum_analysis,
            'flow_signal': flow_signal,
            'trade_count': len(recent_trades),
            'analysis_window_sec': self.TRADE_LOOKBACK_SECONDS
        }
    
    def _detect_whale_trades(self, trades: List[Dict], current_price: float) -> Dict:
        """Detect large single trades (whale market orders)"""
        whale_buys = []
        whale_sells = []
        large_buys = []
        large_sells = []
        
        for trade in trades:
            price = trade.get('price', 0)
            amount = trade.get('amount', 0)
            side = trade.get('side', 'buy')
            usd_value = price * amount
            
            if usd_value >= self.WHALE_TRADE_USD:
                whale_data = {
                    'price': price,
                    'amount': amount,
                    'usd_value': usd_value,
                    'timestamp': trade.get('timestamp'),
                    'distance_from_current': ((price - current_price) / current_price) * 100
                }
                if side == 'buy':
                    whale_buys.append(whale_data)
                else:
                    whale_sells.append(whale_data)
            
            elif usd_value >= self.LARGE_TRADE_USD:
                if side == 'buy':
                    large_buys.append(usd_value)
                else:
                    large_sells.append(usd_value)
        
        # Summary
        whale_buy_count = len(whale_buys)
        whale_sell_count = len(whale_sells)
        whale_buy_volume = sum(t['usd_value'] for t in whale_buys)
        whale_sell_volume = sum(t['usd_value'] for t in whale_sells)
        
        # Determine dominant whale direction
        if whale_buy_count > whale_sell_count * 1.5:
            whale_bias = 'AGGRESSIVE_BUY'
        elif whale_sell_count > whale_buy_count * 1.5:
            whale_bias = 'AGGRESSIVE_SELL'
        else:
            whale_bias = 'NEUTRAL'
        
        return {
            'whale_buy_count': whale_buy_count,
            'whale_sell_count': whale_sell_count,
            'whale_buy_volume_usd': whale_buy_volume,
            'whale_sell_volume_usd': whale_sell_volume,
            'whale_bias': whale_bias,
            'recent_whales': (whale_buys[:3], whale_sells[:3]),  # Latest 3 each
            'large_trade_count': len(large_buys) + len(large_sells)
        }
    
    def _calculate_volume_imbalance(self, trades: List[Dict]) -> Dict:
        """Calculate buy vs sell volume imbalance"""
        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        
        for trade in trades:
            amount = trade.get('amount', 0)
            price = trade.get('price', 0)
            usd_vol = amount * price
            
            if trade.get('side') == 'buy':
                buy_volume += usd_vol
                buy_count += 1
            else:
                sell_volume += usd_vol
                sell_count += 1
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return {'imbalance': 0, 'signal': 'NEUTRAL'}
        
        # Calculate imbalance (-1 to +1)
        imbalance = (buy_volume - sell_volume) / total_volume
        
        # Volume-weighted average price by side
        vwap_buy = buy_volume / buy_count if buy_count > 0 else 0
        vwap_sell = sell_volume / sell_count if sell_count > 0 else 0
        
        # Signal strength
        if imbalance > 0.3:
            signal = 'STRONG_BUY_PRESSURE'
        elif imbalance > 0.1:
            signal = 'BUY_PRESSURE'
        elif imbalance < -0.3:
            signal = 'STRONG_SELL_PRESSURE'
        elif imbalance < -0.1:
            signal = 'SELL_PRESSURE'
        else:
            signal = 'BALANCED'
        
        return {
            'imbalance': imbalance,
            'signal': signal,
            'buy_volume_usd': buy_volume,
            'sell_volume_usd': sell_volume,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'total_volume_usd': total_volume,
            'vwap_buy': vwap_buy,
            'vwap_sell': vwap_sell
        }
    
    def _detect_trade_clustering(self, trades: List[Dict], current_price: float) -> Dict:
        """
        Detect if trades are clustering at specific price levels
        (sign of institutional accumulation/distribution)
        """
        # Group trades by price buckets ($100 increments for BTC)
        price_buckets = {}
        bucket_size = 100
        
        for trade in trades:
            price = trade.get('price', 0)
            bucket = int(price / bucket_size) * bucket_size
            
            if bucket not in price_buckets:
                price_buckets[bucket] = {'buy_vol': 0, 'sell_vol': 0, 'count': 0}
            
            amount = trade.get('amount', 0)
            if trade.get('side') == 'buy':
                price_buckets[bucket]['buy_vol'] += amount
            else:
                price_buckets[bucket]['sell_vol'] += amount
            price_buckets[bucket]['count'] += 1
        
        # Find clusters (price levels with disproportionate activity)
        clusters = []
        avg_count = np.mean([b['count'] for b in price_buckets.values()])
        
        for price, data in price_buckets.items():
            if data['count'] > avg_count * 2:  # 2x average = cluster
                clusters.append({
                    'price': price,
                    'trade_count': data['count'],
                    'buy_vol': data['buy_vol'],
                    'sell_vol': data['sell_vol'],
                    'net_vol': data['buy_vol'] - data['sell_vol'],
                    'distance_from_current': ((price - current_price) / current_price) * 100
                })
        
        # Sort by trade count
        clusters.sort(key=lambda x: x['trade_count'], reverse=True)
        
        # Identify if clustering is bullish or bearish
        if clusters:
            top_cluster = clusters[0]
            if top_cluster['net_vol'] > 0:
                cluster_bias = 'ACCUMULATION'
            elif top_cluster['net_vol'] < 0:
                cluster_bias = 'DISTRIBUTION'
            else:
                cluster_bias = 'NEUTRAL'
        else:
            cluster_bias = 'NO_CLUSTERS'
        
        return {
            'cluster_count': len(clusters),
            'top_clusters': clusters[:3],  # Top 3
            'cluster_bias': cluster_bias
        }
    
    def _calculate_flow_momentum(self, trades: List[Dict]) -> Dict:
        """
        Calculate if order flow momentum is accelerating or decelerating
        """
        if len(trades) < 20:
            return {'momentum': 'INSUFFICIENT_DATA'}
        
        # Split into 2 halves (early vs recent)
        mid_point = len(trades) // 2
        early_trades = trades[:mid_point]
        recent_trades = trades[mid_point:]
        
        # Calculate volume for each half
        early_vol = sum(t.get('amount', 0) * t.get('price', 0) for t in early_trades)
        recent_vol = sum(t.get('amount', 0) * t.get('price', 0) for t in recent_trades)
        
        # Momentum = (recent - early) / early
        if early_vol > 0:
            momentum_pct = ((recent_vol - early_vol) / early_vol) * 100
        else:
            momentum_pct = 0
        
        # Classify
        if momentum_pct > 50:
            momentum = 'ACCELERATING_STRONG'
        elif momentum_pct > 20:
            momentum = 'ACCELERATING'
        elif momentum_pct < -50:
            momentum = 'DECELERATING_STRONG'
        elif momentum_pct < -20:
            momentum = 'DECELERATING'
        else:
            momentum = 'STABLE'
        
        return {
            'momentum': momentum,
            'momentum_pct': momentum_pct,
            'early_volume_usd': early_vol,
            'recent_volume_usd': recent_vol
        }
    
    def _synthesize_flow_signal(
        self,
        whale_analysis: Dict,
        volume_analysis: Dict,
        clustering_analysis: Dict,
        momentum_analysis: Dict
    ) -> Dict:
        """
        Combine all analyses into final order flow signal
        """
        score = 0
        factors = []
        
        # Whale bias
        if whale_analysis['whale_bias'] == 'AGGRESSIVE_BUY':
            score += 30
            factors.append("Whale aggressive buying")
        elif whale_analysis['whale_bias'] == 'AGGRESSIVE_SELL':
            score -= 30
            factors.append("Whale aggressive selling")
        
        # Volume imbalance
        vol_signal = volume_analysis['signal']
        if 'STRONG_BUY' in vol_signal:
            score += 25
            factors.append("Strong buy volume dominance")
        elif 'BUY' in vol_signal:
            score += 15
            factors.append("Buy volume pressure")
        elif 'STRONG_SELL' in vol_signal:
            score -= 25
            factors.append("Strong sell volume dominance")
        elif 'SELL' in vol_signal:
            score -= 15
            factors.append("Sell volume pressure")
        
        # Clustering
        if clustering_analysis['cluster_bias'] == 'ACCUMULATION':
            score += 20
            factors.append("Accumulation clusters detected")
        elif clustering_analysis['cluster_bias'] == 'DISTRIBUTION':
            score -= 20
            factors.append("Distribution clusters detected")
        
        # Momentum
        momentum = momentum_analysis.get('momentum', 'STABLE')
        if 'ACCELERATING' in momentum:
            score += 10
            factors.append(f"Flow momentum {momentum.lower()}")
        elif 'DECELERATING' in momentum:
            score -= 10
            factors.append(f"Flow momentum {momentum.lower()}")
        
        # Final signal
        if score >= 50:
            signal = 'STRONG_BUY'
            confidence = min(score, 100)
        elif score >= 25:
            signal = 'BUY'
            confidence = min(score + 20, 80)
        elif score <= -50:
            signal = 'STRONG_SELL'
            confidence = min(abs(score), 100)
        elif score <= -25:
            signal = 'SELL'
            confidence = min(abs(score) + 20, 80)
        else:
            signal = 'NEUTRAL'
            confidence = 50 - abs(score)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'score': score,
            'factors': factors
        }
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'whale_trades': {'whale_bias': 'NEUTRAL'},
            'volume_imbalance': {'signal': 'NEUTRAL'},
            'clustering': {'cluster_bias': 'NO_CLUSTERS'},
            'momentum': {'momentum': 'INSUFFICIENT_DATA'},
            'flow_signal': {'signal': 'NEUTRAL', 'confidence': 0, 'factors': []},
            'trade_count': 0
        }
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
