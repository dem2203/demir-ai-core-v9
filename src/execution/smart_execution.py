# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - SMART EXECUTION ENGINE
=====================================
Professional order execution with slippage analysis.

FEATURES:
1. Slippage Estimation - Calculate expected slippage before trade
2. TWAP/VWAP Algorithms - Split large orders across time/volume
3. Execution Quality Metrics - Track actual vs expected fills
4. Smart Order Routing - Best price across orderbook levels

USAGE:
    executor = SmartExecutor()
    estimate = executor.estimate_slippage(symbol, size_usd, side)
    execution_plan = executor.create_twap_plan(symbol, size_usd, duration_mins)
"""
import logging
import aiohttp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger("SMART_EXECUTION")


class ExecutionAlgorithm(Enum):
    MARKET = "MARKET"          # Immediate market order
    TWAP = "TWAP"              # Time-Weighted Average Price
    VWAP = "VWAP"              # Volume-Weighted Average Price
    ICEBERG = "ICEBERG"        # Hidden size orders


@dataclass
class SlippageEstimate:
    """Estimated slippage for a given order"""
    symbol: str
    side: str                   # BUY or SELL
    size_usd: float
    estimated_slippage_pct: float
    estimated_slippage_usd: float
    orderbook_depth_usd: float  # Available liquidity
    spread_pct: float           # Current bid-ask spread
    recommendation: str         # Market, TWAP, or VWAP


@dataclass
class ExecutionPlan:
    """Plan for executing a large order"""
    symbol: str
    total_size_usd: float
    algorithm: ExecutionAlgorithm
    num_slices: int
    slice_size_usd: float
    interval_seconds: int
    expected_duration_mins: float
    expected_slippage_pct: float


@dataclass
class ExecutionResult:
    """Result of an execution"""
    symbol: str
    planned_price: float
    actual_avg_price: float
    slippage_pct: float
    slippage_usd: float
    fill_rate: float            # % of order filled
    execution_time_seconds: float


class SmartExecutor:
    """
    Professional execution engine for minimizing slippage.
    """
    
    FUTURES_BASE = "https://fapi.binance.com"
    
    # Thresholds for algorithm selection
    MARKET_ORDER_MAX_SIZE = 5000     # $5K or less = market order OK
    TWAP_THRESHOLD = 50000           # $50K+ = use TWAP
    ICEBERG_THRESHOLD = 100000       # $100K+ = use iceberg
    
    # Default slippage assumptions
    DEFAULT_SPREAD_BPS = 2           # 0.02% typical spread
    DEPTH_IMPACT_FACTOR = 0.1        # 0.1% per $100K notional
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._execution_history: List[ExecutionResult] = []
        
        logger.info("⚡ Smart Execution Engine initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def estimate_slippage(
        self,
        symbol: str,
        size_usd: float,
        side: str  # BUY or SELL
    ) -> SlippageEstimate:
        """
        Estimate slippage for a given order before execution.
        
        Uses orderbook depth analysis to predict price impact.
        """
        try:
            # Get orderbook
            orderbook = await self._get_orderbook(symbol)
            if not orderbook:
                return self._default_estimate(symbol, size_usd, side)
            
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return self._default_estimate(symbol, size_usd, side)
            
            # Calculate spread
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid_price * 100
            
            # Calculate depth-based slippage
            if side == "BUY":
                slippage = self._calculate_depth_impact(asks, size_usd, mid_price)
            else:
                slippage = self._calculate_depth_impact(bids, size_usd, mid_price)
            
            # Add spread cost for market orders
            total_slippage = slippage + (spread_pct / 2)
            
            # Calculate orderbook depth (first 10 levels)
            depth_usd = sum(float(level[0]) * float(level[1]) for level in (asks if side == "BUY" else bids)[:10])
            
            # Recommendation
            if size_usd <= self.MARKET_ORDER_MAX_SIZE:
                recommendation = "MARKET - Small size, slippage acceptable"
            elif size_usd <= self.TWAP_THRESHOLD:
                recommendation = "LIMIT - Use limit orders with patience"
            else:
                recommendation = "TWAP - Split order over 5-15 minutes"
            
            return SlippageEstimate(
                symbol=symbol,
                side=side,
                size_usd=size_usd,
                estimated_slippage_pct=total_slippage,
                estimated_slippage_usd=size_usd * total_slippage / 100,
                orderbook_depth_usd=depth_usd,
                spread_pct=spread_pct,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Slippage estimation error: {e}")
            return self._default_estimate(symbol, size_usd, side)
    
    def _calculate_depth_impact(
        self,
        levels: List,
        size_usd: float,
        mid_price: float
    ) -> float:
        """Calculate price impact from walking through orderbook levels."""
        remaining = size_usd
        weighted_price_sum = 0
        total_qty = 0
        
        for level in levels:
            price = float(level[0])
            qty = float(level[1])
            level_usd = price * qty
            
            if remaining <= 0:
                break
            
            if level_usd >= remaining:
                # Partial fill at this level
                fill_qty = remaining / price
                weighted_price_sum += price * fill_qty
                total_qty += fill_qty
                remaining = 0
            else:
                # Full level consumed
                weighted_price_sum += price * qty
                total_qty += qty
                remaining -= level_usd
        
        if total_qty == 0:
            return self.DEPTH_IMPACT_FACTOR * size_usd / 100000
        
        avg_fill_price = weighted_price_sum / total_qty
        slippage_pct = abs(avg_fill_price - mid_price) / mid_price * 100
        
        return slippage_pct
    
    def _default_estimate(self, symbol: str, size_usd: float, side: str) -> SlippageEstimate:
        """Default slippage estimate when orderbook unavailable."""
        # Simple model: 0.02% base + 0.01% per $10K
        slippage = 0.02 + (size_usd / 10000) * 0.01
        
        return SlippageEstimate(
            symbol=symbol,
            side=side,
            size_usd=size_usd,
            estimated_slippage_pct=slippage,
            estimated_slippage_usd=size_usd * slippage / 100,
            orderbook_depth_usd=0,
            spread_pct=0.02,
            recommendation="LIMIT - Orderbook unavailable, use limit orders"
        )
    
    def create_twap_plan(
        self,
        symbol: str,
        size_usd: float,
        duration_mins: int = 10,
        target_slippage_pct: float = 0.1
    ) -> ExecutionPlan:
        """
        Create a TWAP execution plan for large orders.
        
        TWAP splits the order into equal time-based slices.
        """
        # Calculate optimal number of slices (aim for $5K-$10K per slice)
        optimal_slice = min(max(size_usd / 10, 5000), 10000)
        num_slices = max(2, int(size_usd / optimal_slice))
        
        slice_size = size_usd / num_slices
        interval_seconds = (duration_mins * 60) / num_slices
        
        # Estimate slippage reduction from TWAP
        # Typically reduces slippage by 50-70% compared to single market order
        market_slippage = 0.02 + (size_usd / 10000) * 0.01
        twap_slippage = market_slippage * 0.4  # 60% reduction
        
        return ExecutionPlan(
            symbol=symbol,
            total_size_usd=size_usd,
            algorithm=ExecutionAlgorithm.TWAP,
            num_slices=num_slices,
            slice_size_usd=slice_size,
            interval_seconds=int(interval_seconds),
            expected_duration_mins=duration_mins,
            expected_slippage_pct=twap_slippage
        )
    
    def create_vwap_plan(
        self,
        symbol: str,
        size_usd: float,
        volume_profile: List[float] = None
    ) -> ExecutionPlan:
        """
        Create a VWAP execution plan.
        
        VWAP distributes order based on volume profile.
        Typically uses hourly volume distribution.
        """
        # Default volume profile (typical crypto pattern)
        if volume_profile is None:
            # Higher volume at market open/close, lower during off-hours
            volume_profile = [0.15, 0.12, 0.10, 0.08, 0.08, 0.10, 0.12, 0.15, 0.10]
        
        num_slices = len(volume_profile)
        
        # Normalize profile
        total_weight = sum(volume_profile)
        normalized = [v / total_weight for v in volume_profile]
        
        # Average slice for estimation
        avg_slice = size_usd / num_slices
        
        # VWAP typically performs slightly better than TWAP
        market_slippage = 0.02 + (size_usd / 10000) * 0.01
        vwap_slippage = market_slippage * 0.35  # 65% reduction
        
        return ExecutionPlan(
            symbol=symbol,
            total_size_usd=size_usd,
            algorithm=ExecutionAlgorithm.VWAP,
            num_slices=num_slices,
            slice_size_usd=avg_slice,
            interval_seconds=600,  # 10 mins between slices
            expected_duration_mins=num_slices * 10,
            expected_slippage_pct=vwap_slippage
        )
    
    async def _get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Fetch orderbook from Binance Futures."""
        try:
            session = await self._get_session()
            url = f"{self.FUTURES_BASE}/fapi/v1/depth"
            params = {'symbol': symbol, 'limit': limit}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error(f"Orderbook fetch error: {e}")
            return None
    
    def record_execution(self, result: ExecutionResult):
        """Record execution for quality analysis."""
        self._execution_history.append(result)
        
        # Keep last 100 executions
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]
        
        logger.info(
            f"📊 Execution recorded: {result.symbol} | "
            f"Slippage: {result.slippage_pct:.3f}% ({result.slippage_usd:+.2f} USD)"
        )
    
    def get_execution_quality(self) -> Dict:
        """Get aggregate execution quality metrics."""
        if not self._execution_history:
            return {
                'total_executions': 0,
                'avg_slippage_pct': 0,
                'avg_slippage_usd': 0,
                'avg_fill_rate': 0
            }
        
        return {
            'total_executions': len(self._execution_history),
            'avg_slippage_pct': sum(e.slippage_pct for e in self._execution_history) / len(self._execution_history),
            'avg_slippage_usd': sum(e.slippage_usd for e in self._execution_history) / len(self._execution_history),
            'avg_fill_rate': sum(e.fill_rate for e in self._execution_history) / len(self._execution_history)
        }
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton
_executor: Optional[SmartExecutor] = None

def get_smart_executor() -> SmartExecutor:
    global _executor
    if _executor is None:
        _executor = SmartExecutor()
    return _executor
