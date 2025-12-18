"""
Smart Money Concepts (SMC) Analyzer

Institutional trading patterns detection:
1. Order Blocks - Last opposing candle before strong move
2. Fair Value Gaps (FVG) - Price imbalances
3. Liquidity Zones - Equal highs/lows where stops accumulate

ALL DATA FROM BINANCE OHLCV - NO MOCKS!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("SMC_ANALYZER")


@dataclass
class OrderBlock:
    """Represents an Order Block zone"""
    type: str  # 'BULLISH' or 'BEARISH'
    top: float
    bottom: float
    strength: int  # 0-100
    touched: bool
    broken: bool
    timestamp: str


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap"""
    type: str  # 'BULLISH' or 'BEARISH'
    top: float
    bottom: float
    filled_pct: float  # 0-100
    timestamp: str


@dataclass
class LiquidityZone:
    """Represents a Liquidity accumulation zone"""
    type: str  # 'BUY_STOPS' (above highs) or 'SELL_STOPS' (below lows)
    price: float
    strength: int  # Number of touches
    swept: bool


class SMCAnalyzer:
    """
    Smart Money Concepts Analyzer
    
    Detects institutional trading patterns from OHLCV data only.
    Zero external dependencies - all calculated from price action.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.order_blocks: List[OrderBlock] = []
        self.fvgs: List[FairValueGap] = []
        self.liquidity_zones: List[LiquidityZone] = []
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Full SMC analysis on OHLCV dataframe.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            Dict with order_blocks, fvgs, liquidity_zones, and trading signals
        """
        if df is None or df.empty or len(df) < 20:
            logger.warning("Insufficient data for SMC analysis")
            return self._empty_result()
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            logger.error(f"Missing columns. Required: {required}")
            return self._empty_result()
        
        # Run all detections
        self.order_blocks = self._detect_order_blocks(df)
        self.fvgs = self._detect_fvg(df)
        self.liquidity_zones = self._detect_liquidity_zones(df)
        
        # Get current price context
        current_price = df['close'].iloc[-1]
        
        # Find nearest zones
        nearest_ob = self._find_nearest_order_block(current_price)
        nearest_fvg = self._find_nearest_fvg(current_price)
        nearest_liq = self._find_nearest_liquidity(current_price)
        
        # Generate SMC signal
        smc_signal = self._generate_smc_signal(current_price, nearest_ob, nearest_fvg, nearest_liq)
        
        return {
            'order_blocks': [self._ob_to_dict(ob) for ob in self.order_blocks[-5:]],  # Last 5
            'fvgs': [self._fvg_to_dict(fvg) for fvg in self.fvgs[-5:]],
            'liquidity_zones': [self._liq_to_dict(lz) for lz in self.liquidity_zones[-5:]],
            'nearest_order_block': self._ob_to_dict(nearest_ob) if nearest_ob else None,
            'nearest_fvg': self._fvg_to_dict(nearest_fvg) if nearest_fvg else None,
            'nearest_liquidity': self._liq_to_dict(nearest_liq) if nearest_liq else None,
            'smc_signal': smc_signal,
            'smc_bias': self._calculate_smc_bias(),
            'current_price': current_price
        }
    
    def _detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect Order Blocks - last opposing candle before strong impulsive move.
        
        Bullish OB: Last bearish candle before strong bullish move
        Bearish OB: Last bullish candle before strong bearish move
        """
        order_blocks = []
        
        # Calculate ATR for move strength measurement
        atr = self._calculate_atr(df, 14)
        avg_atr = atr.mean() if len(atr) > 0 else 0
        
        for i in range(3, len(df) - 1):
            # Current and previous candles
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # Candle body sizes
            curr_body = abs(curr['close'] - curr['open'])
            prev_body = abs(prev['close'] - prev['open'])
            
            # Strong move threshold (1.5x ATR)
            strong_move = avg_atr * 1.5 if avg_atr > 0 else curr_body * 2
            
            # Bullish Order Block
            # Previous candle is bearish, current candle is strongly bullish
            if (prev['close'] < prev['open'] and  # Prev is bearish
                curr['close'] > curr['open'] and   # Current is bullish
                curr_body > strong_move):          # Strong move
                
                ob = OrderBlock(
                    type='BULLISH',
                    top=prev['open'],
                    bottom=prev['low'],
                    strength=min(100, int((curr_body / avg_atr) * 30)) if avg_atr > 0 else 70,
                    touched=False,
                    broken=False,
                    timestamp=str(df.index[i-1]) if hasattr(df.index, '__iter__') else str(i-1)
                )
                order_blocks.append(ob)
            
            # Bearish Order Block
            # Previous candle is bullish, current candle is strongly bearish
            if (prev['close'] > prev['open'] and  # Prev is bullish
                curr['close'] < curr['open'] and   # Current is bearish
                curr_body > strong_move):          # Strong move
                
                ob = OrderBlock(
                    type='BEARISH',
                    top=prev['high'],
                    bottom=prev['open'],
                    strength=min(100, int((curr_body / avg_atr) * 30)) if avg_atr > 0 else 70,
                    touched=False,
                    broken=False,
                    timestamp=str(df.index[i-1]) if hasattr(df.index, '__iter__') else str(i-1)
                )
                order_blocks.append(ob)
        
        # Update touched/broken status
        current_price = df['close'].iloc[-1]
        for ob in order_blocks:
            if ob.type == 'BULLISH':
                if current_price <= ob.top and current_price >= ob.bottom:
                    ob.touched = True
                if current_price < ob.bottom:
                    ob.broken = True
            else:  # BEARISH
                if current_price >= ob.bottom and current_price <= ob.top:
                    ob.touched = True
                if current_price > ob.top:
                    ob.broken = True
        
        # Filter out broken OBs, keep only valid ones
        valid_obs = [ob for ob in order_blocks if not ob.broken]
        
        return valid_obs[-10:]  # Keep last 10 valid OBs
    
    def _detect_fvg(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps (Imbalances).
        
        Bullish FVG: Gap between candle[i-1].high and candle[i+1].low
        Bearish FVG: Gap between candle[i-1].low and candle[i+1].high
        """
        fvgs = []
        
        for i in range(1, len(df) - 1):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            next_c = df.iloc[i+1]
            
            # Bullish FVG: gap up
            if next_c['low'] > prev['high']:
                gap_size = next_c['low'] - prev['high']
                # Only significant gaps (> 0.1% of price)
                if gap_size / curr['close'] > 0.001:
                    fvg = FairValueGap(
                        type='BULLISH',
                        top=next_c['low'],
                        bottom=prev['high'],
                        filled_pct=0.0,
                        timestamp=str(df.index[i]) if hasattr(df.index, '__iter__') else str(i)
                    )
                    fvgs.append(fvg)
            
            # Bearish FVG: gap down
            if next_c['high'] < prev['low']:
                gap_size = prev['low'] - next_c['high']
                if gap_size / curr['close'] > 0.001:
                    fvg = FairValueGap(
                        type='BEARISH',
                        top=prev['low'],
                        bottom=next_c['high'],
                        filled_pct=0.0,
                        timestamp=str(df.index[i]) if hasattr(df.index, '__iter__') else str(i)
                    )
                    fvgs.append(fvg)
        
        # Calculate fill percentage for each FVG
        current_price = df['close'].iloc[-1]
        for fvg in fvgs:
            gap_size = fvg.top - fvg.bottom
            if gap_size > 0:
                if fvg.type == 'BULLISH':
                    # How much price has retraced into the gap
                    if current_price < fvg.top:
                        fill = max(0, fvg.top - current_price)
                        fvg.filled_pct = min(100, (fill / gap_size) * 100)
                else:  # BEARISH
                    if current_price > fvg.bottom:
                        fill = max(0, current_price - fvg.bottom)
                        fvg.filled_pct = min(100, (fill / gap_size) * 100)
        
        # Filter out fully filled FVGs
        valid_fvgs = [fvg for fvg in fvgs if fvg.filled_pct < 100]
        
        return valid_fvgs[-10:]
    
    def _detect_liquidity_zones(self, df: pd.DataFrame) -> List[LiquidityZone]:
        """
        Detect Liquidity Zones - areas where stop losses accumulate.
        
        Equal Highs = Buy stops above (liquidity)
        Equal Lows = Sell stops below (liquidity)
        """
        liquidity_zones = []
        tolerance = 0.002  # 0.2% tolerance for "equal" levels
        
        highs = df['high'].values
        lows = df['low'].values
        current_price = df['close'].iloc[-1]
        
        # Find equal highs (potential buy stop liquidity)
        high_clusters = self._find_price_clusters(highs, tolerance)
        for price, count in high_clusters.items():
            if count >= 2 and price > current_price:  # At least 2 touches, above current price
                lz = LiquidityZone(
                    type='BUY_STOPS',
                    price=price,
                    strength=min(count, 5),  # Cap at 5
                    swept=False
                )
                liquidity_zones.append(lz)
        
        # Find equal lows (potential sell stop liquidity)
        low_clusters = self._find_price_clusters(lows, tolerance)
        for price, count in low_clusters.items():
            if count >= 2 and price < current_price:  # At least 2 touches, below current price
                lz = LiquidityZone(
                    type='SELL_STOPS',
                    price=price,
                    strength=min(count, 5),
                    swept=False
                )
                liquidity_zones.append(lz)
        
        # Check if any zones have been swept (price went through)
        all_time_high = df['high'].max()
        all_time_low = df['low'].min()
        
        for lz in liquidity_zones:
            if lz.type == 'BUY_STOPS' and all_time_high > lz.price:
                lz.swept = True
            elif lz.type == 'SELL_STOPS' and all_time_low < lz.price:
                lz.swept = True
        
        # Filter out swept zones
        valid_zones = [lz for lz in liquidity_zones if not lz.swept]
        
        return valid_zones[:10]  # Top 10 by proximity
    
    def _find_price_clusters(self, prices: np.ndarray, tolerance: float) -> Dict[float, int]:
        """Group prices that are within tolerance of each other"""
        clusters = {}
        
        for price in prices:
            found = False
            for cluster_price in list(clusters.keys()):
                if abs(price - cluster_price) / cluster_price < tolerance:
                    clusters[cluster_price] += 1
                    found = True
                    break
            if not found:
                clusters[price] = 1
        
        return clusters
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _find_nearest_order_block(self, price: float) -> Optional[OrderBlock]:
        """Find the nearest valid order block to current price"""
        if not self.order_blocks:
            return None
        
        nearest = None
        min_distance = float('inf')
        
        for ob in self.order_blocks:
            if not ob.broken:
                mid = (ob.top + ob.bottom) / 2
                distance = abs(price - mid)
                if distance < min_distance:
                    min_distance = distance
                    nearest = ob
        
        return nearest
    
    def _find_nearest_fvg(self, price: float) -> Optional[FairValueGap]:
        """Find the nearest unfilled FVG"""
        if not self.fvgs:
            return None
        
        nearest = None
        min_distance = float('inf')
        
        for fvg in self.fvgs:
            if fvg.filled_pct < 100:
                mid = (fvg.top + fvg.bottom) / 2
                distance = abs(price - mid)
                if distance < min_distance:
                    min_distance = distance
                    nearest = fvg
        
        return nearest
    
    def _find_nearest_liquidity(self, price: float) -> Optional[LiquidityZone]:
        """Find the nearest liquidity zone"""
        if not self.liquidity_zones:
            return None
        
        nearest = None
        min_distance = float('inf')
        
        for lz in self.liquidity_zones:
            if not lz.swept:
                distance = abs(price - lz.price)
                if distance < min_distance:
                    min_distance = distance
                    nearest = lz
        
        return nearest
    
    def _generate_smc_signal(self, price: float, ob: Optional[OrderBlock], 
                            fvg: Optional[FairValueGap], liq: Optional[LiquidityZone]) -> Dict:
        """Generate trading signal based on SMC analysis - IMPROVED SCORING"""
        signal = {
            'direction': 'NEUTRAL',
            'strength': 0,
            'reason': '',
            'entry_zone': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        # Order Block proximity signal - WIDER PROXIMITY (2% instead of 0.5%)
        if ob:
            if ob.type == 'BULLISH' and price <= ob.top * 1.02:  # Within 2% of OB
                bullish_score += ob.strength  # Full strength, not /2
                reasons.append(f"Bullish OB zone (%{ob.strength})")
                signal['entry_zone'] = {'top': ob.top, 'bottom': ob.bottom}
                signal['stop_loss'] = ob.bottom * 0.99
            elif ob.type == 'BEARISH' and price >= ob.bottom * 0.98:  # Within 2%
                bearish_score += ob.strength  # Full strength
                reasons.append(f"Bearish OB zone (%{ob.strength})")
                signal['entry_zone'] = {'top': ob.top, 'bottom': ob.bottom}
                signal['stop_loss'] = ob.top * 1.01
        
        # FVG fill opportunity - INCREASED SCORE
        if fvg:
            if fvg.type == 'BULLISH' and fvg.filled_pct < 50:
                bullish_score += 35  # Increased from 25
                reasons.append("Unfilled Bullish FVG")
                signal['take_profit'] = fvg.top
            elif fvg.type == 'BEARISH' and fvg.filled_pct < 50:
                bearish_score += 35  # Increased from 25
                reasons.append("Unfilled Bearish FVG")
                signal['take_profit'] = fvg.bottom
        
        # Liquidity targets - INCREASED MULTIPLIER
        if liq:
            if liq.type == 'BUY_STOPS':
                bullish_score += liq.strength * 8  # Increased from 5
                reasons.append(f"Buy stops above ({liq.strength}x)")
                if not signal['take_profit']:
                    signal['take_profit'] = liq.price
            elif liq.type == 'SELL_STOPS':
                bearish_score += liq.strength * 8  # Increased from 5
                reasons.append(f"Sell stops below ({liq.strength}x)")
                if not signal['take_profit']:
                    signal['take_profit'] = liq.price
        
        # Base market structure score - ADD MINIMUM
        if len(self.order_blocks) > 0:
            bullish_obs = len([x for x in self.order_blocks if x.type == 'BULLISH'])
            bearish_obs = len([x for x in self.order_blocks if x.type == 'BEARISH'])
            
            if bullish_obs > bearish_obs:
                bullish_score += 15
            elif bearish_obs > bullish_obs:
                bearish_score += 15
        
        # Determine direction - LOWER THRESHOLD (15 instead of 25)
        if bullish_score > bearish_score and bullish_score >= 15:
            signal['direction'] = 'BULLISH'
            signal['strength'] = min(100, int(bullish_score) + 40)  # Base 40 confidence
        elif bearish_score > bullish_score and bearish_score >= 15:
            signal['direction'] = 'BEARISH'
            signal['strength'] = min(100, int(bearish_score) + 40)  # Base 40 confidence
        else:
            # Even neutral gets some confidence if we have data
            signal['strength'] = 30 if len(self.order_blocks) > 0 else 0
        
        signal['reason'] = ' | '.join(reasons) if reasons else 'No clear SMC setup'
        
        return signal
    
    def _calculate_smc_bias(self) -> str:
        """Calculate overall SMC market bias"""
        bullish_obs = len([ob for ob in self.order_blocks if ob.type == 'BULLISH'])
        bearish_obs = len([ob for ob in self.order_blocks if ob.type == 'BEARISH'])
        
        bullish_fvgs = len([fvg for fvg in self.fvgs if fvg.type == 'BULLISH'])
        bearish_fvgs = len([fvg for fvg in self.fvgs if fvg.type == 'BEARISH'])
        
        bullish_score = bullish_obs + bullish_fvgs
        bearish_score = bearish_obs + bearish_fvgs
        
        if bullish_score > bearish_score + 2:
            return 'BULLISH'
        elif bearish_score > bullish_score + 2:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _ob_to_dict(self, ob: OrderBlock) -> Dict:
        """Convert OrderBlock to dictionary"""
        return {
            'type': ob.type,
            'top': ob.top,
            'bottom': ob.bottom,
            'strength': ob.strength,
            'touched': ob.touched,
            'broken': ob.broken,
            'timestamp': ob.timestamp
        }
    
    def _fvg_to_dict(self, fvg: FairValueGap) -> Dict:
        """Convert FairValueGap to dictionary"""
        return {
            'type': fvg.type,
            'top': fvg.top,
            'bottom': fvg.bottom,
            'filled_pct': fvg.filled_pct,
            'timestamp': fvg.timestamp
        }
    
    def _liq_to_dict(self, lz: LiquidityZone) -> Dict:
        """Convert LiquidityZone to dictionary"""
        return {
            'type': lz.type,
            'price': lz.price,
            'strength': lz.strength,
            'swept': lz.swept
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result when data is insufficient"""
        return {
            'order_blocks': [],
            'fvgs': [],
            'liquidity_zones': [],
            'nearest_order_block': None,
            'nearest_fvg': None,
            'nearest_liquidity': None,
            'smc_signal': {
                'direction': 'NEUTRAL',
                'strength': 0,
                'reason': 'Insufficient data',
                'entry_zone': None,
                'stop_loss': None,
                'take_profit': None
            },
            'smc_bias': 'NEUTRAL',
            'current_price': 0
        }


# Quick test
if __name__ == "__main__":
    import ccxt
    
    print("Testing SMC Analyzer...")
    
    # Fetch real data
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=200)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Analyze
    smc = SMCAnalyzer()
    result = smc.analyze(df)
    
    print(f"\n[SMC] Analysis for BTC/USDT:")
    print(f"Order Blocks: {len(result['order_blocks'])}")
    print(f"FVGs: {len(result['fvgs'])}")
    print(f"Liquidity Zones: {len(result['liquidity_zones'])}")
    print(f"SMC Bias: {result['smc_bias']}")
    print(f"SMC Signal: {result['smc_signal']}")
