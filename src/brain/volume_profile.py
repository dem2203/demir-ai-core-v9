"""
Enhanced Volume Profile Analyzer

Calculates volume distribution across price levels:
1. VPOC (Volume Point of Control) - Highest volume price
2. VAH/VAL (Value Area High/Low) - 70% of volume range
3. HVN (High Volume Nodes) - Support/resistance zones
4. LVN (Low Volume Nodes) - Fast-move zones

ALL DATA FROM BINANCE OHLCV - NO MOCKS!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("VOLUME_PROFILE")


@dataclass
class VolumeNode:
    """Represents a volume node zone"""
    price: float
    volume: float
    type: str  # 'HVN' or 'LVN'
    significance: int  # 0-100


class VolumeProfileAnalyzer:
    """
    Enhanced Volume Profile Analyzer
    
    Calculates VPOC, HVN, LVN from OHLCV data only.
    Zero external dependencies.
    """
    
    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Full Volume Profile analysis.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            Dict with VPOC, VAH, VAL, HVN, LVN zones
        """
        if df is None or df.empty or len(df) < 20:
            logger.warning("Insufficient data for Volume Profile analysis")
            return self._empty_result()
        
        # Calculate volume profile
        profile = self._calculate_volume_profile(df)
        
        if not profile:
            return self._empty_result()
        
        # Get VPOC
        vpoc = self._find_vpoc(profile)
        
        # Get Value Area (70% of volume)
        vah, val = self._find_value_area(profile)
        
        # Get HVN and LVN zones
        hvn_zones = self._find_hvn(profile)
        lvn_zones = self._find_lvn(profile)
        
        # Current price context
        current_price = df['close'].iloc[-1]
        
        # Calculate price magnets (where price is likely to move)
        price_magnets = self._calculate_price_magnets(current_price, vpoc, hvn_zones, lvn_zones)
        
        # Volume-based signal
        volume_signal = self._generate_volume_signal(current_price, vpoc, vah, val, hvn_zones)
        
        return {
            'vpoc': vpoc,
            'vah': vah,
            'val': val,
            'hvn_zones': [self._node_to_dict(h) for h in hvn_zones[:5]],
            'lvn_zones': [self._node_to_dict(l) for l in lvn_zones[:5]],
            'price_magnets': price_magnets,
            'volume_signal': volume_signal,
            'current_price': current_price,
            'price_position': self._get_price_position(current_price, vpoc, vah, val)
        }
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[float, float]:
        """Calculate volume distribution across price bins"""
        profile = {}
        
        # Find price range
        min_price = df['low'].min()
        max_price = df['high'].max()
        price_range = max_price - min_price
        
        if price_range <= 0:
            return profile
        
        bin_size = price_range / self.num_bins
        
        # Distribute volume across price bins using typical price
        for _, candle in df.iterrows():
            # Use typical price (HLC/3) for volume distribution
            typical_price = (candle['high'] + candle['low'] + candle['close']) / 3
            
            # Find bin
            bin_idx = int((typical_price - min_price) / bin_size)
            bin_idx = max(0, min(bin_idx, self.num_bins - 1))
            
            # Calculate bin price (center of bin)
            bin_price = min_price + (bin_idx + 0.5) * bin_size
            
            # Add volume
            if bin_price not in profile:
                profile[bin_price] = 0
            profile[bin_price] += candle['volume']
        
        return profile
    
    def _find_vpoc(self, profile: Dict[float, float]) -> float:
        """Find Volume Point of Control (highest volume price)"""
        if not profile:
            return 0
        return max(profile.keys(), key=lambda x: profile[x])
    
    def _find_value_area(self, profile: Dict[float, float]) -> Tuple[float, float]:
        """Find Value Area High and Low (70% of total volume)"""
        if not profile:
            return 0, 0
        
        total_volume = sum(profile.values())
        target_volume = total_volume * 0.70
        
        # Sort by price
        sorted_prices = sorted(profile.keys())
        vpoc = self._find_vpoc(profile)
        vpoc_idx = sorted_prices.index(vpoc)
        
        # Expand from VPOC until 70% volume captured
        accumulated_volume = profile[vpoc]
        lower_idx = vpoc_idx
        upper_idx = vpoc_idx
        
        while accumulated_volume < target_volume:
            # Check which side has more volume
            lower_vol = profile.get(sorted_prices[lower_idx - 1], 0) if lower_idx > 0 else 0
            upper_vol = profile.get(sorted_prices[upper_idx + 1], 0) if upper_idx < len(sorted_prices) - 1 else 0
            
            if lower_vol >= upper_vol and lower_idx > 0:
                lower_idx -= 1
                accumulated_volume += profile[sorted_prices[lower_idx]]
            elif upper_idx < len(sorted_prices) - 1:
                upper_idx += 1
                accumulated_volume += profile[sorted_prices[upper_idx]]
            else:
                break
        
        val = sorted_prices[lower_idx]
        vah = sorted_prices[upper_idx]
        
        return vah, val
    
    def _find_hvn(self, profile: Dict[float, float]) -> List[VolumeNode]:
        """Find High Volume Nodes (above 70th percentile)"""
        if not profile:
            return []
        
        volumes = list(profile.values())
        threshold = np.percentile(volumes, 70)
        
        hvn_zones = []
        for price, volume in profile.items():
            if volume >= threshold:
                significance = min(100, int((volume / max(volumes)) * 100))
                hvn_zones.append(VolumeNode(
                    price=price,
                    volume=volume,
                    type='HVN',
                    significance=significance
                ))
        
        # Sort by significance
        hvn_zones.sort(key=lambda x: x.significance, reverse=True)
        return hvn_zones
    
    def _find_lvn(self, profile: Dict[float, float]) -> List[VolumeNode]:
        """Find Low Volume Nodes (below 30th percentile)"""
        if not profile:
            return []
        
        volumes = list(profile.values())
        threshold = np.percentile(volumes, 30)
        max_vol = max(volumes)
        
        lvn_zones = []
        for price, volume in profile.items():
            if volume <= threshold:
                # LVN significance is inverse (lower volume = more significant gap)
                significance = min(100, int((1 - volume / max_vol) * 100))
                lvn_zones.append(VolumeNode(
                    price=price,
                    volume=volume,
                    type='LVN',
                    significance=significance
                ))
        
        # Sort by significance
        lvn_zones.sort(key=lambda x: x.significance, reverse=True)
        return lvn_zones
    
    def _calculate_price_magnets(self, current_price: float, vpoc: float,
                                 hvn_zones: List[VolumeNode], 
                                 lvn_zones: List[VolumeNode]) -> List[Dict]:
        """Calculate likely price targets based on volume profile"""
        magnets = []
        
        # VPOC is always a major magnet
        if vpoc > 0:
            distance = abs(current_price - vpoc)
            direction = 'UP' if vpoc > current_price else 'DOWN'
            magnets.append({
                'price': vpoc,
                'type': 'VPOC',
                'direction': direction,
                'distance_pct': (distance / current_price) * 100,
                'strength': 100
            })
        
        # Nearest HVN zones as magnets
        for hvn in hvn_zones[:3]:
            distance = abs(current_price - hvn.price)
            direction = 'UP' if hvn.price > current_price else 'DOWN'
            magnets.append({
                'price': hvn.price,
                'type': 'HVN',
                'direction': direction,
                'distance_pct': (distance / current_price) * 100,
                'strength': hvn.significance
            })
        
        # LVN zones as fast-move areas (targets beyond)
        for lvn in lvn_zones[:2]:
            distance = abs(current_price - lvn.price)
            direction = 'UP' if lvn.price > current_price else 'DOWN'
            magnets.append({
                'price': lvn.price,
                'type': 'LVN',
                'direction': direction,
                'distance_pct': (distance / current_price) * 100,
                'strength': lvn.significance
            })
        
        # Sort by distance
        magnets.sort(key=lambda x: x['distance_pct'])
        
        return magnets[:5]
    
    def _generate_volume_signal(self, current_price: float, vpoc: float,
                                vah: float, val: float,
                                hvn_zones: List[VolumeNode]) -> Dict:
        """Generate trading signal based on volume profile"""
        signal = {
            'bias': 'NEUTRAL',
            'strength': 0,
            'reason': '',
            'key_level': None
        }
        
        if vpoc == 0:
            return signal
        
        # Price position relative to value area
        if current_price > vah:
            signal['bias'] = 'BULLISH'
            signal['strength'] = 70
            signal['reason'] = 'Price above Value Area High - bullish continuation'
            signal['key_level'] = vah
        elif current_price < val:
            signal['bias'] = 'BEARISH'
            signal['strength'] = 70
            signal['reason'] = 'Price below Value Area Low - bearish continuation'
            signal['key_level'] = val
        elif abs(current_price - vpoc) / vpoc < 0.01:  # Within 1% of VPOC
            signal['bias'] = 'NEUTRAL'
            signal['strength'] = 50
            signal['reason'] = 'Price at VPOC - fair value, consolidation expected'
            signal['key_level'] = vpoc
        else:
            # Between VAL and VAH but not at VPOC
            if current_price > vpoc:
                signal['bias'] = 'SLIGHT_BULLISH'
                signal['strength'] = 55
                signal['reason'] = 'Price above VPOC within Value Area'
            else:
                signal['bias'] = 'SLIGHT_BEARISH'
                signal['strength'] = 55
                signal['reason'] = 'Price below VPOC within Value Area'
            signal['key_level'] = vpoc
        
        return signal
    
    def _get_price_position(self, current_price: float, vpoc: float, 
                           vah: float, val: float) -> str:
        """Get current price position relative to volume profile"""
        if current_price > vah:
            return 'ABOVE_VALUE_AREA'
        elif current_price < val:
            return 'BELOW_VALUE_AREA'
        elif abs(current_price - vpoc) / vpoc < 0.005:
            return 'AT_VPOC'
        elif current_price > vpoc:
            return 'UPPER_VALUE_AREA'
        else:
            return 'LOWER_VALUE_AREA'
    
    def _node_to_dict(self, node: VolumeNode) -> Dict:
        """Convert VolumeNode to dictionary"""
        return {
            'price': node.price,
            'volume': node.volume,
            'type': node.type,
            'significance': node.significance
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result when data is insufficient"""
        return {
            'vpoc': 0,
            'vah': 0,
            'val': 0,
            'hvn_zones': [],
            'lvn_zones': [],
            'price_magnets': [],
            'volume_signal': {'bias': 'NEUTRAL', 'strength': 0, 'reason': 'No data'},
            'current_price': 0,
            'price_position': 'UNKNOWN'
        }


# Quick test
if __name__ == "__main__":
    import ccxt
    
    print("Testing Volume Profile Analyzer...")
    
    # Fetch real data
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=200)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Analyze
    vp = VolumeProfileAnalyzer()
    result = vp.analyze(df)
    
    print(f"\n[VolumeProfile] Analysis for BTC/USDT:")
    print(f"VPOC: ${result['vpoc']:,.2f}")
    print(f"VAH: ${result['vah']:,.2f}")
    print(f"VAL: ${result['val']:,.2f}")
    print(f"HVN Zones: {len(result['hvn_zones'])}")
    print(f"LVN Zones: {len(result['lvn_zones'])}")
    print(f"Price Position: {result['price_position']}")
    print(f"Volume Signal: {result['volume_signal']}")
    print(f"Price Magnets: {result['price_magnets'][:3]}")
