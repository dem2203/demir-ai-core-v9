"""
Liquidation Heatmap Engine - Squeeze Zone Predictor

Why this is powerful:
- High leverage traders (25x, 50x, 100x) get liquidated at predictable levels.
- Price acts like a MAGNET to these levels to clear liquidity.
- Identify "Long Squeeze" (price dumping to hit stops) and "Short Squeeze" (price pumping).

Mathematical Model:
- 100x Leverage = 0.8% distance from entry
- 50x Leverage = 1.8% distance
- 25x Leverage = 3.8% distance
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List

logger = logging.getLogger("LIQUIDATION_TRACKER")

class LiquidationTracker:
    def __init__(self):
        # Estimated liquidation distances (entry -> liq price)
        # Includes maintenance margin buffer
        self.lev_tiers = {
            "100x": 0.008, # 0.8% move kills 100x
            "50x": 0.018,  # 1.8% move kills 50x
            "25x": 0.038   # 3.8% move kills 25x
        }
        
    def analyze_liquidation_zones(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Estimate liquidation clusters based on Swing Highs/Lows
        """
        if df.empty:
            return {}
            
        # 1. Find Pivot Points (Swing Highs/Lows)
        # Looking back 50 candles (approx 2 days on 1h chart)
        window = 50
        recent_high = df['high'].rolling(window=window).max().iloc[-1]
        recent_low = df['low'].rolling(window=window).min().iloc[-1]
        
        clusters = []
        
        # 2. Calculate Short Liquidation Levels (Above Pivot Highs)
        # Shorts enter at highs. Their liq price is HIGHER.
        for tier, dist in self.lev_tiers.items():
            liq_price = recent_high * (1 + dist)
            clusters.append({
                "type": "SHORT_LIQ",
                "leverage": tier,
                "price": liq_price,
                "strength": 1.0 if tier == "50x" else 0.5
            })
            
        # 3. Calculate Long Liquidation Levels (Below Pivot Lows)
        # Longs enter at lows. Their liq price is LOWER.
        for tier, dist in self.lev_tiers.items():
            liq_price = recent_low * (1 - dist)
            clusters.append({
                "type": "LONG_LIQ",
                "leverage": tier,
                "price": liq_price,
                "strength": 1.0 if tier == "50x" else 0.5
            })
            
        # 4. Check Proximity (Magnet Effect)
        magnet_zone = None
        closest_dist = float('inf')
        
        for cluster in clusters:
            dist_pct = abs(current_price - cluster['price']) / current_price
            
            # If within 0.3% range, we are in the Magnet Zone
            if dist_pct < 0.003:
                magnet_zone = cluster
                closest_dist = dist_pct
                
        # 5. Generate Signal
        signal = "NEUTRAL"
        reasoning = "No nearby liquidation clusters"
        
        if magnet_zone:
            if magnet_zone["type"] == "SHORT_LIQ":
                # Price pumping to kill shorts -> Bullish untill hit
                signal = "MAGNET_UP"
                reasoning = f"🧲 Magnet Effect: Pulling to ${magnet_zone['price']:.0f} (Short Squeeze 50x)"
            elif magnet_zone["type"] == "LONG_LIQ":
                # Price dumping to kill longs -> Bearish untill hit
                signal = "MAGNET_DOWN"
                reasoning = f"🧲 Magnet Effect: Pulling to ${magnet_zone['price']:.0f} (Long Squeeze 50x)"
                
        return {
            "signal": signal,
            "magnet_price": magnet_zone['price'] if magnet_zone else None,
            "clusters": clusters,
            "reasoning": reasoning,
            "risk_score": 8 if magnet_zone else 2
        }
