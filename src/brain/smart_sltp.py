"""
Smart SL/TP Calculator

Combines SMC, MTF, and Volume Profile for intelligent SL/TP levels:
1. SL below Order Block (institutional support)
2. TP at FVG fill, Liquidity sweep, or VPOC
3. Multi-target TP system (TP1, TP2, TP3)

ALL DATA FROM BINANCE OHLCV - NO MOCKS!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("SMART_SLTP")


@dataclass
class TradeLevels:
    """Complete trade setup with entry, SL, and multiple TPs"""
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward_1: float
    risk_reward_2: float
    risk_reward_3: float
    sl_reason: str
    tp1_reason: str
    tp2_reason: str
    tp3_reason: str
    risk_pct: float
    direction: str  # 'LONG' or 'SHORT'


class SmartSLTPCalculator:
    """
    Smart Stop Loss / Take Profit Calculator
    
    Uses SMC, MTF, and Volume Profile data to calculate
    institutional-grade trade levels.
    """
    
    def __init__(self, default_risk_pct: float = 1.5):
        self.default_risk_pct = default_risk_pct
    
    def calculate(self, 
                  direction: str,
                  entry_price: float,
                  smc_data: Dict,
                  mtf_data: Dict,
                  vp_data: Dict,
                  atr: float = 0) -> Dict:
        """
        Calculate smart SL/TP levels based on all analysis.
        
        Args:
            direction: 'LONG' or 'SHORT'
            entry_price: Current/entry price
            smc_data: SMC analysis result
            mtf_data: MTF analysis result
            vp_data: Volume Profile result
            atr: ATR value for fallback calculations
            
        Returns:
            Dict with complete trade setup
        """
        if not entry_price or entry_price <= 0:
            return self._empty_result()
        
        # Calculate SL based on SMC Order Blocks
        stop_loss, sl_reason = self._calculate_sl(direction, entry_price, smc_data, atr)
        
        # Calculate multi-target TPs
        tp1, tp1_reason = self._calculate_tp1(direction, entry_price, smc_data, vp_data)
        tp2, tp2_reason = self._calculate_tp2(direction, entry_price, smc_data, vp_data)
        tp3, tp3_reason = self._calculate_tp3(direction, entry_price, smc_data, vp_data)
        
        # Validate levels
        stop_loss, tp1, tp2, tp3 = self._validate_levels(direction, entry_price, stop_loss, tp1, tp2, tp3)
        
        # Calculate Risk/Reward ratios
        risk = abs(entry_price - stop_loss)
        rr1 = abs(tp1 - entry_price) / risk if risk > 0 else 0
        rr2 = abs(tp2 - entry_price) / risk if risk > 0 else 0
        rr3 = abs(tp3 - entry_price) / risk if risk > 0 else 0
        
        # PHASE 126: ENFORCE MINIMUM R:R 1:2
        MIN_RR = 2.0  # Minimum acceptable R:R
        
        # If TP1 R:R is below minimum, recalculate using ATR
        if rr1 < MIN_RR:
            logger.info(f"⚠️ Initial R:R {rr1:.2f} < {MIN_RR} - adjusting targets")
            
            # Use ATR-based targets for better R:R
            if atr > 0:
                if direction == 'LONG':
                    tp1 = entry_price + (atr * MIN_RR)  # 2x ATR target
                    tp2 = entry_price + (atr * 3.0)     # 3x ATR target
                    tp3 = entry_price + (atr * 4.0)     # 4x ATR target
                else:
                    tp1 = entry_price - (atr * MIN_RR)
                    tp2 = entry_price - (atr * 3.0)
                    tp3 = entry_price - (atr * 4.0)
                
                tp1_reason = f"ATR-based R:R {MIN_RR}"
                tp2_reason = "ATR-based 3x"
                tp3_reason = "ATR-based 4x"
            else:
                # Fallback: percentage-based with min R:R
                min_target_pct = risk / entry_price * MIN_RR * 100  # At least 2x risk
                min_target_pct = max(min_target_pct, 2.0)  # Minimum 2%
                
                if direction == 'LONG':
                    tp1 = entry_price * (1 + min_target_pct / 100)
                    tp2 = entry_price * (1 + min_target_pct * 1.5 / 100)
                    tp3 = entry_price * (1 + min_target_pct * 2 / 100)
                else:
                    tp1 = entry_price * (1 - min_target_pct / 100)
                    tp2 = entry_price * (1 - min_target_pct * 1.5 / 100)
                    tp3 = entry_price * (1 - min_target_pct * 2 / 100)
                
                tp1_reason = f"Adjusted for R:R {MIN_RR}"
                tp2_reason = "Adjusted 1.5x"
                tp3_reason = "Adjusted 2x"
            
            # Recalculate R:R
            rr1 = abs(tp1 - entry_price) / risk if risk > 0 else 0
            rr2 = abs(tp2 - entry_price) / risk if risk > 0 else 0
            rr3 = abs(tp3 - entry_price) / risk if risk > 0 else 0
            
            logger.info(f"✅ Adjusted R:R: {rr1:.2f} / {rr2:.2f} / {rr3:.2f}")
        
        # Calculate risk percentage
        risk_pct = (risk / entry_price) * 100
        
        # PHASE 126: Final R:R validation - reject if still below minimum
        is_valid_rr = rr1 >= MIN_RR
        
        return {
            'direction': direction,
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'take_profit_3': tp3,
            'risk_reward_1': round(rr1, 2),
            'risk_reward_2': round(rr2, 2),
            'risk_reward_3': round(rr3, 2),
            'risk_pct': round(risk_pct, 2),
            'sl_reason': sl_reason,
            'tp1_reason': tp1_reason,
            'tp2_reason': tp2_reason,
            'tp3_reason': tp3_reason,
            'valid': risk_pct <= 5.0 and is_valid_rr,  # PHASE 126: Must have R:R >= 2
            'quality': self._assess_setup_quality(rr1, mtf_data),
            'rr_adjusted': rr1 >= MIN_RR  # Flag if we had to adjust
        }
    
    def _calculate_sl(self, direction: str, entry: float, 
                     smc_data: Dict, atr: float) -> Tuple[float, str]:
        """Calculate Stop Loss based on SMC Order Blocks"""
        
        # Try to use nearest Order Block
        nearest_ob = smc_data.get('nearest_order_block')
        
        if nearest_ob and direction == 'LONG':
            if nearest_ob['type'] == 'BULLISH':
                # SL below bullish OB
                sl = nearest_ob['bottom'] * 0.998  # 0.2% below OB
                return sl, f"Below Bullish OB (${nearest_ob['bottom']:,.0f})"
        
        elif nearest_ob and direction == 'SHORT':
            if nearest_ob['type'] == 'BEARISH':
                # SL above bearish OB
                sl = nearest_ob['top'] * 1.002  # 0.2% above OB
                return sl, f"Above Bearish OB (${nearest_ob['top']:,.0f})"
        
        # Fallback to ATR-based SL
        if atr > 0:
            multiplier = 1.5
            if direction == 'LONG':
                sl = entry - (atr * multiplier)
            else:
                sl = entry + (atr * multiplier)
            return sl, f"ATR-based ({multiplier}x ATR)"
        
        # Last resort: percentage-based
        risk_pct = 0.015  # 1.5%
        if direction == 'LONG':
            sl = entry * (1 - risk_pct)
        else:
            sl = entry * (1 + risk_pct)
        return sl, f"Fixed 1.5% risk"
    
    def _calculate_tp1(self, direction: str, entry: float,
                      smc_data: Dict, vp_data: Dict) -> Tuple[float, str]:
        """Calculate TP1 - Nearest target (FVG fill or minor HVN)"""
        
        # Try FVG fill first (fastest target)
        nearest_fvg = smc_data.get('nearest_fvg')
        if nearest_fvg:
            if direction == 'LONG' and nearest_fvg['type'] == 'BULLISH':
                tp = nearest_fvg['top']
                return tp, f"FVG fill (${tp:,.0f})"
            elif direction == 'SHORT' and nearest_fvg['type'] == 'BEARISH':
                tp = nearest_fvg['bottom']
                return tp, f"FVG fill (${tp:,.0f})"
        
        # Try nearest HVN
        hvn_zones = vp_data.get('hvn_zones', [])
        for hvn in hvn_zones:
            if direction == 'LONG' and hvn['price'] > entry:
                return hvn['price'], f"HVN (${hvn['price']:,.0f})"
            elif direction == 'SHORT' and hvn['price'] < entry:
                return hvn['price'], f"HVN (${hvn['price']:,.0f})"
        
        # Fallback: 1% target
        if direction == 'LONG':
            tp = entry * 1.01
        else:
            tp = entry * 0.99
        return tp, "1% move"
    
    def _calculate_tp2(self, direction: str, entry: float,
                      smc_data: Dict, vp_data: Dict) -> Tuple[float, str]:
        """Calculate TP2 - Medium target (VPOC or liquidity)"""
        
        # Try VPOC
        vpoc = vp_data.get('vpoc', 0)
        if vpoc > 0:
            if direction == 'LONG' and vpoc > entry * 1.005:
                return vpoc, f"VPOC (${vpoc:,.0f})"
            elif direction == 'SHORT' and vpoc < entry * 0.995:
                return vpoc, f"VPOC (${vpoc:,.0f})"
        
        # Try liquidity zones
        nearest_liq = smc_data.get('nearest_liquidity')
        if nearest_liq:
            if direction == 'LONG' and nearest_liq['type'] == 'BUY_STOPS':
                return nearest_liq['price'], f"Buy Stops (${nearest_liq['price']:,.0f})"
            elif direction == 'SHORT' and nearest_liq['type'] == 'SELL_STOPS':
                return nearest_liq['price'], f"Sell Stops (${nearest_liq['price']:,.0f})"
        
        # Fallback: 2% target
        if direction == 'LONG':
            tp = entry * 1.02
        else:
            tp = entry * 0.98
        return tp, "2% move"
    
    def _calculate_tp3(self, direction: str, entry: float,
                      smc_data: Dict, vp_data: Dict) -> Tuple[float, str]:
        """Calculate TP3 - Extended target (VAH/VAL or major OB)"""
        
        # Try Value Area bounds
        vah = vp_data.get('vah', 0)
        val = vp_data.get('val', 0)
        
        if direction == 'LONG' and vah > entry * 1.01:
            return vah, f"VAH (${vah:,.0f})"
        elif direction == 'SHORT' and val < entry * 0.99:
            return val, f"VAL (${val:,.0f})"
        
        # Try distant Order Blocks
        order_blocks = smc_data.get('order_blocks', [])
        for ob in order_blocks:
            if direction == 'LONG' and ob['type'] == 'BEARISH' and ob['bottom'] > entry * 1.02:
                return ob['bottom'], f"Bearish OB target (${ob['bottom']:,.0f})"
            elif direction == 'SHORT' and ob['type'] == 'BULLISH' and ob['top'] < entry * 0.98:
                return ob['top'], f"Bullish OB target (${ob['top']:,.0f})"
        
        # Fallback: 3% target
        if direction == 'LONG':
            tp = entry * 1.03
        else:
            tp = entry * 0.97
        return tp, "3% move"
    
    def _validate_levels(self, direction: str, entry: float,
                        sl: float, tp1: float, tp2: float, tp3: float) -> Tuple:
        """Ensure all levels make sense for the direction"""
        
        if direction == 'LONG':
            # SL must be below entry
            if sl >= entry:
                sl = entry * 0.985
            # TPs must be above entry and in order
            if tp1 <= entry:
                tp1 = entry * 1.01
            if tp2 <= tp1:
                tp2 = tp1 * 1.01
            if tp3 <= tp2:
                tp3 = tp2 * 1.01
        else:  # SHORT
            # SL must be above entry
            if sl <= entry:
                sl = entry * 1.015
            # TPs must be below entry and in order
            if tp1 >= entry:
                tp1 = entry * 0.99
            if tp2 >= tp1:
                tp2 = tp1 * 0.99
            if tp3 >= tp2:
                tp3 = tp2 * 0.99
        
        return sl, tp1, tp2, tp3
    
    def _assess_setup_quality(self, rr1: float, mtf_data: Dict) -> str:
        """Assess overall trade setup quality"""
        score = 0
        
        # R:R quality
        if rr1 >= 2:
            score += 40
        elif rr1 >= 1.5:
            score += 25
        elif rr1 >= 1:
            score += 10
        
        # MTF confluence
        confluence_score = mtf_data.get('confluence_score', 0)
        if confluence_score >= 100:
            score += 40
        elif confluence_score >= 67:
            score += 25
        else:
            score += 10
        
        # Entry quality
        entry_score = mtf_data.get('entry_quality', {}).get('score', 0)
        if entry_score >= 80:
            score += 20
        elif entry_score >= 50:
            score += 10
        
        if score >= 80:
            return 'EXCELLENT'
        elif score >= 60:
            return 'GOOD'
        elif score >= 40:
            return 'FAIR'
        else:
            return 'POOR'
    
    def _empty_result(self) -> Dict:
        """Return empty result"""
        return {
            'direction': 'NONE',
            'entry': 0,
            'stop_loss': 0,
            'take_profit_1': 0,
            'take_profit_2': 0,
            'take_profit_3': 0,
            'risk_reward_1': 0,
            'risk_reward_2': 0,
            'risk_reward_3': 0,
            'risk_pct': 0,
            'sl_reason': '',
            'tp1_reason': '',
            'tp2_reason': '',
            'tp3_reason': '',
            'valid': False,
            'quality': 'UNKNOWN'
        }


# Quick test
if __name__ == "__main__":
    print("Testing Smart SL/TP Calculator...")
    
    # Mock data for testing
    smc_data = {
        'nearest_order_block': {'type': 'BULLISH', 'top': 90500, 'bottom': 90000},
        'nearest_fvg': {'type': 'BULLISH', 'top': 91500, 'bottom': 91000},
        'nearest_liquidity': {'type': 'BUY_STOPS', 'price': 92000},
        'order_blocks': []
    }
    
    mtf_data = {
        'confluence_score': 100,
        'entry_quality': {'score': 75}
    }
    
    vp_data = {
        'vpoc': 91000,
        'vah': 92500,
        'val': 89500,
        'hvn_zones': [{'price': 91200, 'significance': 80}]
    }
    
    # Calculate
    calculator = SmartSLTPCalculator()
    result = calculator.calculate(
        direction='LONG',
        entry_price=90200,
        smc_data=smc_data,
        mtf_data=mtf_data,
        vp_data=vp_data,
        atr=500
    )
    
    print(f"\n[SmartSLTP] Trade Setup for LONG @ $90,200:")
    print(f"Stop Loss: ${result['stop_loss']:,.2f} ({result['sl_reason']})")
    print(f"TP1: ${result['take_profit_1']:,.2f} ({result['tp1_reason']}) - R:R {result['risk_reward_1']}")
    print(f"TP2: ${result['take_profit_2']:,.2f} ({result['tp2_reason']}) - R:R {result['risk_reward_2']}")
    print(f"TP3: ${result['take_profit_3']:,.2f} ({result['tp3_reason']}) - R:R {result['risk_reward_3']}")
    print(f"Risk: {result['risk_pct']}%")
    print(f"Quality: {result['quality']}")
