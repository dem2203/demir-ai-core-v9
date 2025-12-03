import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("STATE_BUILDER")

class StateVectorBuilder:
    """
    DEMIR AI V23.0 - UNIFIED STATE VECTOR BUILDER
    
    Aggregates ALL market intelligence into a single 40+ dimension state vector
    for end-to-end RL decision making.
    
    This is the brain's "sensory input" - everything the AI knows about the market.
    """
    
    STATE_DIM = 42  # Total dimensions
    
    def __init__(self):
        self.last_state = None
    
    def build(self, 
              lstm_output: Optional[Dict] = None,
              fractal_data: Optional[Dict] = None,
              orderbook_data: Optional[Dict] = None,
              correlation_data: Optional[Dict] = None,
              funding_data: Optional[Dict] = None,
              volatility_data: Optional[Dict] = None,
              anomaly_data: Optional[Dict] = None,
              macro_data: Optional[Dict] = None,
              position_data: Optional[Dict] = None,
              performance_data: Optional[Dict] = None) -> np.ndarray:
        """
        Builds unified state vector from all intelligence sources.
        
        Returns:
            np.ndarray of shape (42,) - The complete market state
        """
        
        state = np.zeros(self.STATE_DIM)
        idx = 0
        
        # 1. LSTM Outputs (3D)
        if lstm_output:
            state[idx:idx+3] = [
                lstm_output.get('prediction', 0.5),      # Price prediction
                lstm_output.get('confidence', 0.5),      # Model confidence
                lstm_output.get('trend_strength', 0.0)   # Trend magnitude
            ]
        idx += 3
        
        # 2. Fractal Analysis (3D)
        if fractal_data:
            state[idx:idx+3] = [
                self._encode_trend(fractal_data.get('15m', 'NEUTRAL')),
                self._encode_trend(fractal_data.get('1H', 'NEUTRAL')),
                self._encode_trend(fractal_data.get('4H', 'NEUTRAL'))
            ]
        idx += 3
        
        # 3. Order Book Intelligence (5D)
        if orderbook_data:
            whale_support = orderbook_data.get('whale_support', 0)
            whale_resistance = orderbook_data.get('whale_resistance', 0)
            current_price = orderbook_data.get('current_price', 1)
            
            state[idx:idx+5] = [
                (whale_support / current_price - 1) if whale_support else 0,  # Distance to support
                (whale_resistance / current_price - 1) if whale_resistance else 0,  # Distance to resistance
                orderbook_data.get('flow_imbalance', 0) / 1e7,  # Normalized to millions
                orderbook_data.get('bid_ask_ratio', 1.0) - 1.0,  # Center at 0
                orderbook_data.get('depth_score', 0.5)  # 0-1 score
            ]
        idx += 5
        
        # 4. Correlation Risk (4D)
        if correlation_data:
            state[idx:idx+4] = [
                correlation_data.get('btc_spx_corr', 0.0),
                correlation_data.get('btc_dxy_corr', 0.0),
                correlation_data.get('corr_stability', 0.5),
                1.0 if correlation_data.get('regime_shift', False) else 0.0
            ]
        idx += 4
        
        # 5. Funding & Sentiment (3D)
        if funding_data:
            state[idx:idx+3] = [
                np.clip(funding_data.get('binance_rate', 0) * 1000, -1, 1),  # Scale to [-1, 1]
                np.clip(funding_data.get('bybit_rate', 0) * 1000, -1, 1),
                funding_data.get('divergence', 0) * 1000
            ]
        idx += 3
        
        # 6. Volatility & Regime (5D)
        if volatility_data:
            state[idx:idx+5] = [
                volatility_data.get('garch_forecast', 0.02) * 10,  # Scale volatility
                volatility_data.get('hurst', 0.5),
                self._encode_regime(volatility_data.get('regime', 'SIDEWAYS')),
                volatility_data.get('atr_position', 0.5),  # 0=lower band, 1=upper band
                volatility_data.get('volume_profile_position', 0.5)
            ]
        idx += 5
        
        # 7. Anomaly Detection (3D)
        if anomaly_data:
            state[idx:idx+3] = [
                1.0 if anomaly_data.get('is_anomaly', False) else 0.0,
                anomaly_data.get('volume_surge', 1.0) / 5.0,  # Normalize
                anomaly_data.get('price_change_pct', 0.0) / 5.0  # Normalize
            ]
        idx += 3
        
        # 8. Market Context (5D)
        if macro_data:
            hour = datetime.now().hour
            day = datetime.now().weekday()
            
            state[idx:idx+5] = [
                (macro_data.get('dxy', 100) - 100) / 10,  # Normalize DXY
                (macro_data.get('vix', 20) - 20) / 20,    # Normalize VIX
                np.sin(2 * np.pi * hour / 24),  # Time of day (cyclic)
                np.cos(2 * np.pi * hour / 24),
                day / 7.0  # Day of week
            ]
        idx += 5
        
        # 9. Position State (3D)
        if position_data:
            state[idx:idx+3] = [
                position_data.get('position', 0),  # -1, 0, 1
                position_data.get('days_in_position', 0) / 30.0,  # Normalize
                position_data.get('unrealized_pnl_pct', 0.0) / 10.0  # Normalize
            ]
        idx += 3
        
        # 10. Historical Performance (3D)
        if performance_data:
            state[idx:idx+3] = [
                performance_data.get('win_rate', 0.5),
                (performance_data.get('sharpe_ratio', 0.0) + 1) / 3.0,  # Normalize to [0, 1]
                performance_data.get('max_drawdown', 0.0) / -0.5  # Normalize
            ]
        idx += 3
        
        # Sanity check
        assert idx == self.STATE_DIM, f"State dimension mismatch: {idx} != {self.STATE_DIM}"
        
        self.last_state = state
        logger.debug(f"Built state vector: shape={state.shape}, range=[{state.min():.3f}, {state.max():.3f}]")
        
        return state
    
    def _encode_trend(self, trend: str) -> float:
        """Encodes trend string to float: BEARISH=-1, NEUTRAL=0, BULLISH=1"""
        mapping = {'BEARISH': -1.0, 'NEUTRAL': 0.0, 'BULLISH': 1.0}
        return mapping.get(trend, 0.0)
    
    def _encode_regime(self, regime: str) -> float:
        """Encodes market regime: TRENDING=1, SIDEWAYS=0, VOLATILE=-1"""
        mapping = {'TRENDING': 1.0, 'SIDEWAYS': 0.0, 'VOLATILE': -1.0}
        return mapping.get(regime, 0.0)
    
    def get_state_description(self) -> Dict:
        """Returns human-readable description of last state vector."""
        if self.last_state is None:
            return {}
        
        return {
            'lstm': self.last_state[0:3].tolist(),
            'fractal': self.last_state[3:6].tolist(),
            'orderbook': self.last_state[6:11].tolist(),
            'correlation': self.last_state[11:15].tolist(),
            'funding': self.last_state[15:18].tolist(),
            'volatility': self.last_state[18:23].tolist(),
            'anomaly': self.last_state[23:26].tolist(),
            'macro': self.last_state[26:31].tolist(),
            'position': self.last_state[31:34].tolist(),
            'performance': self.last_state[34:37].tolist()
        }
