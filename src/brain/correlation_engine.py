import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("CORRELATION_ENGINE")

class CorrelationEngine:
    """
    DEMIR AI V20.0 - CORRELATION MATRIX ENGINE
    
    Calculates rolling correlations between crypto assets and macro indices.
    Helps avoid signals when correlations are unstable (regime change).
    """
    
    CORRELATION_WINDOW = 30  # 30-day rolling correlation
    STABILITY_THRESHOLD = 0.3  # If correlation changes by >30%, it's unstable
    
    def __init__(self):
        self.historical_correlations = {}  # Cache for historical data
        self.last_update = None
    
    def calculate_correlation_matrix(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates correlation matrix from multiple asset dataframes.
        
        Args:
            data_dict: {'BTC': df with 'close', 'ETH': df, 'SPX': df, ...}
        
        Returns:
            Correlation matrix as DataFrame
        """
        if not data_dict:
            return pd.DataFrame()
        
        try:
            # Combine all close prices into one dataframe
            combined = pd.DataFrame()
            
            for asset_name, df in data_dict.items():
                if 'close' in df.columns:
                    # Use tail to get recent data for correlation
                    combined[asset_name] = df['close'].tail(self.CORRELATION_WINDOW).values
            
            if combined.empty:
                return pd.DataFrame()
            
            # Calculate correlation matrix
            corr_matrix = combined.corr()
            
            logger.info(f"Correlation Matrix calculated for {len(combined.columns)} assets")
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return pd.DataFrame()
    
    def detect_correlation_breakdown(self, current_corr: float, historical_corr: float) -> bool:
        """
        Detects if correlation has broken down (regime change signal).
        
        Example: If BTC-SPX was 0.7 but now is 0.2, correlation broke down.
        """
        if historical_corr is None:
            return False
        
        change = abs(current_corr - historical_corr)
        
        if change > self.STABILITY_THRESHOLD:
            logger.warning(f"⚠️ CORRELATION BREAKDOWN: Historical {historical_corr:.2f} → Current {current_corr:.2f}")
            return True
        
        return False
    
    def check_signal_correlation_risk(self, 
                                      corr_matrix: pd.DataFrame, 
                                      signal_asset: str,
                                      macro_assets: List[str] = ['SPX', 'NDQ', 'DXY']) -> Dict:
        """
        Checks if the signal asset's correlation with macro indices is risky.
        
        Returns:
            {
                'safe_to_trade': bool,
                'risk_reason': str,
                'correlations': {...}
            }
        """
        if corr_matrix.empty or signal_asset not in corr_matrix.columns:
            return {'safe_to_trade': True, 'risk_reason': 'No correlation data', 'correlations': {}}
        
        correlations = {}
        risk_flags = []
        
        for macro in macro_assets:
            if macro in corr_matrix.columns:
                corr_value = corr_matrix.loc[signal_asset, macro]
                correlations[macro] = corr_value
                
                # Risk Check: If BTC is highly correlated with SPX and SPX is falling
                # we might want to avoid long signals
                # (This logic can be expanded based on actual macro trend)
                
                if abs(corr_value) > 0.8:
                    risk_flags.append(f"High correlation with {macro} ({corr_value:.2f})")
        
        safe_to_trade = len(risk_flags) == 0
        risk_reason = '; '.join(risk_flags) if risk_flags else 'Correlations stable'
        
        return {
            'safe_to_trade': safe_to_trade,
            'risk_reason': risk_reason,
            'correlations': correlations
        }
    
    def get_heatmap_data(self, corr_matrix: pd.DataFrame) -> Dict:
        """
        Formats correlation matrix for Dashboard heatmap visualization.
        """
        if corr_matrix.empty:
            return {}
        
        return {
            'columns': corr_matrix.columns.tolist(),
            'index': corr_matrix.index.tolist(),
            'values': corr_matrix.values.tolist()
        }
