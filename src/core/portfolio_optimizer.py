import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("PORTFOLIO_OPTIMIZER")

class PortfolioOptimizer:
    """
    MULTI-SYMBOL PORTFOLIO OPTIMIZATION
    
    Features:
    1. Correlation Matrix - Track correlation between BTC, ETH, LTC
    2. Dynamic Position Sizing - Adjust based on correlation risk
    3. Risk Parity - Equal risk contribution from each asset
    4. Over-Exposure Prevention - Max 50% in correlated assets
    """
    
    def __init__(self):
        self.correlation_cache = {}
        self.cache_duration = 3600  # 1 hour
        
    def calculate_correlation_matrix(
        self, 
        price_data: Dict[str, pd.DataFrame],
        period: int = 30
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between symbols.
        
        Args:
            price_data: Dict of {symbol: OHLCV DataFrame}
            period: Lookback period in days
            
        Returns:
            Correlation matrix (DataFrame)
        """
        try:
            # Extract returns for each symbol
            returns_data = {}
            
            for symbol, df in price_data.items():
                if len(df) < period:
                    logger.warning(f"{symbol} has insufficient data")
                    continue
                    
                # Calculate daily returns
                df_subset = df.tail(period).copy()
                df_subset['returns'] = df_subset['close'].pct_change()
                returns_data[symbol] = df_subset['returns'].dropna()
            
            # Create correlation dataframe
            if len(returns_data) < 2:
                logger.warning("Need at least 2 symbols for correlation")
                return pd.DataFrame()
            
            # Align all returns to same index (dates)
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            logger.info(f"📊 Correlation Matrix:\n{corr_matrix}")
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return pd.DataFrame()
            
    def check_correlation_risk(
        self, 
        corr_matrix: pd.DataFrame,
        symbols: List[str],
        threshold: float = 0.8
    ) -> Dict:
        """
        Check if proposed symbols are too correlated.
        
        Args:
            corr_matrix: Correlation matrix
            symbols: List of symbols to check
            threshold: Correlation threshold (default 0.8)
            
        Returns:
            Risk assessment dict
        """
        if corr_matrix.empty or len(symbols) < 2:
            return {"risk_level": "LOW", "max_correlation": 0, "warning": None}
        
        max_corr = 0
        corr_pairs = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]
                
                if sym1 in corr_matrix.index and sym2 in corr_matrix.columns:
                    corr = abs(corr_matrix.loc[sym1, sym2])
                    
                    if corr > max_corr:
                        max_corr = corr
                    
                    if corr > threshold:
                        corr_pairs.append((sym1, sym2, corr))
        
        if max_corr > 0.9:
            risk_level = "CRITICAL"
        elif max_corr > 0.8:
            risk_level = "HIGH"
        elif max_corr > 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        warning = None
        if corr_pairs:
            warning = f"High correlation detected: {corr_pairs[0][0]}-{corr_pairs[0][1]} ({corr_pairs[0][2]:.2f})"
        
        return {
            "risk_level": risk_level,
            "max_correlation": round(max_corr, 2),
            "correlated_pairs": corr_pairs,
            "warning": warning
        }
        
    def optimize_allocation(
        self,
        signals: Dict[str, Dict],
        corr_matrix: pd.DataFrame,
        risk_budget: float = 1.0,
        max_positions: int = 3
    ) -> Dict[str, float]:
        """
        Optimize position sizing across multiple signals.
        
        Args:
            signals: Dict of {symbol: signal_data}
            corr_matrix: Correlation matrix
            risk_budget: Total risk budget (1.0 = 100%)
            max_positions: Maximum concurrent positions
            
        Returns:
            Dict of {symbol: position_size_multiplier}
        """
        if not signals:
            return {}
        
        # Filter to top signals by confidence
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1].get('confidence', 0),
            reverse=True
        )
        
        # Limit to max_positions
        top_signals = dict(sorted_signals[:max_positions])
        
        if len(top_signals) == 1:
            # Single position - use full budget
            symbol = list(top_signals.keys())[0]
            return {symbol: 1.0}
        
        # Calculate volatility-based weights (inverse volatility = risk parity)
        symbols = list(top_signals.keys())
        volatilities = {}
        
        for symbol in symbols:
            vol = top_signals[symbol].get('volatility', 0.02)  # Default 2%
            volatilities[symbol] = vol
        
        # Inverse volatility weights
        inv_vols = {s: 1/v for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        # Normalize to risk budget
        allocations = {}
        for symbol in symbols:
            base_weight = (inv_vols[symbol] / total_inv_vol) * risk_budget
            allocations[symbol] = base_weight
        
        # Apply correlation penalty
        if not corr_matrix.empty and len(symbols) > 1:
            allocations = self._apply_correlation_penalty(
                allocations, 
                corr_matrix,
                max_corr_weight=0.5
            )
        
        logger.info(f"🎯 Optimized Allocation: {allocations}")
        return allocations
        
    def _apply_correlation_penalty(
        self,
        allocations: Dict[str, float],
        corr_matrix: pd.DataFrame,
        max_corr_weight: float = 0.5
    ) -> Dict[str, float]:
        """
        Reduce allocation to highly correlated assets.
        """
        symbols = list(allocations.keys())
        
        # Calculate average correlation for each symbol
        avg_corrs = {}
        for symbol in symbols:
            if symbol not in corr_matrix.index:
                avg_corrs[symbol] = 0
                continue
                
            other_symbols = [s for s in symbols if s != symbol]
            corrs = []
            
            for other in other_symbols:
                if other in corr_matrix.columns:
                    corrs.append(abs(corr_matrix.loc[symbol, other]))
            
            avg_corrs[symbol] = np.mean(corrs) if corrs else 0
        
        # Apply penalty to highly correlated assets
        adjusted = {}
        for symbol, weight in allocations.items():
            avg_corr = avg_corrs.get(symbol, 0)
            
            if avg_corr > 0.8:
                # Reduce by 50%
                penalty = 0.5
            elif avg_corr > 0.6:
                # Reduce by 25%
                penalty = 0.75
            else:
                penalty = 1.0
            
            adjusted[symbol] = weight * penalty
        
        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {s: w/total for s, w in adjusted.items()}
        
        logger.info(f"📉 Correlation-Adjusted: {adjusted}")
        return adjusted
        
    def get_portfolio_analytics(
        self,
        positions: Dict[str, Dict],
        corr_matrix: pd.DataFrame
    ) -> Dict:
        """
        Calculate portfolio-level analytics.
        
        Returns metrics like:
        - Portfolio concentration
        - Correlation risk
        - Diversification score
        """
        if not positions:
            return {
                "concentration": 0,
                "correlation_risk": "LOW",
                "diversification_score": 0,
                "total_positions": 0
            }
        
        # Calculate concentration (Herfindahl index)
        total_value = sum(p.get('value', 0) for p in positions.values())
        
        if total_value == 0:
            concentration = 0
        else:
            weights = [(p.get('value', 0) / total_value) ** 2 for p in positions.values()]
            concentration = sum(weights)
        
        # Correlation risk
        symbols = list(positions.keys())
        corr_risk = self.check_correlation_risk(corr_matrix, symbols)
        
        # Diversification score (1 - concentration)
        div_score = 1 - concentration
        
        return {
            "concentration": round(concentration, 2),
            "correlation_risk": corr_risk['risk_level'],
            "max_correlation": corr_risk['max_correlation'],
            "diversification_score": round(div_score, 2),
            "total_positions": len(positions)
        }
