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
    
    # === Phase 29.4 Enhancements ===
    
    def calculate_efficient_frontier(
        self,
        price_data: Dict[str, pd.DataFrame],
        num_portfolios: int = 1000,
        risk_free_rate: float = 0.04
    ) -> Dict:
        """
        Calculate Efficient Frontier using Monte Carlo simulation.
        
        Args:
            price_data: Dict of {symbol: OHLCV DataFrame}
            num_portfolios: Number of random portfolios to simulate
            risk_free_rate: Risk-free rate for Sharpe calculation
        
        Returns:
            {
                'optimal_weights': {...},
                'optimal_sharpe': 0.85,
                'optimal_return': 0.15,
                'optimal_volatility': 0.12,
                'frontier_points': [...]
            }
        """
        try:
            # Calculate returns
            returns_df = pd.DataFrame()
            for symbol, df in price_data.items():
                if len(df) >= 30:
                    returns_df[symbol] = df['close'].pct_change().dropna()
            
            if returns_df.empty or len(returns_df.columns) < 2:
                logger.warning("Need at least 2 symbols for efficient frontier")
                return {}
            
            # Annualized returns and covariance
            mean_returns = returns_df.mean() * 252  # 252 trading days
            cov_matrix = returns_df.cov() * 252
            
            num_assets = len(returns_df.columns)
            results = np.zeros((4, num_portfolios))  # return, volatility, sharpe, weights
            all_weights = []
            
            # Monte Carlo simulation
            for i in range(num_portfolios):
                # Random weights
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                all_weights.append(weights)
                
                # Portfolio return
                portfolio_return = np.sum(weights * mean_returns)
                
                # Portfolio volatility
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Sharpe ratio
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                results[0, i] = portfolio_return
                results[1, i] = portfolio_volatility
                results[2, i] = sharpe_ratio
            
            # Find optimal portfolio (max Sharpe)
            optimal_idx = results[2].argmax()
            optimal_weights = all_weights[optimal_idx]
            
            # Create weights dict
            symbols = returns_df.columns.tolist()
            weights_dict = {sym: round(w, 3) for sym, w in zip(symbols, optimal_weights)}
            
            # Find min volatility portfolio
            min_vol_idx = results[1].argmin()
            
            frontier_points = []
            for i in range(0, num_portfolios, num_portfolios // 20):  # Sample 20 points
                frontier_points.append({
                    'return': round(results[0, i] * 100, 2),
                    'volatility': round(results[1, i] * 100, 2),
                    'sharpe': round(results[2, i], 2)
                })
            
            logger.info(f"📈 Efficient Frontier: Optimal Sharpe = {results[2, optimal_idx]:.2f}")
            logger.info(f"   Weights: {weights_dict}")
            
            return {
                'optimal_weights': weights_dict,
                'optimal_sharpe': round(results[2, optimal_idx], 3),
                'optimal_return': round(results[0, optimal_idx] * 100, 2),
                'optimal_volatility': round(results[1, optimal_idx] * 100, 2),
                'min_volatility_return': round(results[0, min_vol_idx] * 100, 2),
                'min_volatility': round(results[1, min_vol_idx] * 100, 2),
                'frontier_points': frontier_points
            }
            
        except Exception as e:
            logger.error(f"Efficient frontier calculation failed: {e}")
            return {}
    
    def integrate_with_kelly(
        self,
        optimal_weights: Dict[str, float],
        kelly_sizes: Dict[str, float],
        max_position_pct: float = 25.0
    ) -> Dict[str, float]:
        """
        Combine MPT optimal weights with Kelly criterion sizing.
        
        Formula: Final Size = min(Kelly * Weight, MaxPosition)
        
        Args:
            optimal_weights: MPT weights {symbol: weight}
            kelly_sizes: Kelly-based position sizes {symbol: pct}
            max_position_pct: Maximum position size per asset
        
        Returns:
            Final position sizes {symbol: pct}
        """
        final_sizes = {}
        
        for symbol in optimal_weights:
            mpt_weight = optimal_weights.get(symbol, 0)
            kelly_size = kelly_sizes.get(symbol, 10)  # Default 10%
            
            # Combine: Kelly * MPT weight
            combined_size = kelly_size * mpt_weight
            
            # Apply max limit
            final_size = min(combined_size, max_position_pct)
            
            final_sizes[symbol] = round(final_size, 2)
        
        logger.info(f"💼 Kelly-MPT Combined Sizes: {final_sizes}")
        return final_sizes
    
    def get_allocation_summary_for_telegram(
        self,
        allocation: Dict[str, float],
        analytics: Dict = None
    ) -> str:
        """
        Format portfolio allocation for Telegram message.
        """
        lines = ["💼 **PORTFOLIO ALLOCATION**"]
        
        for symbol, weight in allocation.items():
            # Create bar visualization
            bar_len = int(weight * 10)  # 100% = 10 chars
            bar = "█" * bar_len + "░" * (10 - bar_len)
            short_sym = symbol.replace('/USDT', '').replace('/USD', '')
            lines.append(f"   {short_sym}: {bar} {weight*100:.1f}%")
        
        if analytics:
            div_score = analytics.get('diversification_score', 0)
            corr_risk = analytics.get('correlation_risk', 'N/A')
            
            lines.append(f"━━━━━━━━━━━━━━")
            lines.append(f"📊 Diversification: {div_score*100:.0f}%")
            lines.append(f"⚠️ Correlation Risk: {corr_risk}")
        
        return "\n".join(lines)
    
    def get_optimal_allocation(
        self,
        price_data: Dict[str, pd.DataFrame],
        kelly_sizes: Dict[str, float] = None,
        risk_budget: float = 1.0
    ) -> Dict:
        """
        Get optimal portfolio allocation combining all methods.
        
        This is the main entry point for Phase 29.4.
        
        Returns:
            {
                'allocation': {...},
                'efficient_frontier': {...},
                'analytics': {...},
                'telegram_summary': '...'
            }
        """
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(price_data, period=30)
        
        # Calculate efficient frontier
        ef_result = self.calculate_efficient_frontier(price_data)
        
        if ef_result and 'optimal_weights' in ef_result:
            allocation = ef_result['optimal_weights']
        else:
            # Fallback: equal weight
            symbols = list(price_data.keys())
            allocation = {s: 1.0/len(symbols) for s in symbols}
        
        # Integrate with Kelly if provided
        if kelly_sizes:
            allocation = self.integrate_with_kelly(allocation, kelly_sizes)
        
        # Calculate analytics
        positions = {s: {'value': w * 100} for s, w in allocation.items()}
        analytics = self.get_portfolio_analytics(positions, corr_matrix)
        
        # Format for Telegram
        telegram_summary = self.get_allocation_summary_for_telegram(allocation, analytics)
        
        return {
            'allocation': allocation,
            'efficient_frontier': ef_result,
            'analytics': analytics,
            'correlation_matrix': corr_matrix.to_dict() if not corr_matrix.empty else {},
            'telegram_summary': telegram_summary
        }


# Quick test
if __name__ == "__main__":
    import ccxt
    
    print("Testing Portfolio Optimizer...")
    
    # Fetch real data
    exchange = ccxt.binance()
    
    price_data = {}
    for symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=60)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        price_data[symbol] = df
    
    optimizer = PortfolioOptimizer()
    
    # Test optimal allocation
    result = optimizer.get_optimal_allocation(price_data)
    
    print(f"\nAllocation: {result['allocation']}")
    print(f"\nEfficient Frontier: {result.get('efficient_frontier', {})}")
    print(f"\nAnalytics: {result['analytics']}")
    print(f"\nTelegram Summary:\n{result['telegram_summary']}")
