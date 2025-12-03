import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from scipy.optimize import minimize

logger = logging.getLogger("PORTFOLIO_OPTIMIZER")

class PortfolioOptimizer:
    """
    DEMIR AI V21.0 - PORTFOLIO OPTIMIZER
    
    Uses Modern Portfolio Theory (MPT) to find optimal asset allocation.
    Maximizes Sharpe ratio (risk-adjusted returns).
    """
    
    LOOKBACK_DAYS = 90  # 90-day returns for optimization
    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
    
    def __init__(self):
        self.optimal_weights = None
        self.expected_return = None
        self.expected_volatility = None
    
    def calculate_returns(self, price_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates daily returns for each asset.
        
        Args:
            price_data_dict: {'BTC/USDT': df, 'ETH/USDT': df, ...}
        
        Returns:
            DataFrame with returns for each asset
        """
        returns_dict = {}
        
        for symbol, df in price_data_dict.items():
            if 'close' in df.columns and len(df) >= self.LOOKBACK_DAYS:
                prices = df['close'].tail(self.LOOKBACK_DAYS)
                returns = prices.pct_change().dropna()
                returns_dict[symbol] = returns
        
        if not returns_dict:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_dict)
        return returns_df
    
    def portfolio_performance(self, weights: np.array, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> tuple:
        """
        Calculates portfolio return and volatility.
        """
        returns = np.sum(mean_returns * weights) * 252  # Annualized
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return returns, volatility
    
    def negative_sharpe_ratio(self, weights: np.array, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Negative Sharpe ratio (for minimization).
        """
        ret, vol = self.portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (ret - self.RISK_FREE_RATE) / vol
        return -sharpe  # Minimize negative = maximize positive
    
    def optimize(self, price_data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Finds optimal portfolio weights.
        
        Returns:
            {
                'weights': {'BTC/USDT': 0.5, 'ETH/USDT': 0.3, ...},
                'expected_return': 0.25,
                'expected_volatility': 0.40,
                'sharpe_ratio': 0.575
            }
        """
        # Calculate returns
        returns_df = self.calculate_returns(price_data_dict)
        
        if returns_df.empty:
            logger.warning("Insufficient data for portfolio optimization")
            return {}
        
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        num_assets = len(returns_df.columns)
        
        # Initial guess: equal weight
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: no short selling (0 <= w <= 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.error("Portfolio optimization failed")
            return {}
        
        # Extract results
        optimal_weights = result.x
        exp_ret, exp_vol = self.portfolio_performance(optimal_weights, mean_returns, cov_matrix)
        sharpe = (exp_ret - self.RISK_FREE_RATE) / exp_vol
        
        # Build result dict
        weights_dict = {symbol: float(weight) for symbol, weight in zip(returns_df.columns, optimal_weights)}
        
        self.optimal_weights = weights_dict
        self.expected_return = exp_ret
        self.expected_volatility = exp_vol
        
        logger.info(f"📊 Optimal Portfolio: {weights_dict}")
        logger.info(f"   Expected Return: {exp_ret*100:.2f}%")
        logger.info(f"   Expected Volatility: {exp_vol*100:.2f}%")
        logger.info(f"   Sharpe Ratio: {sharpe:.3f}")
        
        return {
            'weights': weights_dict,
            'expected_return': exp_ret,
            'expected_volatility': exp_vol,
            'sharpe_ratio': sharpe
        }
