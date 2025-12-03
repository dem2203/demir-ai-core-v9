import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize

logger = logging.getLogger("PORTFOLIO_MANAGER")

class PortfolioManager:
    """
    MODERN PORTFOLIO THEORY (MPT) ENGINE
    
    Amaç: Sharpe Oranını (Risk başına düşen kar) maksimize eden
    portföy ağırlıklarını bulmak.
    """
    
    def __init__(self):
        pass
        
    def optimize_allocation(self, price_history: pd.DataFrame) -> dict:
        """
        price_history: Sütunları coin isimleri (BTC, ETH...) olan DataFrame.
        """
        try:
            # Günlük getiriler
            returns = price_history.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_assets = len(price_history.columns)
            
            if num_assets < 2:
                return {col: 1.0 for col in price_history.columns}

            # Negatif Sharpe Oranı (Minimize etmek için)
            def negative_sharpe(weights):
                portfolio_return = np.sum(mean_returns * weights) * 252 * 24 # Saatlik -> Yıllık
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252 * 24)
                return -portfolio_return / portfolio_std

            # Kısıtlamalar: Ağırlıklar toplamı 1 olmalı
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            # Her coin en az %0, en fazla %100 olabilir
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Başlangıç: Eşit dağılım
            init_guess = num_assets * [1. / num_assets,]
            
            result = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
            allocation = {}
            for i, col in enumerate(price_history.columns):
                allocation[col] = round(result.x[i], 2)
                
            logger.info(f"Optimal Allocation: {allocation}")
            return allocation
            
        except Exception as e:
            logger.error(f"Portfolio Optimization Failed: {e}")
            # Hata durumunda eşit dağıt
            return {col: 1.0/len(price_history.columns) for col in price_history.columns}
