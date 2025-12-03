import pandas as pd
import numpy as np
import logging
from typing import Optional
from arch import arch_model

logger = logging.getLogger("VOLATILITY_FORECASTER")

class VolatilityForecaster:
    """
    DEMIR AI V21.0 - GARCH VOLATILITY FORECASTER
    
    Uses GARCH(1,1) model to predict future volatility.
    Helps AI avoid trading during unstable periods.
    """
    
    FORECAST_HORIZON = 24  # Hours ahead to forecast
    TRAINING_WINDOW = 500  # Hours of data to train on
    HIGH_VOL_THRESHOLD = 0.05  # 5% volatility is "too high"
    
    def __init__(self):
        self.model = None
        self.last_forecast = None
    
    def fit_garch(self, returns: pd.Series) -> bool:
        """
        Fits GARCH(1,1) model on return series.
        """
        try:
            # GARCH(1,1): Most common specification
            self.model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            self.model_fit = self.model.fit(disp='off')
            logger.info("GARCH(1,1) model fitted successfully")
            return True
        except Exception as e:
            logger.error(f"GARCH fitting failed: {e}")
            return False
    
    def forecast_volatility(self, price_data: pd.DataFrame) -> Optional[float]:
        """
        Forecasts next-hour volatility using GARCH.
        
        Args:
            price_data: DataFrame with 'close' prices (at least 500 rows)
        
        Returns:
            Forecasted volatility (annualized standard deviation)
        """
        if price_data is None or len(price_data) < self.TRAINING_WINDOW:
            logger.warning("Insufficient data for GARCH forecasting")
            return None
        
        try:
            # Calculate returns
            prices = price_data['close'].tail(self.TRAINING_WINDOW)
            returns = prices.pct_change().dropna() * 100  # Percentage returns
            
            # Fit model
            if not self.fit_garch(returns):
                return None
            
            # Forecast
            forecast = self.model_fit.forecast(horizon=self.FORECAST_HORIZON)
            forecasted_variance = forecast.variance.values[-1, :]
            
            # Convert variance to volatility (std dev)
            forecasted_vol = np.sqrt(forecasted_variance.mean())
            
            self.last_forecast = forecasted_vol / 100  # Back to decimal
            
            logger.info(f"📊 Forecasted Volatility (24h): {self.last_forecast*100:.2f}%")
            
            return self.last_forecast
            
        except Exception as e:
            logger.error(f"Volatility forecast error: {e}")
            return None
    
    def is_volatility_too_high(self, forecasted_vol: float) -> bool:
        """
        Determines if forecasted volatility is too risky to trade.
        """
        if forecasted_vol is None:
            return False
        
        if forecasted_vol > self.HIGH_VOL_THRESHOLD:
            logger.warning(f"⚠️ HIGH VOLATILITY WARNING: {forecasted_vol*100:.2f}% > {self.HIGH_VOL_THRESHOLD*100}%")
            return True
        
        return False
