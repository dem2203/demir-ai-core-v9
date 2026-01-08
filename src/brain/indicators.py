import pandas as pd
import numpy as np

class Indicators:
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        atr = Indicators.atr(df, period)
        hl2 = (df['high'] + df['low']) / 2
        
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        # SuperTrend Calculation
        st_df = pd.DataFrame(index=df.index)
        st_df['tr'] = atr
        st_df['upper'] = upperband
        st_df['lower'] = lowerband
        st_df['trend'] = 1 # 1: Bull, -1: Bear
        st_df['supertrend'] = 0.0
        
        # Iterative calculation needed for SuperTrend usually, or vector trick
        # Simplified vector approach for speed, exact match requires loop
        # For production robustness, we'll use a loop here as performance impact on 100 candles is neglible
        
        close = df['close'].values
        upper = upperband.values
        lower = lowerband.values
        trend = np.ones(len(df))
        st = np.zeros(len(df))
        
        for i in range(1, len(df)):
            if close[i] > upper[i-1]:
                trend[i] = 1
            elif close[i] < lower[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
                if trend[i] == 1:
                    lower[i] = max(lower[i], lower[i-1])
                else:
                    upper[i] = min(upper[i], upper[i-1])
            
            st[i] = lower[i] if trend[i] == 1 else upper[i]
            
        st_df['supertrend'] = st
        st_df['trend'] = trend
        return st_df

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0):
        sma = df['close'].rolling(period).mean()
        std_dev = df['close'].rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        bandwidth = (upper - lower) / sma
        return upper, lower, bandwidth
