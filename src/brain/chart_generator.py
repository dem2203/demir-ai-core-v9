import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import logging
import os
from datetime import datetime
from src.config import Config
from src.brain.indicators import Indicators

logger = logging.getLogger("CHART_GENERATOR")

class ChartGenerator:
    """
    Generates professional-looking charts for AI Vision analysis.
    """
    def __init__(self):
        os.makedirs(Config.CHARTS_DIR, exist_ok=True)
        
    def generate_chart(self, symbol: str, df: pd.DataFrame) -> str:
        """
        Generate a TradingView-style chart with indicators.
        Returns: filepath to the saved PNG
        """
        if df.empty:
            return None
            
        try:
            # Calculate indicators
            st = Indicators.supertrend(df)
            upper, lower, bandwidth = Indicators.bollinger_bands(df)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                           gridspec_kw={'height_ratios': [3, 1]})
            
            # Price & Indicators (Top)
            ax1.plot(df.index, df['close'], label='Price', color='#2962FF', linewidth=1.5)
            ax1.plot(df.index, st['supertrend'], label='SuperTrend', 
                    color='#26a69a' if st['trend'].iloc[-1] == 1 else '#ef5350', linewidth=1.2)
            ax1.fill_between(df.index, upper, lower, alpha=0.2, color='gray', label='Bollinger Bands')
            
            ax1.set_title(f'{symbol} - 1H Chart Analysis', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price (USDT)', fontsize=10)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Volume (Bottom)
            colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' 
                     for i in range(len(df))]
            ax2.bar(df.index, df['volume'], color=colors, alpha=0.6)
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(Config.CHARTS_DIR, f"{symbol}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Chart saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return None
