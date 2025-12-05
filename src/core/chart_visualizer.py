import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger("CHART_VISUALIZER")

class ChartVisualizer:
    """
    LIVE TRADING CHART - TradingView Style
    
    Features:
    1. Interactive candlestick charts
    2. Entry/Exit markers for paper trades
    3. SL/TP lines
    4. Real-time position tracking
    5. P&L annotations
    """
    
    def __init__(self):
        self.theme = "plotly_dark"
        
    def create_trading_chart(
        self, 
        df: pd.DataFrame, 
        symbol: str,
        trades: List[Dict] = None,
        current_position: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create interactive trading chart with trade markers.
        
        Args:
            df: OHLCV dataframe
            symbol: Trading pair (e.g., 'BTC/USDT')
            trades: List of executed trades
            current_position: Currently open position (if any)
        
        Returns:
            Plotly Figure object
        """
        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price Action', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # 1. Candlestick Chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ),
            row=1, col=1
        )
        
        # 2. Volume Bars
        colors = ['#00ff00' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ff0000' 
                  for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # 3. Add Trade Markers
        if trades:
            self._add_trade_markers(fig, trades, df)
        
        # 4. Add Current Position Lines (SL/TP)
        if current_position:
            self._add_position_lines(fig, current_position, df)
        
        # 5. Styling
        fig.update_layout(
            template=self.theme,
            title=f"{symbol} - Live Paper Trading",
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
        
    def _add_trade_markers(self, fig: go.Figure, trades: List[Dict], df: pd.DataFrame):
        """Add entry/exit markers to the chart"""
        entry_times = []
        entry_prices = []
        entry_labels = []
        
        exit_times = []
        exit_prices = []
        exit_labels = []
        
        for trade in trades:
            action = trade.get('action')
            timestamp = trade.get('time')
            price = trade.get('price', 0)
            pnl = trade.get('pnl', 0)
            
            if action == 'BUY':
                entry_times.append(timestamp)
                entry_prices.append(price)
                entry_labels.append(f"LONG Entry<br>${price:,.2f}")
                
            elif action == 'SELL':
                if pnl != 0:  # Exit trade
                    exit_times.append(timestamp)
                    exit_prices.append(price)
                    pnl_emoji = "🟢" if pnl > 0 else "🔴"
                    exit_labels.append(f"{pnl_emoji} Exit<br>${price:,.2f}<br>P&L: ${pnl:,.2f}")
                else:  # Short entry
                    entry_times.append(timestamp)
                    entry_prices.append(price)
                    entry_labels.append(f"SHORT Entry<br>${price:,.2f}")
        
        # Add LONG/SHORT entry markers (green triangles)
        if entry_times:
            fig.add_trace(
                go.Scatter(
                    x=entry_times,
                    y=entry_prices,
                    mode='markers+text',
                    name='Entry',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='#00ff00',
                        line=dict(color='white', width=2)
                    ),
                    text=entry_labels,
                    textposition='top center',
                    textfont=dict(size=10, color='white'),
                    hoverinfo='text'
                ),
                row=1, col=1
            )
        
        # Add exit markers (red crosses)
        if exit_times:
            fig.add_trace(
                go.Scatter(
                    x=exit_times,
                    y=exit_prices,
                    mode='markers+text',
                    name='Exit',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='#ff0000',
                        line=dict(width=3)
                    ),
                    text=exit_labels,
                    textposition='bottom center',
                    textfont=dict(size=10, color='white'),
                    hoverinfo='text'
                ),
                row=1, col=1
            )
            
    def _add_position_lines(self, fig: go.Figure, position: Dict, df: pd.DataFrame):
        """Add SL/TP lines for current open position"""
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)
        side = position.get('side', 'LONG')
        
        # Entry line (yellow dashed)
        fig.add_hline(
            y=entry_price,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Entry: ${entry_price:,.2f}",
            annotation_position="right",
            row=1, col=1
        )
        
        # Stop Loss line (red)
        if stop_loss > 0:
            fig.add_hline(
                y=stop_loss,
                line_dash="dot",
                line_color="red",
                annotation_text=f"SL: ${stop_loss:,.2f}",
                annotation_position="right",
                row=1, col=1
            )
        
        # Take Profit line (green)
        if take_profit > 0:
            fig.add_hline(
                y=take_profit,
                line_dash="dot",
                line_color="lime",
                annotation_text=f"TP: ${take_profit:,.2f}",
                annotation_position="right",
                row=1, col=1
            )
            
    def create_simple_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create a simple candlestick chart (no trades)"""
        return self.create_trading_chart(df, symbol, trades=None, current_position=None)
