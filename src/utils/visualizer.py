import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class MarketVisualizer:
    """
    PROFESYONEL GRAFİK MOTORU
    TradingView benzeri mum grafikleri ve indikatör panelleri oluşturur.
    """
    
    @staticmethod
    def create_advanced_chart(df: pd.DataFrame, symbol: str, trade_history: list = None):
        """
        Fiyat, Hacim, RSI ve Alım-Satım noktalarını içeren detaylı grafik çizer.
        """
        if df is None or df.empty: return None
        
        # Son 100 mumu al (Grafik çok sıkışmasın)
        df_view = df.tail(100)
        
        # Alt alta 3 panel oluştur: Fiyat, Hacim, RSI
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{symbol} Price Action & Signals", "Volume", "RSI Momentum")
        )

        # --- 1. PANEL: MUM GRAFİĞİ ---
        fig.add_trace(go.Candlestick(
            x=df_view['timestamp'],
            open=df_view['open'],
            high=df_view['high'],
            low=df_view['low'],
            close=df_view['close'],
            name='Price'
        ), row=1, col=1)

        # Hareketli Ortalamalar (Varsa)
        if 'vwap' in df_view.columns:
            fig.add_trace(go.Scatter(x=df_view['timestamp'], y=df_view['vwap'], line=dict(color='orange', width=1), name='VWAP'), row=1, col=1)

        # Ichimoku Bulutu (Varsa)
        if 'span_a' in df_view.columns and 'span_b' in df_view.columns:
            fig.add_trace(go.Scatter(
                x=df_view['timestamp'], y=df_view['span_a'],
                line=dict(color='rgba(0,0,0,0)'), showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_view['timestamp'], y=df_view['span_b'],
                fill='tonexty', fillcolor='rgba(0, 250, 154, 0.1)',
                line=dict(color='rgba(0,0,0,0)'), name='Ichimoku Cloud'
            ), row=1, col=1)

        # --- ALIM / SATIM İŞARETLERİ ---
        if trade_history:
            trades = pd.DataFrame(trade_history)
            # Zaman formatını eşle
            # (Bu kısım dashboard'dan gelen verinin formatına göre ayarlanmalı)
            
            buys = trades[trades['action'] == 'BUY']
            sells = trades[trades['action'] == 'SELL']
            
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys['time'], y=buys['price'],
                    mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='BUY Signal'
                ), row=1, col=1)
                
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells['time'], y=sells['price'],
                    mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='SELL Signal'
                ), row=1, col=1)

        # --- 2. PANEL: HACİM ---
        colors = ['red' if row['open'] > row['close'] else 'green' for i, row in df_view.iterrows()]
        fig.add_trace(go.Bar(
            x=df_view['timestamp'], y=df_view['volume'],
            marker_color=colors, name='Volume'
        ), row=2, col=1)

        # --- 3. PANEL: RSI ---
        fig.add_trace(go.Scatter(x=df_view['timestamp'], y=df_view['rsi'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
        
        # RSI Referans Çizgileri
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # --- GÖRSEL AYARLAR ---
        fig.update_layout(
            template="plotly_dark",
            height=800,
            xaxis_rangeslider_visible=False,
            title_text=f"Institutional Analysis: {symbol}",
            font=dict(family="Courier New, monospace")
        )
        
        return fig
