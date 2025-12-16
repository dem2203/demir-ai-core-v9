"""
DEMIR AI v24.0 - SUPERHUMAN TRADING TERMINAL
=============================================
Modern, Clean, AI-Focused Dashboard

Sayfalar:
1. 🔮 Predictive Intel (Ana sayfa - Önceden Uyarı)
2. 📊 Market Pulse (Genel görünüm)
3. 🎯 Trade Signals (Aktif sinyaller)
4. 💼 Portfolio (Pozisyonlar)
5. 🧠 AI Brain (Beyin durumu)
6. 📈 Charts (Grafikler)
7. ⚙️ Settings (Ayarlar)
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime

# Page config - MUST BE FIRST
st.set_page_config(
    page_title="DEMIR AI - Superhuman Trading",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- IMPORTS ---
from src.config.settings import Config

# --- DARK THEME CSS ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2d2d44;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .signal-card-buy {
        background: linear-gradient(135deg, #0a2a1a 0%, #1a4a2a 100%);
        border: 2px solid #00ff88;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .signal-card-sell {
        background: linear-gradient(135deg, #2a0a0a 0%, #4a1a1a 100%);
        border: 2px solid #ff4444;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Glow effects */
    .glow-green { text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88; }
    .glow-red { text-shadow: 0 0 10px #ff4444, 0 0 20px #ff4444; }
    .glow-blue { text-shadow: 0 0 10px #4488ff, 0 0 20px #4488ff; }
    
    /* Headers */
    h1, h2, h3 { color: #ffffff !important; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_json(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None

def get_signal_emoji(decision):
    if decision == "BUY":
        return "🟢"
    elif decision == "SELL":
        return "🔴"
    else:
        return "⚪"

def format_price(price):
    if price >= 1000:
        return f"${price:,.0f}"
    elif price >= 1:
        return f"${price:.2f}"
    else:
        return f"${price:.4f}"

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="font-size: 2.5rem; margin: 0;">🦅</h1>
    <h2 style="margin: 5px 0; font-size: 1.5rem;">DEMIR AI</h2>
    <p style="color: #888; font-size: 0.9rem;">Superhuman Trading v24.0</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🔮 Predictive Intel",
        "📊 Market Pulse", 
        "🎯 Trade Signals",
        "💼 Portfolio",
        "🧠 AI Brain",
        "📈 Charts",
        "⚙️ Settings"
    ],
    label_visibility="collapsed"
)

# System Status
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

data = load_json("dashboard_data.json")
if data:
    st.sidebar.success("🟢 AI ONLINE")
    st.sidebar.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.sidebar.error("🔴 WAITING DATA")

# =====================================================
# PAGE 1: PREDICTIVE INTEL (Ana Sayfa)
# =====================================================
if page == "🔮 Predictive Intel":
    st.markdown("""
    <h1 style="text-align: center; font-size: 3rem; margin-bottom: 0;">
        🔮 Predictive Intelligence
    </h1>
    <p style="text-align: center; color: #888; font-size: 1.2rem;">
        Hareket olmadan ÖNCE tespit • Leading Indicators • Entry/SL/TP
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not data:
        st.warning("📡 Waiting for AI data stream...")
        st.stop()
    
    # Get derivatives data for predictive indicators
    main_symbol = Config.TARGET_COINS[0] if Config.TARGET_COINS else "BTC/USDT"
    main_data = data.get(main_symbol, {})
    derivatives = main_data.get('derivatives', {})
    sentiment = main_data.get('sentiment_data', {})
    
    # === PREDICTIVE INDICATORS ROW ===
    st.markdown("### 📊 Leading Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        funding_rate = derivatives.get('funding_rate', 0)
        fr_pct = funding_rate * 100 if funding_rate else 0
        fr_color = "🔴" if fr_pct > 0.05 else "🟢" if fr_pct < -0.03 else "⚪"
        st.metric("Funding Rate", f"{fr_pct:.4f}%", delta=f"{fr_color} {'Extreme Long' if fr_pct > 0.05 else 'Extreme Short' if fr_pct < -0.03 else 'Normal'}")
    
    with col2:
        ls_ratio = derivatives.get('long_short_ratio', 1.0)
        ls_color = "🔴" if ls_ratio > 2.0 else "🟢" if ls_ratio < 0.5 else "⚪"
        st.metric("Long/Short Ratio", f"{ls_ratio:.2f}", delta=f"{ls_color} {'Herkes Long!' if ls_ratio > 2.0 else 'Herkes Short!' if ls_ratio < 0.5 else 'Dengeli'}")
    
    with col3:
        oi = derivatives.get('open_interest', 0)
        st.metric("Open Interest", f"${oi/1e9:.2f}B" if oi > 1e9 else f"${oi/1e6:.0f}M")
    
    with col4:
        fg_index = sentiment.get('fear_greed_index', 50) if sentiment else 50
        fg_label = "😱 Extreme Fear" if fg_index <= 20 else "😰 Fear" if fg_index <= 40 else "😐 Neutral" if fg_index <= 60 else "😁 Greed" if fg_index <= 80 else "🤑 Extreme Greed"
        st.metric("Fear & Greed", f"{fg_index}/100", delta=fg_label)
    
    st.markdown("---")
    
    # === PREDICTIVE SIGNALS ===
    st.markdown("### 🎯 Active Predictive Signals")
    
    # Check for predictive conditions
    signals_found = []
    
    # Funding Rate Signal
    if fr_pct >= 0.05:
        signals_found.append({
            'type': 'SHORT',
            'reason': f'Funding Rate %{fr_pct:.3f} - Aşırı long birikimi',
            'confidence': min(70 + int(fr_pct * 100), 90),
            'action': 'Short pozisyon veya long kapat'
        })
    elif fr_pct <= -0.03:
        signals_found.append({
            'type': 'LONG',
            'reason': f'Funding Rate %{fr_pct:.3f} - Aşırı short birikimi',
            'confidence': min(65 + int(abs(fr_pct) * 100), 85),
            'action': 'Long pozisyon veya short kapat'
        })
    
    # L/S Ratio Signal
    if ls_ratio >= 2.5:
        signals_found.append({
            'type': 'SHORT',
            'reason': f'L/S Ratio {ls_ratio:.2f} - Herkes long, düzeltme riski',
            'confidence': 70,
            'action': 'Short squeeze dikkat'
        })
    elif ls_ratio <= 0.4:
        signals_found.append({
            'type': 'LONG',
            'reason': f'L/S Ratio {ls_ratio:.2f} - Herkes short, squeeze potansiyeli',
            'confidence': 70,
            'action': 'Long squeeze potansiyeli'
        })
    
    # Fear & Greed Signal
    if fg_index <= 20:
        signals_found.append({
            'type': 'LONG',
            'reason': f'Extreme Fear ({fg_index}/100) - Tarihsel dip sinyali',
            'confidence': 65,
            'action': 'Kademeli alım düşün'
        })
    elif fg_index >= 80:
        signals_found.append({
            'type': 'SHORT',
            'reason': f'Extreme Greed ({fg_index}/100) - Tepe riski',
            'confidence': 60,
            'action': 'Kar al veya hedge'
        })
    
    if signals_found:
        for sig in signals_found:
            card_class = "signal-card-buy" if sig['type'] == 'LONG' else "signal-card-sell"
            emoji = "🟢" if sig['type'] == 'LONG' else "🔴"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h3 style="margin: 0;">{emoji} PREDICTIVE: {sig['type']}</h3>
                <p style="font-size: 1.1rem; margin: 10px 0;">{sig['reason']}</p>
                <p style="color: #888;">📊 Confidence: {sig['confidence']}% | ➡️ {sig['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("✅ Şu an extreme sinyal yok. Piyasa dengeli.")
    
    st.markdown("---")
    
    # === MARKET SNAPSHOT ===
    st.markdown("### 📈 Quick Market View")
    
    cols = st.columns(len(data))
    for i, (symbol, info) in enumerate(data.items()):
        with cols[i]:
            price = info.get('price', 0)
            decision = info.get('ai_decision', 'NEUTRAL')
            confidence = info.get('ai_confidence', 0)
            emoji = get_signal_emoji(decision)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1.2rem;">{symbol}</h3>
                <h2 style="margin: 5px 0;">{format_price(price)}</h2>
                <p style="font-size: 1.5rem; margin: 0;">{emoji} {decision}</p>
                <p style="color: #888; margin: 0;">Confidence: {confidence:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# PAGE 2: MARKET PULSE
# =====================================================
elif page == "📊 Market Pulse":
    st.markdown("""
    <h1 style="text-align: center;">📊 Market Pulse</h1>
    <p style="text-align: center; color: #888;">Real-time market overview</p>
    """, unsafe_allow_html=True)
    
    if not data:
        st.warning("📡 Waiting for data...")
        st.stop()
    
    # Coin cards in grid
    for symbol, info in data.items():
        with st.expander(f"{get_signal_emoji(info.get('ai_decision', 'NEUTRAL'))} **{symbol}** - {format_price(info.get('price', 0))}", expanded=True):
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Signal", info.get('ai_decision', 'N/A'))
            col2.metric("Confidence", f"{info.get('ai_confidence', 0):.0f}%")
            col3.metric("Regime", info.get('regime', 'N/A'))
            col4.metric("Fractal", f"{info.get('fractal_score', 0):.0f}%")
            
            st.caption(f"💡 {info.get('reason', 'No reason available')}")

# =====================================================
# PAGE 3: TRADE SIGNALS
# =====================================================
elif page == "🎯 Trade Signals":
    st.markdown("""
    <h1 style="text-align: center;">🎯 Active Trade Signals</h1>
    <p style="text-align: center; color: #888;">Entry • Stop Loss • Take Profit</p>
    """, unsafe_allow_html=True)
    
    if not data:
        st.warning("📡 Waiting for data...")
        st.stop()
    
    for symbol, info in data.items():
        decision = info.get('ai_decision', 'NEUTRAL')
        if decision in ['BUY', 'SELL']:
            
            # Get SL/TP data
            smart_levels = info.get('smart_levels', {})
            entry = info.get('price', 0)
            sl = smart_levels.get('stop_loss', entry * 0.97)
            tp1 = smart_levels.get('tp1', entry * 1.02)
            tp2 = smart_levels.get('tp2', entry * 1.04)
            tp3 = smart_levels.get('tp3', entry * 1.06)
            
            card_class = "signal-card-buy" if decision == "BUY" else "signal-card-sell"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h2 style="margin: 0;">{get_signal_emoji(decision)} {symbol} - {decision}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("💰 Entry", format_price(entry))
            c2.metric("🛡️ Stop Loss", format_price(sl))
            c3.metric("🎯 TP1", format_price(tp1))
            c4.metric("🎯 TP2", format_price(tp2))
            c5.metric("🎯 TP3", format_price(tp3))
            
            st.caption(f"📊 Confidence: {info.get('ai_confidence', 0):.0f}% | 💡 {info.get('reason', '')[:100]}...")
            st.markdown("---")
    
    # No signals case
    buy_sell_count = sum(1 for s, i in data.items() if i.get('ai_decision') in ['BUY', 'SELL'])
    if buy_sell_count == 0:
        st.info("⏳ No active trade signals. AI is scanning for opportunities...")

# =====================================================
# PAGE 4: PORTFOLIO
# =====================================================
elif page == "💼 Portfolio":
    st.markdown("""
    <h1 style="text-align: center;">💼 Portfolio Tracker</h1>
    <p style="text-align: center; color: #888;">Advisory positions • Paper trading</p>
    """, unsafe_allow_html=True)
    
    portfolio = load_json("portfolio.json")
    
    if not portfolio:
        st.info("📭 No portfolio data yet. Waiting for first trade...")
    else:
        balance = portfolio.get('balance', 10000)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("💵 Balance", f"${balance:,.2f}")
        col2.metric("📊 Positions", len(portfolio.get('positions', {})))
        col3.metric("📜 Total Trades", len(portfolio.get('history', [])))
        
        # Positions
        st.markdown("### Open Positions")
        if portfolio.get('positions'):
            for sym, pos in portfolio['positions'].items():
                with st.expander(f"📍 {sym}", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Entry", format_price(pos.get('entry_price', 0)))
                    c2.metric("Size", f"{pos.get('amount', 0):.4f}")
                    c3.metric("Side", pos.get('side', 'LONG'))
        else:
            st.info("No open positions")

# =====================================================
# PAGE 5: AI BRAIN
# =====================================================
elif page == "🧠 AI Brain":
    st.markdown("""
    <h1 style="text-align: center;">🧠 AI Brain Monitor</h1>
    <p style="text-align: center; color: #888;">Neural network • RL Agent • Attention</p>
    """, unsafe_allow_html=True)
    
    if not data:
        st.warning("📡 Waiting for brain data...")
        st.stop()
    
    main_symbol = Config.TARGET_COINS[0] if Config.TARGET_COINS else "BTC/USDT"
    info = data.get(main_symbol, {})
    brain_state = info.get('brain_state', {})
    
    # Brain status
    col1, col2 = st.columns([1, 2])
    
    with col1:
        decision = info.get('ai_decision', 'NEUTRAL')
        color = "#00ff88" if decision == "BUY" else "#ff4444" if decision == "SELL" else "#888888"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #1a1a2e, #2d2d44); border-radius: 20px; border: 2px solid {color};">
            <h1 style="font-size: 4rem; margin: 0; color: {color};">{decision}</h1>
            <p style="color: #888;">Current AI Decision</p>
            <h2 style="margin: 10px 0;">{info.get('ai_confidence', 0):.0f}%</h2>
            <p style="color: #888;">Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔬 Attention Weights")
        
        if brain_state:
            import plotly.graph_objects as go
            
            categories = ['Technical', 'Pattern', 'LSTM', 'HTF Trend', 'On-Chain']
            values = [
                brain_state.get('tech_attention', 0.1),
                brain_state.get('pattern_attention', 0.1),
                brain_state.get('lstm_attention', 0.1),
                brain_state.get('htf_attention', 0.1),
                brain_state.get('onchain_attention', 0.1)
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(0, 255, 136, 0.3)',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 0.5], gridcolor='#333'),
                    angularaxis=dict(gridcolor='#333'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fff'),
                showlegend=False,
                margin=dict(l=60, r=60, t=40, b=40),
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brain state not initialized yet")

# =====================================================
# PAGE 6: CHARTS
# =====================================================
elif page == "📈 Charts":
    st.markdown("""
    <h1 style="text-align: center;">📈 Live Charts</h1>
    """, unsafe_allow_html=True)
    
    try:
        from src.core.chart_visualizer import ChartVisualizer
        from src.data_ingestion.connectors.binance_connector import BinanceConnector
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "LTC/USDT"])
        with col2:
            timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
        
        connector = BinanceConnector()
        df = connector.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        
        if not df.empty:
            visualizer = ChartVisualizer()
            fig = visualizer.create_trading_chart(df, symbol, [], None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No chart data available")
    except Exception as e:
        st.error(f"Chart error: {e}")

# =====================================================
# PAGE 7: SETTINGS
# =====================================================
elif page == "⚙️ Settings":
    st.markdown("""
    <h1 style="text-align: center;">⚙️ Settings & Debug</h1>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔑 API Keys Status")
    
    keys = {
        "Binance": "✅" if Config.BINANCE_API_KEY else "❌",
        "Telegram": "✅" if Config.TELEGRAM_TOKEN else "❌",
        "Gemini AI": "✅" if Config.GEMINI_API_KEY else "❌",
        "OpenAI": "✅" if Config.OPENAI_API_KEY else "❌",
        "FRED (Macro)": "✅" if Config.FRED_API_KEY else "❌"
    }
    
    for key, status in keys.items():
        st.write(f"{status} **{key}**")
    
    st.markdown("---")
    st.markdown("### 📄 Raw Data")
    
    with st.expander("Dashboard Data"):
        st.json(data)
    
    with st.expander("Portfolio Data"):
        st.json(load_json("portfolio.json"))

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("🦅 DEMIR AI v24.0")
st.sidebar.caption("Built with 🔥 by Demir")
