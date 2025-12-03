import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
from src.backtest.backtester import Backtester
from src.config.settings import Config

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="DEMIR AI - Command Center",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Stil ---
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;} 
    .stMetric {text-align: center;}
</style>
""", unsafe_allow_html=True)

st.title("🦅 DEMIR AI - Institutional Trading Terminal")

# --- Yan Menü ---
page = st.sidebar.radio("System Modules", ["📡 Live Dashboard", "🧪 Backtest Lab (Time Machine)"])

# --- Veri Okuma (Hata Korumalı) ---
def load_data():
    if os.path.exists("dashboard_data.json"):
        try:
            with open("dashboard_data.json", 'r') as f:
                content = f.read()
                if not content: return {} 
                return json.loads(content)
        except: return {}
    return {}

# ==========================================
# SAYFA 1: CANLI İZLEME
# ==========================================
if page == "📡 Live Dashboard":
    st.caption(f"Tracking Assets: {', '.join(Config.TARGET_COINS)}")
    
    # DÜZELTME: experimental_rerun yerine rerun kullanıyoruz
    if st.button('🔄 Refresh System Data'):
        st.rerun()

    data = load_data()

    if not data:
        st.info("📡 System is initializing... Waiting for Data Fusion...")
        time.sleep(2)
        st.rerun()
    else:
        # DÜZELTME: BTC verisini bul, yoksa ilkini al
        main_symbol = Config.TARGET_COINS[0] # BTC/USDT
        btc_data = data.get(main_symbol)
        
        # Eğer BTC henüz gelmediyse (nadir olur), listedeki ilk veriyi kullan
        if btc_data:
            main_info = btc_data
            display_symbol = main_symbol
        else:
            first_key = list(data.keys())[0]
            main_info = data[first_key]
            display_symbol = main_info['symbol']
        
        # --- ÜST PANEL ---
        st.markdown("### 🌍 Global Market Pulse")
        c1, c2, c3, c4 = st.columns(4)
        
        # Stooq'tan gelen makro veriler (macro_DXY, macro_VIX)
        # Not: İsimler bazen değişebilir, güvenli get kullanalım
        dxy = main_info.get('dxy', 0)
        vix = main_info.get('vix', 0)
        
        c1.metric("🇺🇸 DXY", f"{dxy:.2f}" if dxy > 0 else "Loading...")
        c2.metric("😨 VIX", f"{vix:.2f}" if vix > 0 else "Loading...")
        c3.metric(f"₿ {display_symbol} Price", f"${main_info.get('price', 0):,.2f}")
        
        dec = main_info.get('ai_decision', 'NEUTRAL')
        conf = main_info.get('ai_confidence', 0)
        color = "normal"
        if dec == "BUY": color = "inverse"
        elif dec == "SELL": color = "off"
        
        c4.metric("🧠 AI Decision (LSTM)", dec, f"{conf:.1f}% Conviction", delta_color=color)

        st.markdown("---")

        # --- ORTA PANEL ---
        st.markdown("### 📊 Asset Analysis Board")
        df_display = pd.DataFrame(data.values())
        
        # Sütunları seç ve göster
        cols = ['symbol', 'price', 'ai_decision', 'ai_confidence', 'rsi', 'macd', 'trend', 'volatility']
        valid_cols = [c for c in cols if c in df_display.columns]
        
        st.dataframe(
            df_display[valid_cols],
            use_container_width=True,
            column_config={
                "ai_confidence": st.column_config.ProgressColumn(
                    "AI Conviction", format="%.1f%%", min_value=0, max_value=100
                ),
            }
        )

        # --- ALT PANEL ---
        st.markdown("### 🤖 AI Reasoning Engine")
        for symbol, info in data.items():
            with st.expander(f"🔍 Inspect Logic: {symbol}", expanded=False):
                st.write(f"**LSTM Prediction:** {info.get('ai_decision')} ({info.get('ai_confidence'):.1f}%)")
                c1, c2 = st.columns(2)
                c1.write(f"RSI: {info.get('rsi'):.2f}")
                c2.write(f"Trend: {info.get('trend')}")
        
        st.caption(f"Last Update: {main_info.get('timestamp', 'Unknown')}")

# ==========================================
# SAYFA 2: BACKTEST LAB
# ==========================================
elif page == "🧪 Backtest Lab (Time Machine)":
    st.header("⏳ Time Machine Simulation")
    
    col1, col2, col3 = st.columns(3)
    with col1: symbol = st.selectbox("Select Asset", Config.TARGET_COINS)
    with col2: days = st.slider("Lookback Period (Days)", 7, 60, 30)
    with col3: start_bal = st.number_input("Starting Balance ($)", value=10000)
        
    if st.button("🚀 START SIMULATION"):
        with st.spinner(f"AI is simulating {days} days history for {symbol}..."):
            async def run_test():
                bt = Backtester(initial_balance=start_bal)
                return await bt.run_backtest(symbol, days)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_test())
                if "error" in result: st.error(f"Backtest Failed: {result['error']}")
                else:
                    st.success("Simulation Complete!")
                    m1, m2, m3, m4 = st.columns(4)
                    roi = result['roi']
                    pnl = result['final_balance'] - start_bal
                    m1.metric("Net ROI", f"{roi:.2f}%", delta_color="normal" if roi >= 0 else "inverse")
                    m2.metric("Win Rate", f"{result['win_rate']:.1f}%")
                    m3.metric("Total Trades", result['total_trades'])
                    m4.metric("Net Profit", f"${pnl:,.2f}")
                    
                    if result['trades']:
                        trades_df = pd.DataFrame(result['trades'])
                        st.line_chart(trades_df[trades_df['action']=='SELL'].set_index('time')['balance'])
                        st.dataframe(trades_df)
                    else: st.warning("No trades executed.")
            except Exception as e: st.error(f"Error: {e}")
