import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
from src.backtest.backtester import Backtester

# Sayfa Ayarları
st.set_page_config(page_title="DEMIR AI", page_icon="🦅", layout="wide", initial_sidebar_state="expanded")

# Stil
st.markdown("""<style>.metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;} .stMetric {text-align: center;}</style>""", unsafe_allow_html=True)

st.title("🦅 DEMIR AI - Institutional Trading Terminal")

# Yan Menü
page = st.sidebar.radio("Navigation", ["📡 Live Dashboard", "🧪 Backtest Lab"])

def load_data():
    if os.path.exists("dashboard_data.json"):
        try:
            with open("dashboard_data.json", 'r') as f:
                content = f.read()
                if not content: return {} 
                return json.loads(content)
        except: return {}
    return {}

# --- SAYFA 1: CANLI DASHBOARD ---
if page == "📡 Live Dashboard":
    if st.button('🔄 Refresh System Data'): st.rerun()
    data = load_data()
    
    if not data:
        st.info("📡 System is initializing... Waiting for Data Fusion...")
    else:
        btc_data = data.get("BTC/USDT", list(data.values())[0])
        
        st.markdown("### 🌍 Global Market Pulse")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🇺🇸 DXY", f"{btc_data.get('dxy', 0):.2f}")
        c2.metric("😨 VIX", f"{btc_data.get('vix', 0):.2f}")
        c3.metric(f"₿ {btc_data['symbol']} Price", f"${btc_data['price']:,.2f}")
        
        dec = btc_data.get('ai_decision', 'NEUTRAL')
        color = "normal"
        if dec == "BUY": color = "inverse"
        elif dec == "SELL": color = "off"
        c4.metric("🧠 AI Decision (LSTM)", dec, f"{btc_data.get('ai_confidence', 0):.1f}% Conviction", delta_color=color)
        
        st.markdown("---")
        st.markdown("### 📊 Asset Analysis Board")
        st.dataframe(pd.DataFrame(data.values())[['symbol', 'price', 'ai_decision', 'ai_confidence', 'rsi', 'trend', 'volatility']], use_container_width=True)
        
        st.markdown("### 🤖 AI Reasoning Engine")
        for symbol, info in data.items():
            with st.expander(f"🔍 Inspect Logic: {symbol}", expanded=False):
                st.write(f"**LSTM Prediction:** {info.get('ai_decision')} ({info.get('ai_confidence'):.1f}%)")
                st.write(f"**Market Factors:** DXY: {info.get('dxy')}, VIX: {info.get('vix')}, Trend: {info.get('trend')}")

# --- SAYFA 2: BACKTEST LAB ---
elif page == "🧪 Backtest Lab":
    st.header("⏳ Time Machine (Strategy Verification)")
    st.caption("Yapay zekanın geçmiş 2 aydaki performansını simüle et.")
    
    col1, col2, col3 = st.columns(3)
    with col1: symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
    with col2: days = st.slider("Lookback Period (Days)", 7, 60, 30)
    with col3: start_bal = st.number_input("Start Balance ($)", value=10000)
        
    if st.button("🚀 Run Simulation"):
        with st.spinner(f"AI is traveling back in time ({days} days)... Calculating LSTM predictions..."):
            async def run_test():
                bt = Backtester(initial_balance=start_bal)
                return await bt.run_backtest(symbol, days)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_test())
                
                if "error" in result:
                    st.error(f"Backtest Failed: {result['error']}")
                else:
                    # Sonuç Metrikleri
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Final Balance", f"${result['final_balance']:,.2f}", f"{result['roi']:.2f}%")
                    m2.metric("Total Trades", result['total_trades'])
                    m3.metric("Win Rate", f"{result['win_rate']:.1f}%")
                    m4.metric("Net Profit", f"${result['final_balance'] - start_bal:,.2f}")
                    
                    # Grafik
                    if result['trades']:
                        trades_df = pd.DataFrame(result['trades'])
                        
                        st.subheader("📈 Balance Growth")
                        # Sadece satış (kar realizasyonu) anlarını grafiğe dök
                        balance_curve = trades_df[trades_df['action'] == 'SELL'][['time', 'balance']]
                        if not balance_curve.empty:
                            balance_curve.set_index('time', inplace=True)
                            st.line_chart(balance_curve)
                        
                        st.subheader("📜 Trade Logs")
                        st.dataframe(trades_df)
                    else:
                        st.warning("No trades were executed in this period (AI was too cautious).")
                        
            except Exception as e:
                st.error(f"System Error: {e}")
