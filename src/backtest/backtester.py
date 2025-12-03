import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
from src.backtest.backtester import Backtester # <--- Yeni Modül

# Sayfa Ayarları
st.set_page_config(
    page_title="DEMIR AI - Command Center",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦅 DEMIR AI - Institutional Terminal")

# Yan Menü (Navigasyon)
page = st.sidebar.radio("Navigation", ["📡 Live Dashboard", "🧪 Backtest Lab", "🧠 Brain Health"])

def load_data():
    if os.path.exists("dashboard_data.json"):
        with open("dashboard_data.json", 'r') as f:
            return json.load(f)
    return {}

if page == "📡 Live Dashboard":
    # --- CANLI İZLEME EKRANI (Eski Kodun Aynısı) ---
    if st.button('🔄 Refresh'):
        st.rerun()
        
    data = load_data()
    if not data:
        st.info("Waiting for data stream...")
    else:
        main_info = list(data.values())[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DXY", f"{main_info.get('dxy', 0):.2f}")
        c2.metric("VIX", f"{main_info.get('vix', 0):.2f}")
        c3.metric("BTC Price", f"${main_info.get('price', 0):,.2f}")
        
        dec = main_info.get('ai_decision', 'NEUTRAL')
        color = "normal"
        if dec == "BUY": color = "inverse"
        elif dec == "SELL": color = "off"
        c4.metric("AI Decision", dec, f"{main_info.get('ai_confidence', 0):.1f}%", delta_color=color)
        
        st.dataframe(pd.DataFrame(data.values()))

elif page == "🧪 Backtest Lab":
    # --- YENİ TEST EKRANI ---
    st.header("⏳ Time Machine (Strategy Verification)")
    st.caption("Yapay zekanın geçmiş performansını simüle et.")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    with col2:
        days = st.slider("Lookback Period (Days)", 7, 90, 30)
        
    if st.button("🚀 Run Simulation"):
        with st.spinner("AI is traveling back in time... Calculating LSTM predictions..."):
            # Backtesti çalıştır (Async olduğu için wrapper lazım)
            async def run_test():
                bt = Backtester()
                return await bt.run_backtest(symbol, days)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_test())
                
                if "error" in result:
                    st.error(f"Backtest Failed: {result['error']}")
                else:
                    # Sonuçları Göster
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Net ROI", f"{result['roi']:.2f}%", delta_color="normal")
                    m2.metric("Win Rate", f"{result['win_rate']:.1f}%")
                    m3.metric("Total Trades", result['total_trades'])
                    
                    st.success(f"Final Balance: ${result['final_balance']:.2f} (Started with $10,000)")
                    
                    # İşlem Geçmişi
                    if result['trades']:
                        trades_df = pd.DataFrame(result['trades'])
                        st.subheader("Trade Logs")
                        st.dataframe(trades_df)
                        
                        # Basit Grafik
                        st.line_chart(trades_df['price'])
            except Exception as e:
                st.error(f"System Error: {e}")

elif page == "🧠 Brain Health":
    st.info("Model Status and Neural Network metrics will be displayed here.")
