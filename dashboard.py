import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
from src.backtest.backtester import Backtester # <--- Yeni Modül İmportu

# Sayfa Ayarları
st.set_page_config(
    page_title="DEMIR AI - Command Center",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık ve Stil
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;} 
    .stMetric {text-align: center;}
</style>
""", unsafe_allow_html=True)

st.title("🦅 DEMIR AI - Institutional Trading Terminal")

# --- YAN MENÜ (NAVİGASYON) ---
# Burası yeni eklendi, sayfalar arası geçiş sağlar.
page = st.sidebar.radio("System Modules", ["📡 Live Dashboard", "🧪 Backtest Lab (Time Machine)"])

# Veri Okuma Fonksiyonu (Hata Korumalı)
def load_data():
    if os.path.exists("dashboard_data.json"):
        try:
            with open("dashboard_data.json", 'r') as f:
                content = f.read()
                if not content: return {} 
                return json.loads(content)
        except:
            return {}
    return {}

# ==========================================
# SAYFA 1: CANLI İZLEME (Mevcut Kodlar)
# ==========================================
if page == "📡 Live Dashboard":
    st.caption("Deep Learning (LSTM) • Global Macro Fusion • Real-Time Analysis")
    
    if st.button('🔄 Refresh System Data'):
        st.rerun()

    data = load_data()

    if not data:
        st.info("📡 System is initializing... Waiting for Data Fusion (Crypto + Macro)...")
        time.sleep(2)
        st.rerun()
    else:
        # BTC Verisini Bul
        btc_data = data.get("BTC/USDT", {})
        if not btc_data:
            main_info = list(data.values())[0] # BTC yoksa ilkini al
            display_symbol = main_info['symbol']
        else:
            main_info = btc_data
            display_symbol = "BTC/USDT"
        
        # --- ÜST PANEL ---
        st.markdown("### 🌍 Global Market Pulse")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            val = main_info.get('dxy', 0)
            st.metric("🇺🇸 DXY (Dollar Index)", f"{val:.2f}" if val > 0 else "Loading...")
        
        with c2:
            val = main_info.get('vix', 0)
            st.metric("😨 VIX (Fear Index)", f"{val:.2f}" if val > 0 else "Loading...")
            
        with c3:
            price = main_info.get('price', 0)
            st.metric(f"₿ {display_symbol} Price", f"${price:,.2f}")

        with c4:
            conf = main_info.get('ai_confidence', 0)
            decision = main_info.get('ai_decision', 'NEUTRAL')
            color = "off"
            if decision == "BUY": color = "normal" 
            elif decision == "SELL": color = "inverse"
            st.metric("🧠 AI Decision (LSTM)", decision, f"{conf:.1f}% Conviction", delta_color=color)

        st.markdown("---")

        # --- ORTA PANEL ---
        st.markdown("### 📊 Asset Analysis Board")
        df_display = pd.DataFrame(data.values())
        
        st.dataframe(
            df_display[['symbol', 'price', 'ai_decision', 'ai_confidence', 'rsi', 'macd', 'trend', 'volatility']],
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
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**LSTM Model Prediction:**")
                    if info['ai_decision'] == "BUY": st.success(f"BULLISH ({info['ai_confidence']:.1f}%)")
                    elif info['ai_decision'] == "SELL": st.error(f"BEARISH ({info['ai_confidence']:.1f}%)")
                    else: st.warning("NEUTRAL / UNCERTAIN")
                with col2:
                    factors = []
                    if info.get('dxy', 0) > 104: factors.append("⚠️ Strong Dollar (DXY > 104).")
                    if info.get('vix', 0) > 20: factors.append("⚠️ High Market Fear (VIX > 20).")
                    if info['rsi'] < 30: factors.append("✅ RSI Oversold (< 30).")
                    if info['trend'] == "UP": factors.append("✅ Trend is Bullish.")
                    
                    if not factors: st.write("Market is choppy. AI is waiting.")
                    else: 
                        for f in factors: st.write(f"- {f}")
        
        st.caption(f"Last Update: {main_info.get('timestamp', 'Unknown')}")

# ==========================================
# SAYFA 2: BACKTEST LAB (YENİ!)
# ==========================================
elif page == "🧪 Backtest Lab (Time Machine)":
    st.header("⏳ Time Machine Simulation")
    st.markdown("""
    Bu modül, **LSTM Yapay Zeka Modelini** geçmiş veriler üzerinde çalıştırır.
    *"Eğer bu botu geçen ay kullansaydım ne olurdu?"* sorusunun cevabını verir.
    """)
    
    # Ayarlar
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox("Select Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT"])
    with col2:
        days = st.slider("Lookback Period (Days)", 7, 60, 30)
    with col3:
        start_bal = st.number_input("Starting Balance ($)", value=10000)
        
    if st.button("🚀 START SIMULATION"):
        with st.spinner(f"AI is traveling {days} days back in time... Downloading Data & Calculating..."):
            
            # Async fonksiyonu Streamlit içinde çalıştırmak için wrapper
            async def run_test():
                bt = Backtester(initial_balance=start_bal)
                return await bt.run_backtest(symbol, days)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_test())
                
                if "error" in result:
                    st.error(f"Simulation Failed: {result['error']}")
                else:
                    # --- SONUÇ KARTLARI ---
                    st.success("Simulation Complete!")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    roi_color = "normal" if result['roi'] >= 0 else "inverse"
                    m1.metric("Net ROI", f"{result['roi']:.2f}%", delta_color=roi_color)
                    m2.metric("Win Rate", f"{result['win_rate']:.1f}%")
                    m3.metric("Total Trades", result['total_trades'])
                    
                    pnl = result['final_balance'] - start_bal
                    m4.metric("Net Profit", f"${pnl:,.2f}", delta_color="normal" if pnl >= 0 else "inverse")
                    
                    st.metric("Final Wallet Balance", f"${result['final_balance']:,.2f}")
                    
                    # --- GRAFİK VE TABLO ---
                    if result['trades']:
                        trades_df = pd.DataFrame(result['trades'])
                        
                        st.subheader("📈 Portfolio Growth Curve")
                        # Sadece bakiye değişimini çiz
                        if 'balance' in trades_df.columns:
                            chart_data = trades_df.set_index('time')['balance']
                            st.line_chart(chart_data)
                        
                        st.subheader("📜 Transaction History")
                        st.dataframe(trades_df, use_container_width=True)
                    else:
                        st.warning("No trades were executed in this period. The AI was too cautious.")
                        
            except Exception as e:
                st.error(f"System Error during Backtest: {e}")
