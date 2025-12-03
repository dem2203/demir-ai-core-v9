import streamlit as st
import pandas as pd
import json
import os
import time

# Sayfa Ayarları
st.set_page_config(
    page_title="DEMIR AI - Strategic Command",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Başlık ve Stil
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .stMetric {text-align: center;}
</style>
""", unsafe_allow_html=True)

st.title("🦅 DEMIR AI - Institutional Trading Terminal")
st.caption("Deep Learning (LSTM) • Global Macro Fusion • Sentiment Free")

# Veri Okuma (Çökme Korumalı)
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

if st.button('🔄 Refresh System Data'):
    st.rerun()

data = load_data()

if not data:
    st.info("📡 System is initializing... Waiting for Data Fusion...")
    time.sleep(3)
    st.rerun()
else:
    # --- ÜST PANEL: KÜRESEL GÖSTERGELER ---
    st.markdown("### 🌍 Global Market Pulse")
    
    # DÜZELTME BURADA: Rastgele ilk veriyi alma, BTC'yi bul.
    btc_data = data.get("BTC/USDT", {})
    
    # Eğer BTC yoksa (henüz çekilmediyse) ilk veriyi al ama başlığına dikkat et
    if not btc_data:
        main_info = list(data.values())[0]
        display_symbol = main_info['symbol']
    else:
        main_info = btc_data
        display_symbol = "BTC/USDT"
    
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

    # --- ORTA PANEL: DETAYLI ANALİZ ---
    st.markdown("### 📊 Asset Analysis Board")
    
    df_display = pd.DataFrame(data.values())
    
    st.dataframe(
        df_display[['symbol', 'price', 'ai_decision', 'ai_confidence', 'rsi', 'macd', 'trend', 'volatility']],
        use_container_width=True,
        column_config={
            "ai_confidence": st.column_config.ProgressColumn(
                "AI Conviction",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        }
    )

    # --- ALT PANEL: YAPAY ZEKA GÖRÜŞÜ ---
    st.markdown("### 🤖 AI Reasoning Engine")
    
    for symbol, info in data.items():
        with st.expander(f"🔍 Inspect Logic: {symbol}", expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**LSTM Model Prediction:**")
                if info['ai_decision'] == "BUY":
                    st.success(f"BULLISH ({info['ai_confidence']:.1f}%)")
                elif info['ai_decision'] == "SELL":
                    st.error(f"BEARISH ({info['ai_confidence']:.1f}%)")
                else:
                    st.warning("NEUTRAL / UNCERTAIN")
            
            with col2:
                factors = []
                if info.get('dxy', 0) > 104: factors.append("⚠️ Strong Dollar (DXY > 104) is suppressing assets.")
                if info.get('vix', 0) > 20: factors.append("⚠️ High Market Fear (VIX > 20).")
                if info['rsi'] < 30: factors.append("✅ Technicals are Oversold (RSI < 30).")
                if info['trend'] == "UP": factors.append("✅ Trend is technically Bullish.")
                
                if not factors:
                    st.write("Market is choppy. AI is waiting for a clear setup.")
                else:
                    for f in factors:
                        st.write(f"- {f}")

    st.caption(f"Last Update: {main_info.get('timestamp', 'Unknown')}")
