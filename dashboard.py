import streamlit as st
import pandas as pd
import json
import time
import os

# Sayfa Ayarları
st.set_page_config(
    page_title="DEMIR AI - Strategic Command",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık
st.title("🧠 DEMIR AI - Intelligent Market Analysis")
st.markdown("### `AI Co-Pilot & Strategic Advisor`")

# Veri Okuma Fonksiyonu
def load_data():
    if os.path.exists("dashboard_data.json"):
        with open("dashboard_data.json", 'r') as f:
            return json.load(f)
    return {}

# Otomatik Yenileme Butonu
if st.button('🔄 Refresh Data'):
    st.rerun()

# Ana Veri Döngüsü
data = load_data()

if not data:
    st.warning("⏳ Waiting for AI Analysis... (Bot is thinking)")
else:
    # 1. Genel Piyasa Özeti (Metrikler)
    st.markdown("---")
    cols = st.columns(len(data))
    
    for idx, (symbol, info) in enumerate(data.items()):
        with cols[idx]:
            # Renk Ayarı
            color = "normal"
            if info['ai_decision'] == "BUY": color = "inverse"
            elif info['ai_decision'] == "SELL": color = "off"
            
            st.metric(
                label=f"{symbol}",
                value=f"${info['price']:.2f}",
                delta=f"{info['ai_decision']} ({info['ai_confidence']:.1f}%)" if info['ai_decision'] != "NEUTRAL" else "NEUTRAL"
            )

    # 2. Detaylı Analiz Tablosu
    st.markdown("### 📊 Live AI Analysis Board")
    
    # Veriyi Tablo Formatına Çevir
    df_display = pd.DataFrame(data.values())
    
    # Sütunları düzenle ve renklendir
    def highlight_signal(val):
        color = 'white'
        if val == 'BUY': color = '#2ecc71' # Yeşil
        elif val == 'SELL': color = '#e74c3c' # Kırmızı
        return f'background-color: {color}; color: black; font-weight: bold'

    st.dataframe(
        df_display[['symbol', 'price', 'ai_decision', 'ai_confidence', 'rsi', 'macd', 'adx', 'trend', 'volatility']],
        use_container_width=True,
        column_config={
            "ai_confidence": st.column_config.ProgressColumn(
                "AI Confidence",
                help="Yapay Zeka ne kadar emin?",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        }
    )

    # 3. İçgörü ve Nedenler (Why?)
    st.markdown("### 💡 AI Reasoning & Internal Factors")
    
    for symbol, info in data.items():
        with st.expander(f"🔍 Deep Dive: {symbol} Analysis"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.write(f"**Trend Status:** {info['trend']}")
                if info['trend'] == "UP":
                    st.success("Trend is Bullish (Above VWAP)")
                else:
                    st.error("Trend is Bearish (Below VWAP)")
            
            with c2:
                st.write(f"**RSI Momentum:** {info['rsi']:.2f}")
                if info['rsi'] < 30: st.warning("Oversold (Buy Zone?)")
                elif info['rsi'] > 70: st.warning("Overbought (Sell Zone?)")
                else: st.info("Neutral Zone")
                
            with c3:
                st.write(f"**Volatility:** {info['volatility']}")
                if info['volatility'] == "LOW":
                    st.info("Market is squeezing (Prepare for breakout)")
                else:
                    st.warning("High Volatility (Be careful)")

# Alt Bilgi
st.markdown("---")
st.caption("Demir AI v10.1 | Running on Railway | Not Financial Advice")