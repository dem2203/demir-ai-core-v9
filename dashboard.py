import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
from src.backtest.backtester import Backtester
from src.config.settings import Config
from src.execution.paper_trader import PaperTrader # <-- Kağıt İşlem Modülü

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

# --- Yan Menü (GÜNCELLENDİ) ---
page = st.sidebar.radio("System Modules", [
    "📡 Live Dashboard", 
    "💼 Live Portfolio (Paper)", # <-- Yeni Sekme
    "🧪 Backtest Lab (Time Machine)"
])

# --- Veri Okuma Fonksiyonu (JSON) ---
def load_json(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                content = f.read()
                if not content: return {} 
                return json.loads(content)
        except: return {}
    return {}

# ==========================================
# SAYFA 1: CANLI PİYASA İZLEME
# ==========================================
if page == "📡 Live Dashboard":
    st.caption(f"Tracking Assets: {', '.join(Config.TARGET_COINS)}")
    
    if st.button('🔄 Refresh Market Data'):
        st.rerun()

    data = load_json("dashboard_data.json")

    if not data:
        st.info("📡 System is initializing... Waiting for Data Fusion...")
        time.sleep(2)
        st.rerun()
    else:
        # BTC Verisini Bul
        main_symbol = Config.TARGET_COINS[0] 
        btc_data = data.get(main_symbol)
        
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
        
        dxy = main_info.get('dxy', 0)
        vix = main_info.get('vix', 0)
        
        c1.metric("🇺🇸 DXY (Dollar Index)", f"{dxy:.2f}" if dxy > 0 else "Loading...")
        c2.metric("😨 VIX (Fear Index)", f"{vix:.2f}" if vix > 0 else "Loading...")
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
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**LSTM Model Prediction:**")
                    decision = info.get('ai_decision', 'NEUTRAL')
                    conf = info.get('ai_confidence', 0)
                    if decision == "BUY": st.success(f"BULLISH ({conf:.1f}%)")
                    elif decision == "SELL": st.error(f"BEARISH ({conf:.1f}%)")
                    else: st.warning("NEUTRAL / UNCERTAIN")
                
                with col2:
                    factors = []
                    if info.get('dxy', 0) > 104: factors.append("⚠️ Strong Dollar (DXY > 104).")
                    if info.get('vix', 0) > 20: factors.append("⚠️ High Market Fear (VIX > 20).")
                    if info.get('rsi', 50) < 30: factors.append("✅ RSI Oversold (< 30).")
                    
                    if not factors: st.write("Market is choppy. AI is waiting.")
                    else: 
                        for f in factors: st.write(f"- {f}")
        
        st.caption(f"Last Update: {main_info.get('timestamp', 'Unknown')}")

# ==========================================
# SAYFA 2: SANAL CÜZDAN (YENİ!)
# ==========================================
elif page == "💼 Live Portfolio (Paper)":
    st.header("💼 Active Paper Trading Portfolio")
    st.caption("Gerçek para riske atılmadan yapılan canlı simülasyon sonuçları.")
    
    if st.button('🔄 Refresh Portfolio'):
        st.rerun()
    
    # Portfolio dosyasını oku
    portfolio = load_json("portfolio.json")
    
    if not portfolio:
        st.warning("⚠️ No portfolio data found yet. Wait for the AI to execute the first trade.")
        # Dosya yoksa varsayılan değerleri göster
        equity = 10000.0
        balance = 10000.0
        pnl = 0.0
    else:
        equity = portfolio.get('equity', 0)
        balance = portfolio.get('balance', 0)
        start_bal = PaperTrader.INITIAL_BALANCE
        pnl = equity - start_bal
    
    # --- ANA METRİKLER ---
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Total Equity (Varlık)", f"${equity:,.2f}")
    m2.metric("Available Cash (Nakit)", f"${balance:,.2f}")
    
    pnl_color = "normal" if pnl >= 0 else "inverse"
    m3.metric("Total PnL (Kar/Zarar)", f"${pnl:,.2f}", delta_color=pnl_color)
    
    st.markdown("---")
    
    # --- AÇIK POZİSYONLAR ---
    st.subheader("🔓 Open Positions")
    if portfolio and portfolio.get('positions'):
        # Sözlükten listeye çevir
        pos_list = []
        for sym, data in portfolio['positions'].items():
            row = data.copy()
            row['symbol'] = sym
            pos_list.append(row)
        
        df_pos = pd.DataFrame(pos_list)
        
        # Tabloyu güzelleştir
        st.dataframe(
            df_pos[['symbol', 'entry_price', 'current', 'amount', 'pnl', 'pnl_pct']],
            use_container_width=True,
            column_config={
                "pnl": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                "pnl_pct": st.column_config.NumberColumn("PnL (%)", format="%.2f%%"),
            }
        )
    else:
        st.info("No open positions currently. AI is scanning for opportunities.")
        
    st.markdown("---")

    # --- İŞLEM GEÇMİŞİ ---
    st.subheader("📜 Trade History")
    if portfolio and portfolio.get('history'):
        df_hist = pd.DataFrame(portfolio['history'])
        # Tersten sırala (En yeni en üstte)
        df_hist = df_hist.iloc[::-1]
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.caption("No completed trades yet.")

# ==========================================
# SAYFA 3: BACKTEST LAB
# ==========================================
elif page == "🧪 Backtest Lab (Time Machine)":
    st.header("⏳ Time Machine Simulation")
    
    col1, col2, col3 = st.columns(3)
    with col1: symbol = st.selectbox("Select Asset", Config.TARGET_COINS)
    with col2: days = st.slider("Lookback Period (Days)", 7, 60, 30)
    with col3: start_bal = st.number_input("Starting Balance ($)", value=10000)
        
    if st.button("🚀 START SIMULATION"):
        with st.spinner(f"AI is traveling {days} days back in time..."):
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
                    st.success("Simulation Complete!")
                    m1, m2, m3, m4 = st.columns(4)
                    roi = result['roi']
                    pnl = result['final_balance'] - start_bal
                    
                    roi_color = "normal" if roi >= 0 else "inverse"
                    
                    m1.metric("Net ROI", f"{roi:.2f}%", delta_color=roi_color)
                    m2.metric("Win Rate", f"{result['win_rate']:.1f}%")
                    m3.metric("Total Trades", result['total_trades'])
                    m4.metric("Net Profit", f"${pnl:,.2f}")
                    
                    st.metric("Final Wallet Balance", f"${result['final_balance']:,.2f}")
                    
                    if result['trades']:
                        trades_df = pd.DataFrame(result['trades'])
                        st.line_chart(trades_df[trades_df['action']=='SELL'].set_index('time')['balance'])
                        st.dataframe(trades_df)
                    else:
                        st.warning("No trades executed.")
            except Exception as e:
                st.error(f"Error: {e}")
