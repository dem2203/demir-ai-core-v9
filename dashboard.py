import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
from src.backtest.backtester import Backtester
from src.config.settings import Config
from src.execution.paper_trader import PaperTrader
from src.brain.optimizer import StrategyOptimizer 
from src.utils.visualizer import MarketVisualizer 

st.set_page_config(page_title="DEMIR AI", page_icon="🦅", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;} .stMetric {text-align: center;} .stDataFrame {font-size: 12px;}</style>""", unsafe_allow_html=True)

st.title("🦅 DEMIR AI - Institutional Trading Terminal")

page = st.sidebar.radio("System Modules", ["📡 Live Dashboard", "💼 Live Portfolio (Paper)", "🧪 Backtest Lab", "⚙️ Strategy Optimizer"])

def load_json(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                content = f.read()
                if not content: return {} 
                return json.loads(content)
        except: return {}
    return {}

# --- 1. LIVE DASHBOARD ---
if page == "📡 Live Dashboard":
    st.caption(f"Tracking Assets: {', '.join(Config.TARGET_COINS)}")
    if st.button('🔄 Refresh'): st.rerun()
    data = load_json("dashboard_data.json")
    
    if not data:
        st.info("📡 System is initializing... Waiting for Data Fusion...")
        time.sleep(2)
        st.rerun()
    else:
        main_sym = Config.TARGET_COINS[0]
        info = data.get(main_sym, list(data.values())[0])
        
        # --- MAKRO PANEL (GENİŞLETİLMİŞ) ---
        st.markdown("### 🌍 Global Macro Pulse")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🇺🇸 DXY", f"{info.get('dxy', 0):.2f}")
        c2.metric("😨 VIX", f"{info.get('vix', 0):.2f}")
        c3.metric("📈 SPX", f"{info.get('spx', 0):.0f}")
        c4.metric("🟡 GOLD", f"${info.get('gold', 0):.1f}")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("⚪ SILVER", f"${info.get('silver', 0):.2f}")
        c6.metric("🛢️ OIL", f"${info.get('oil', 0):.2f}")
        corr = info.get('corr_spx', 0)
        c7.metric("🔗 Correlation", f"{corr:.2f}", delta_color="normal" if corr > 0.5 else "off")
        c8.metric(f"₿ {info.get('symbol')}", f"${info.get('price', 0):,.2f}")
        
        st.markdown("---")
        
        dec = info.get('ai_decision', 'NEUTRAL')
        color = "normal"
        if dec == "BUY": color = "inverse"
        elif dec == "SELL": color = "off"
        st.metric("🧠 AI Decision (LSTM)", dec, f"{info.get('ai_confidence', 0):.1f}%", delta_color=color)
        
        st.markdown("### 📊 Asset Analysis Board")
        df_display = pd.DataFrame(data.values())
        cols = ['symbol', 'price', 'ai_decision', 'ai_confidence', 'regime', 'rsi', 'trend']
        valid_cols = [c for c in cols if c in df_display.columns]
        st.dataframe(df_display[valid_cols], use_container_width=True)

# --- 2. PORTFOLIO ---
elif page == "💼 Live Portfolio (Paper)":
    st.header("💼 Active Paper Trading Portfolio")
    if st.button('🔄 Refresh'): st.rerun()
    portfolio = load_json("portfolio.json")
    
    if not portfolio: st.warning("No trades yet.")
    else:
        m1, m2, m3 = st.columns(3)
        equity = portfolio.get('equity', 0)
        m1.metric("Equity", f"${equity:,.2f}")
        m2.metric("Cash", f"${portfolio.get('balance', 0):,.2f}")
        m3.metric("PnL", f"${equity - 10000:,.2f}")
        
        if portfolio.get('positions'):
            st.dataframe(pd.DataFrame(portfolio['positions'].values()))
        else: st.info("No open positions.")
        
        if portfolio.get('history'):
            st.subheader("📜 Trade History")
            st.dataframe(pd.DataFrame(portfolio['history']).iloc[::-1])

# --- 3. BACKTEST ---
elif page == "🧪 Backtest Lab":
    st.header("⏳ Time Machine Simulation")
    c1, c2 = st.columns(2)
    with c1: symbol = st.selectbox("Asset", Config.TARGET_COINS)
    with c2: days = st.slider("Days", 7, 60, 30)
    
    if st.button("🚀 Run Backtest"):
        with st.spinner("Simulating..."):
            async def run():
                bt = Backtester()
                return await bt.run_backtest(symbol, days)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(run())
            
            if "error" in res: st.error(res['error'])
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("ROI", f"{res['roi']:.2f}%")
                c2.metric("Win Rate", f"{res['win_rate']:.1f}%")
                c3.metric("Trades", res['total_trades'])
                if res['trades']: 
                    df = pd.DataFrame(res['trades'])
                    st.line_chart(df[df['action']=='SELL'].set_index('time')['balance'])
                    st.dataframe(df)

# --- 4. OPTIMIZER ---
elif page == "⚙️ Strategy Optimizer":
    st.header("🧬 Genetic Strategy Optimizer")
    target_sym = st.selectbox("Target", Config.TARGET_COINS)
    if st.button("🧬 FIND BEST"):
        with st.spinner("Optimizing..."):
            async def run_opt():
                opt = StrategyOptimizer()
                return await opt.optimize(target_sym, 30)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            best = loop.run_until_complete(run_opt())['best_config']
            if best: st.success(f"Best ROI: {best['roi']:.2f}%"); st.json(best['params'])
            else: st.error("Failed.")
