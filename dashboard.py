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
st.markdown("""<style>.metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;} .stMetric {text-align: center;}</style>""", unsafe_allow_html=True)
st.title("🦅 DEMIR AI - Institutional Trading Terminal")

page = st.sidebar.radio("System Modules", ["📡 Live Dashboard", "💼 Live Portfolio", "🧪 Backtest Lab", "⚙️ Optimizer"])

def load_json(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                content = f.read()
                if not content: return {} 
                return json.loads(content)
        except: return {}
    return {}

if page == "📡 Live Dashboard":
    if st.button('🔄 Refresh'): st.rerun()
    data = load_json("dashboard_data.json")
    
    if not data:
        st.info("📡 Initializing...")
    else:
        main_sym = Config.TARGET_COINS[0]
        info = data.get(main_sym, list(data.values())[0])
        
        # --- MAKRO PANEL ---
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("DXY", f"{info.get('dxy', 0):.2f}")
        c2.metric("VIX", f"{info.get('vix', 0):.2f}")
        c3.metric("SPX", f"{info.get('spx', 0):.0f}")
        c4.metric("GOLD", f"${info.get('gold', 0):.0f}")
        c5.metric("SILVER", f"${info.get('silver', 0):.2f}")
        c6.metric("OIL", f"${info.get('oil', 0):.2f}")
        
        st.markdown("---")
        
        # --- AI ANALİZ & FUTURES ---
        col_main, col_funding = st.columns([2, 1])
        
        dec = info.get('ai_decision', 'NEUTRAL')
        col_main.metric("🧠 AI Decision", dec, f"{info.get('ai_confidence', 0):.1f}%", delta_color="normal" if dec=="BUY" else "off")
        
        # Funding Rate Göstergesi
        fr = info.get('funding_rate', 0)
        fr_color = "inverse" if fr > 0.01 else "normal"
        col_funding.metric("🔥 Funding Rate", f"{fr:.4f}%", "High Risk" if fr > 0.05 else "Normal", delta_color=fr_color)

        st.dataframe(pd.DataFrame(data.values())[['symbol', 'price', 'ai_decision', 'funding_rate', 'regime', 'rsi']], use_container_width=True)

elif page == "💼 Live Portfolio":
    st.header("Active Portfolio")
    if st.button('Refresh'): st.rerun()
    portfolio = load_json("portfolio.json")
    
    if not portfolio: st.warning("No trades.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Equity", f"${portfolio.get('equity', 0):,.2f}")
        m2.metric("Cash", f"${portfolio.get('balance', 0):,.2f}")
        m3.metric("PnL", f"${portfolio.get('equity', 0)-10000:,.2f}")
        
        if portfolio.get('positions'):
            st.dataframe(pd.DataFrame(portfolio['positions'].values()))
        if portfolio.get('history'):
            st.dataframe(pd.DataFrame(portfolio['history']).iloc[::-1])

elif page == "🧪 Backtest Lab":
    st.header("Backtest Simulation")
    c1, c2 = st.columns(2)
    with c1: symbol = st.selectbox("Asset", Config.TARGET_COINS)
    with c2: days = st.slider("Days", 7, 60, 30)
    if st.button("Run"):
        with st.spinner("Simulating..."):
            async def run():
                bt = Backtester()
                return await bt.run_backtest(symbol, days)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(run())
            if "error" in res: st.error(res['error'])
            else:
                st.metric("ROI", f"{res['roi']:.2f}%")
                if res['trades']:
                    df = pd.DataFrame(res['trades'])
                    st.line_chart(df[df['action']=='SELL'].set_index('time')['balance'])
                    st.dataframe(df)

elif page == "⚙️ Optimizer":
    st.header("Strategy Optimizer")
    sym = st.selectbox("Target", Config.TARGET_COINS)
    if st.button("Optimize"):
        with st.spinner("Optimizing..."):
            async def run_opt():
                opt = StrategyOptimizer()
                return await opt.optimize(sym, 30)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(run_opt())
            st.write(res['best_config'])
