import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
import plotly.express as px # Grafik için
from src.backtest.backtester import Backtester
from src.config.settings import Config
from src.execution.paper_trader import PaperTrader
from src.brain.optimizer import StrategyOptimizer 
from src.utils.visualizer import MarketVisualizer 
from src.brain.portfolio_manager import PortfolioManager # <-- YENİ: Markowitz Zekası

st.set_page_config(page_title="DEMIR AI", page_icon="🦅", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;} .stMetric {text-align: center;} .stDataFrame {font-size: 12px;}</style>""", unsafe_allow_html=True)
st.title("🦅 DEMIR AI - Institutional Trading Terminal")

page = st.sidebar.radio("System Modules", ["📡 Live Dashboard", "💼 Live Portfolio", "🧪 Backtest Lab", "⚙️ Optimizer"])

def load_json(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f: return json.load(f)
        except: return {}
    return {}

# --- 1. LIVE DASHBOARD ---
if page == "📡 Live Dashboard":
    st.caption(f"Tracking Assets: {', '.join(Config.TARGET_COINS)}")
    if st.button('🔄 Refresh'): st.rerun()
    data = load_json("dashboard_data.json")
    
    if not data:
        st.info("📡 System is initializing...")
        time.sleep(2)
        st.rerun()
    else:
        main_sym = Config.TARGET_COINS[0]
        info = data.get(main_sym, list(data.values())[0])
        
        # MAKRO PANEL
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("🇺🇸 DXY", f"{info.get('dxy', 0):.2f}")
        c2.metric("😨 VIX", f"{info.get('vix', 0):.2f}")
        c3.metric("📈 SPX", f"{info.get('spx', 0):.0f}")
        c4.metric("🟡 GOLD", f"${info.get('gold', 0):.1f}")
        c5.metric("⚪ SILVER", f"${info.get('silver', 0):.2f}")
        
        # Funding Rate Rengi
        fr = info.get('funding_rate', 0)
        fr_col = "inverse" if fr > 0.01 else "normal"
        c6.metric("🔥 Funding", f"{fr:.4f}%", delta_color=fr_col)
        
        st.markdown("---")
        
        # AI KARARI
        dec = info.get('ai_decision', 'NEUTRAL')
        color = "normal" if dec=="BUY" else ("inverse" if dec=="SELL" else "off")
        st.metric(f"🧠 AI Decision ({info.get('symbol')})", dec, f"{info.get('ai_confidence', 0):.1f}% Conf.", delta_color=color)

        # DETAYLI TABLO
        st.markdown("### 📊 Asset Analysis Board")
        df = pd.DataFrame(data.values())
        cols = ['symbol', 'price', 'ai_decision', 'ai_confidence', 'regime', 'rsi', 'funding_rate']
        st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True)

# --- 2. PORTFOLIO ---
elif page == "💼 Live Portfolio":
    st.header("Active Portfolio")
    if st.button('Refresh'): st.rerun()
    portfolio = load_json("portfolio.json")
    
    if not portfolio: st.warning("No trades yet.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Equity", f"${portfolio.get('equity', 0):,.2f}")
        m2.metric("Cash", f"${portfolio.get('balance', 0):,.2f}")
        m3.metric("PnL", f"${portfolio.get('equity', 0)-10000:,.2f}")
        
        if portfolio.get('positions'):
            st.dataframe(pd.DataFrame(portfolio['positions'].values()))
        if portfolio.get('history'):
            st.dataframe(pd.DataFrame(portfolio['history']).iloc[::-1])

# --- 3. BACKTEST LAB (YENİ: PORTFÖY MATEMATİĞİ EKLENDİ) ---
elif page == "🧪 Backtest Lab":
    st.header("⏳ Time Machine & Portfolio Science")
    
    tab1, tab2 = st.tabs(["Strategy Backtest", "Harry Markowitz Optimization"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1: symbol = st.selectbox("Asset", Config.TARGET_COINS)
        with c2: days = st.slider("Days", 7, 60, 30)
        
        if st.button("Run Strategy Test"):
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
                        df = res.get('df')
                        if df is not None:
                            fig = MarketVisualizer.create_advanced_chart(df, symbol, res['trades'])
                            st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(pd.DataFrame(res['trades']))
    
    with tab2:
        st.caption("Modern Portföy Teorisi (MPT) kullanarak en verimli dağılımı hesaplar.")
        if st.button("🧮 Optimize Portfolio Weights"):
            with st.spinner("Calculating Efficient Frontier..."):
                # Burası için Backtester'ı kullanıp sadece veri çekeceğiz
                async def get_data():
                    bt = Backtester()
                    prices = pd.DataFrame()
                    for sym in Config.TARGET_COINS:
                        raw = await bt.crypto.fetch_candles(sym, limit=500)
                        if raw:
                            df = pd.DataFrame(raw)
                            prices[sym] = df['close']
                    await bt.crypto.close()
                    return prices
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                price_data = loop.run_until_complete(get_data())
                
                if not price_data.empty:
                    pm = PortfolioManager()
                    weights = pm.optimize_allocation(price_data)
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.success("Optimal Allocation Found!")
                        st.json(weights)
                    with c2:
                        # Pasta Grafiği
                        df_w = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
                        fig = px.pie(df_w, values='Weight', names='Asset', title='AI Suggested Portfolio')
                        st.plotly_chart(fig)
                else:
                    st.error("Could not fetch price data.")

# --- 4. OPTIMIZER ---
elif page == "⚙️ Optimizer":
    st.header("Genetic Strategy Optimizer")
    sym = st.selectbox("Target", Config.TARGET_COINS)
    if st.button("Find Best Settings"):
        with st.spinner("Optimizing..."):
            async def run_opt():
                opt = StrategyOptimizer()
                return await opt.optimize(sym, 30)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(run_opt())
            st.write(res['best_config'])
