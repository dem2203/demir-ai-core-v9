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

# --- Ayarlar ---
st.set_page_config(
    page_title="DEMIR AI - Command Center",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;} 
    .stMetric {text-align: center;}
    .stDataFrame {font-size: 12px;}
</style>
""", unsafe_allow_html=True)

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

# --- 1. LIVE ---
if page == "📡 Live Dashboard":
    st.caption(f"Tracking Assets: {', '.join(Config.TARGET_COINS)}")
    if st.button('🔄 Refresh Market Data'): st.rerun()
    data = load_json("dashboard_data.json")

    if not data:
        st.info("📡 System is initializing... Waiting for Data Fusion...")
        time.sleep(2)
        st.rerun()
    else:
        main_sym = Config.TARGET_COINS[0] 
        btc_data = data.get(main_sym, list(data.values())[0])
        
        st.markdown("### 🌍 Global Macro Conditions")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        c1.metric("🇺🇸 DXY", f"{btc_data.get('dxy', 0):.2f}")
        c2.metric("😨 VIX", f"{btc_data.get('vix', 0):.2f}")
        c3.metric("📈 SPX", f"{btc_data.get('spx', 0):.0f}")
        c4.metric("🟡 GOLD", f"${btc_data.get('gold', 0):.1f}")
        c5.metric("🔗 CORR", f"{btc_data.get('corr_spx', 0):.2f}")
        c6.metric(f"₿ {btc_data.get('symbol')}", f"${btc_data.get('price', 0):,.2f}")
        
        st.markdown("---")
        
        dec = btc_data.get('ai_decision', 'NEUTRAL')
        color = "normal"
        if dec == "BUY": color = "inverse"
        elif dec == "SELL": color = "off"
        st.metric("🧠 AI Decision (LSTM)", dec, f"{btc_data.get('ai_confidence', 0):.1f}% Conf.", delta_color=color)

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

# --- 3. BACKTEST LAB (GÜNCELLENDİ) ---
elif page == "🧪 Backtest Lab":
    st.header("⏳ Time Machine Simulation")
    c1, c2, c3 = st.columns(3)
    with c1: symbol = st.selectbox("Asset", Config.TARGET_COINS)
    with c2: days = st.slider("Days", 7, 60, 30)
    with c3: start_bal = st.number_input("Start Balance", value=10000)
    
    if st.button("🚀 START SIMULATION"):
        with st.spinner("Running AI Simulation..."):
            async def run():
                bt = Backtester(initial_balance=start_bal)
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
                    st.success("Simulation Successful!")
                    
                    # --- PROFESYONEL GRAFİK ÇİZİMİ ---
                    st.markdown("### 🕯️ Trade Analysis Chart")
                    
                    # Backtester'dan dönen veriyi ve trade logunu kullan
                    df_sim = res.get('df')
                    
                    if df_sim is not None and not df_sim.empty:
                        # Timestamp'i datetime'a çevir (Plotly için)
                        df_sim['timestamp'] = pd.to_datetime(df_sim['timestamp'], unit='ms')
                        
                        # Visualizer'ı çağır
                        fig = MarketVisualizer.create_advanced_chart(df_sim, symbol, res['trades'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No price data available for charting.")
                    
                    st.subheader("📜 Detailed Logs")
                    st.dataframe(pd.DataFrame(res['trades']))
                else:
                    st.warning("No trades executed in this period.")

# --- 4. OPTIMIZER ---
elif page == "⚙️ Strategy Optimizer":
    st.header("🧬 Genetic Strategy Optimizer")
    target_sym = st.selectbox("Target", Config.TARGET_COINS, key="opt_sym")
    opt_days = st.slider("Period", 15, 60, 30, key="opt_days")
    
    if st.button("🧬 FIND BEST"):
        with st.spinner("Optimizing..."):
            async def run_opt():
                opt = StrategyOptimizer()
                return await opt.optimize(target_sym, opt_days)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            best = loop.run_until_complete(run_opt())['best_config']
            if best: st.success(f"Best ROI: {best['roi']:.2f}%"); st.json(best['params'])
            else: st.error("Failed.")
