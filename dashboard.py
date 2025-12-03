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
from src.core.risk_manager import RiskManager # Yeni

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="DEMIR AI - Institutional Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Profesyonel CSS (Dark Mode & Typography) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 800;
    }
    
</style>
""", unsafe_allow_html=True)

st.title("🦅 DEMIR AI - Institutional Trading Terminal")
st.caption("v19.0 | Zero-Mock | Fractal Confluence | Dynamic Risk")

# --- Yan Menü ---
page = st.sidebar.radio("System Modules", [
    "📡 Live Market Intelligence", 
    "💼 Advisory Portfolio", 
    "🧪 Backtest Lab",
    "⚙️ Strategy Optimizer"
])

def load_json(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                content = f.read()
                if not content: return {} 
                return json.loads(content)
        except: return {}
    return {}

risk_manager = RiskManager()

# ==========================================
# 1. CANLI İZLEME (Live Market Intelligence)
# ==========================================
if page == "📡 Live Market Intelligence":
    st.sidebar.markdown("---")
    st.sidebar.info("System Status: **ONLINE**")
    
    if st.button('🔄 Refresh Data'): st.rerun()

    data = load_json("dashboard_data.json")

    if not data:
        st.warning("📡 Waiting for Live Data Stream... (No Mock Data Displayed)")
        st.info("System is in 'Zero-Mock' mode. If markets are closed or API is down, no data will be shown.")
        time.sleep(2)
        st.rerun()
    else:
        # Ana Gösterge Paneli
        main_symbol = Config.TARGET_COINS[0] 
        btc_data = data.get(main_symbol)
        
        if btc_data:
            main_info = btc_data
            display_symbol = main_symbol
        elif len(data) > 0:
            first_key = list(data.keys())[0]
            main_info = data[first_key]
            display_symbol = main_info['symbol']
        else:
            st.error("Data stream interrupted.")
            st.stop()
        
        # Global Metrics
        c1, c2, c3, c4 = st.columns(4)
        
        dxy = main_info.get('dxy', 0)
        vix = main_info.get('vix', 0)
        price = main_info.get('price', 0)
        
        c1.metric("🇺🇸 DXY Index", f"{dxy:.2f}" if dxy > 0 else "N/A")
        c2.metric("😨 VIX Index", f"{vix:.2f}" if vix > 0 else "N/A")
        c3.metric(f"₿ {display_symbol}", f"${price:,.2f}" if price > 0 else "N/A")
        
        dec = main_info.get('ai_decision', 'NEUTRAL')
        conf = main_info.get('ai_confidence', 0)
        
        delta_color = "off"
        if dec == "BUY": delta_color = "normal"
        elif dec == "SELL": delta_color = "inverse"
        
        c4.metric("🧠 AI Signal", dec, f"{conf:.1f}% Conf.", delta_color=delta_color)

        st.markdown("---")
        
        # Detaylı Tablo
        st.markdown("### 📊 Market Analysis Board")
        
        # Veriyi zenginleştir (Kelly Size ekle)
        display_data = []
        for sym, info in data.items():
            info_copy = info.copy()
            conf = info.get('ai_confidence', 0)
            # Kelly Hesapla
            kelly_size = risk_manager.calculate_kelly_size(conf) if info.get('ai_decision') != 'NEUTRAL' else 0
            info_copy['kelly_size'] = kelly_size
            display_data.append(info_copy)
            
        df_display = pd.DataFrame(display_data)
        
        cols = ['symbol', 'price', 'ai_decision', 'ai_confidence', 'kelly_size', 'fractal_score', 'hurst']
        valid_cols = [c for c in cols if c in df_display.columns]
        
        if not df_display.empty:
            st.dataframe(
                df_display[valid_cols],
                use_container_width=True,
                column_config={
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "ai_confidence": st.column_config.ProgressColumn("Confidence", format="%.1f%%", min_value=0, max_value=100),
                    "kelly_size": st.column_config.NumberColumn("Kelly Risk (%)", format="%.2f%%"),
                    "fractal_score": st.column_config.NumberColumn("Fractal Score", format="%.1f"),
                    "hurst": st.column_config.NumberColumn("Hurst Exp", format="%.2f"),
                }
            )

        # AI Mantığı Detayları
        st.markdown("### 🤖 AI Reasoning Engine")
        c_left, c_right = st.columns(2)
        
        for i, (symbol, info) in enumerate(data.items()):
            col = c_left if i % 2 == 0 else c_right
            with col:
                with st.expander(f"🔍 {symbol} Analysis Details", expanded=True):
                    st.write(f"**Decision:** {info.get('ai_decision')}")
                    st.write(f"**Reason:** {info.get('reason', 'N/A')}")
                    
                    # Fractal Göstergesi
                    f_score = info.get('fractal_score', 0)
                    if f_score > 80: st.success(f"Fractal Alignment: PERFECT ({f_score:.0f})")
                    elif f_score > 50: st.warning(f"Fractal Alignment: MODERATE ({f_score:.0f})")
                    else: st.error(f"Fractal Alignment: WEAK ({f_score:.0f})")
                    
                    st.write(f"**Regime:** {info.get('regime', 'UNKNOWN')}")
                    st.write(f"**Funding Risk:** {info.get('funding_rate', 0):.4f}%")

# ==========================================
# 2. SANAL CÜZDAN (Advisory Portfolio)
# ==========================================
elif page == "💼 Advisory Portfolio":
    st.header("💼 Advisory Portfolio Tracker")
    st.caption("Simulated execution of AI signals. No real funds at risk.")
    
    if st.button('🔄 Refresh Portfolio'): st.rerun()
    
    portfolio = load_json("portfolio.json")
    market_data = load_json("dashboard_data.json")
    
    if not portfolio:
        st.info("Waiting for the first AI signal execution...")
    else:
        balance = portfolio.get('balance', 0)
        equity = balance
        positions_data = []
        
        if portfolio.get('positions'):
            for sym, pos in portfolio['positions'].items():
                current_price = pos.get('entry_price', 0) 
                if market_data and sym in market_data:
                    current_price = market_data[sym].get('price', pos['entry_price'])
                
                amount = pos.get('amount', 0)
                cost = pos.get('cost', 0)
                market_val = amount * current_price
                equity += market_val
                unrealized_pnl = market_val - cost
                pnl_pct = (unrealized_pnl / cost) * 100 if cost > 0 else 0
                
                positions_data.append({
                    "symbol": sym, 
                    "entry": pos.get('entry_price', 0), 
                    "current": current_price,
                    "pnl": unrealized_pnl, 
                    "pnl_pct": pnl_pct
                })
        
        pnl_total = equity - PaperTrader.INITIAL_BALANCE
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Equity", f"${equity:,.2f}")
        m2.metric("Cash Balance", f"${balance:,.2f}")
        m3.metric("Total PnL", f"${pnl_total:,.2f}", delta_color="normal" if pnl_total >= 0 else "inverse")
        
        st.markdown("---")
        st.subheader("🔓 Open Positions")
        
        if positions_data:
            df_pos = pd.DataFrame(positions_data)
            st.dataframe(
                df_pos, 
                use_container_width=True,
                column_config={
                    "entry": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
                    "current": st.column_config.NumberColumn("Current Price", format="$%.2f"),
                    "pnl": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                    "pnl_pct": st.column_config.NumberColumn("PnL (%)", format="%.2f%%"),
                }
            )
        else: 
            st.info("No open positions. AI is scanning for opportunities.")
            
        st.subheader("📜 Trade History")
        if portfolio.get('history'):
            df_hist = pd.DataFrame(portfolio['history'])
            if not df_hist.empty:
                st.dataframe(df_hist.iloc[::-1], use_container_width=True)

# ==========================================
# 3. BACKTEST LAB
# ==========================================
elif page == "🧪 Backtest Lab":
    st.header("⏳ Historical Simulation")
    c1, c2 = st.columns(2)
    with c1: symbol = st.selectbox("Asset", Config.TARGET_COINS)
    with c2: days = st.slider("Lookback Days", 7, 60, 30)
    
    if st.button("🚀 Run Simulation"):
        with st.spinner("Crunching numbers..."):
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
                c3.metric("Total Trades", res['total_trades'])
                if res['trades']: 
                    df = pd.DataFrame(res['trades'])
                    st.line_chart(df[df['action']=='SELL'].set_index('time')['balance'])
                    st.dataframe(df)

# ==========================================
# 4. OPTIMIZER
# ==========================================
elif page == "⚙️ Strategy Optimizer":
    st.header("🧬 Genetic Strategy Optimizer")
    st.info("Uses genetic algorithms to find optimal parameters for the current market regime.")
    
    target_sym = st.selectbox("Target Asset", Config.TARGET_COINS, key="opt_sym")
    
    if st.button("🧬 Start Optimization"):
        with st.spinner("Optimizing... This may take a while."):
            async def run_opt():
                opt = StrategyOptimizer()
                return await opt.optimize(target_sym, 30)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(run_opt())
            
            if res.get('best_config'):
                best = res['best_config']
                st.success(f"✅ Optimization Complete! Best ROI: {best['roi']:.2f}%")
                st.json(best['params'])
            else:
                st.error("Optimization failed.")
