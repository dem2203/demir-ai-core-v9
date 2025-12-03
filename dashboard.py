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
    .stDataFrame {font-size: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🦅 DEMIR AI - Institutional Trading Terminal")

# --- Yan Menü ---
page = st.sidebar.radio("System Modules", [
    "📡 Live Dashboard", 
    "💼 Live Portfolio (Paper)", 
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

# ==========================================
# 1. CANLI İZLEME
# ==========================================
if page == "📡 Live Dashboard":
    st.caption(f"Tracking Assets: {', '.join(Config.TARGET_COINS)}")
    if st.button('🔄 Refresh Market Data'): st.rerun()

    data = load_json("dashboard_data.json")

    if not data:
        st.info("📡 System is initializing... Waiting for Data Fusion...")
        time.sleep(2)
        st.rerun()
    else:
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
            st.warning("No asset data available yet.")
            st.stop()
        
        # Global Metrics
        c1, c2, c3, c4 = st.columns(4)
        dxy = main_info.get('dxy', 0)
        vix = main_info.get('vix', 0)
        c1.metric("🇺🇸 DXY", f"{dxy:.2f}" if dxy > 0 else "Loading...")
        c2.metric("😨 VIX", f"{vix:.2f}" if vix > 0 else "Loading...")
        c3.metric(f"₿ {display_symbol} Price", f"${main_info.get('price', 0):,.2f}")
        
        dec = main_info.get('ai_decision', 'NEUTRAL')
        conf = main_info.get('ai_confidence', 0)
        color = "normal"
        if dec == "BUY": color = "inverse"
        elif dec == "SELL": color = "off"
        c4.metric("🧠 AI Decision (LSTM)", dec, f"{conf:.1f}% Conviction", delta_color=color)

        st.markdown("---")
        
        # Tablo
        st.markdown("### 📊 Asset Analysis Board")
        df_display = pd.DataFrame(data.values())
        
        cols = ['symbol', 'price', 'ai_decision', 'ai_confidence', 'regime', 'rsi', 'trend']
        valid_cols = [c for c in cols if c in df_display.columns]
        
        if not df_display.empty:
            st.dataframe(
                df_display[valid_cols],
                use_container_width=True,
                column_config={
                    "ai_confidence": st.column_config.ProgressColumn("AI Conviction", format="%.1f%%", min_value=0, max_value=100),
                }
            )

        # AI Mantığı
        st.markdown("### 🤖 AI Reasoning Engine")
        for symbol, info in data.items():
            with st.expander(f"🔍 Inspect Logic: {symbol}", expanded=False):
                st.write(f"**LSTM Prediction:** {info.get('ai_decision')} ({info.get('ai_confidence'):.1f}%)")
                st.info(f"Regime: {info.get('regime', 'UNKNOWN')}")

# ==========================================
# 2. SANAL CÜZDAN (DÜZELTİLDİ - EŞLEŞME GARANTİLİ)
# ==========================================
elif page == "💼 Live Portfolio (Paper)":
    st.header("💼 Active Paper Trading Portfolio")
    if st.button('🔄 Refresh Portfolio'): st.rerun()
    
    portfolio = load_json("portfolio.json")
    market_data = load_json("dashboard_data.json")
    
    if not portfolio:
        st.warning("⚠️ No portfolio data found yet. Wait for the AI to execute the first trade.")
    else:
        balance = portfolio.get('balance', 0)
        equity = balance
        positions_data = []
        
        # Canlı PnL Hesaplama
        if portfolio.get('positions'):
            for sym, pos in portfolio['positions'].items():
                # Canlı fiyatı bul
                current_price = pos.get('entry_price', 0) 
                if market_data and sym in market_data:
                    current_price = market_data[sym].get('price', pos['entry_price'])
                
                amount = pos.get('amount', 0)
                cost = pos.get('cost', 0)
                
                market_val = amount * current_price
                equity += market_val
                
                unrealized_pnl = market_val - cost
                
                pnl_pct = 0
                if cost > 0:
                    pnl_pct = (unrealized_pnl / cost) * 100
                
                # Tablo verisi (Hepsi küçük harf key)
                positions_data.append({
                    "symbol": sym, 
                    "entry_price": pos.get('entry_price', 0), 
                    "current_price": current_price,
                    "amount": amount, 
                    "pnl": unrealized_pnl, 
                    "pnl_pct": pnl_pct
                })
        
        pnl_total = equity - PaperTrader.INITIAL_BALANCE
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Equity", f"${equity:,.2f}")
        m2.metric("Available Cash", f"${balance:,.2f}")
        m3.metric("Total PnL", f"${pnl_total:,.2f}", delta_color="normal" if pnl_total >= 0 else "inverse")
        
        st.markdown("---")
        st.subheader("🔓 Open Positions")
        
        if positions_data:
            df_pos = pd.DataFrame(positions_data)
            # Sütunları seç ve göster (Key isimleri yukarıdakiyle AYNI olmalı)
            cols_to_show = ['symbol', 'entry_price', 'current_price', 'amount', 'pnl', 'pnl_pct']
            
            st.dataframe(
                df_pos[cols_to_show], 
                use_container_width=True,
                column_config={
                    "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                    "current_price": st.column_config.NumberColumn("Current", format="$%.2f"),
                    "pnl": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                    "pnl_pct": st.column_config.NumberColumn("PnL (%)", format="%.2f%%"),
                }
            )
        else: 
            st.info("No open positions currently. AI is scanning...")
            
        st.subheader("📜 Trade History")
        if portfolio.get('history'):
            df_hist = pd.DataFrame(portfolio['history'])
            if not df_hist.empty:
                st.dataframe(df_hist.iloc[::-1], use_container_width=True)
            else:
                st.caption("History is empty.")

# ==========================================
# 3. BACKTEST LAB
# ==========================================
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
                else:
                    st.warning("No trades executed.")

# ==========================================
# 4. OPTIMIZER
# ==========================================
elif page == "⚙️ Strategy Optimizer":
    st.header("🧬 Genetic Strategy Optimizer")
    target_sym = st.selectbox("Target", Config.TARGET_COINS, key="opt_sym")
    opt_days = st.slider("Period", 15, 60, 30, key="opt_days")
    
    if st.button("🧬 FIND BEST SETTINGS"):
        with st.spinner("Optimizing..."):
            async def run_opt():
                opt = StrategyOptimizer()
                return await opt.optimize(target_sym, opt_days)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(run_opt())
            
            if res.get('best_config'):
                best = res['best_config']
                st.success(f"✅ Best ROI: {best['roi']:.2f}%")
                st.json(best['params'])
                st.dataframe(pd.DataFrame(res['all_results']).sort_values('roi', ascending=False))
            else:
                st.error("Optimization failed.")
