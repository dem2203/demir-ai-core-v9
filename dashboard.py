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

# --- Yan Menü (4 MODÜL) ---
page = st.sidebar.radio("System Modules", [
    "📡 Live Dashboard", 
    "💼 Live Portfolio (Paper)", 
    "🧪 Backtest Lab (Time Machine)",
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
# 1. CANLI İZLEME (MACRO + AI)
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
        btc_data = data.get(main_symbol, list(data.values())[0])
        
        # --- ÜST PANEL: 5 FAKTÖRLÜ MAKRO ANALİZ ---
        st.markdown("### 🌍 Global Macro Conditions")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        # Verileri güvenli çek
        dxy = btc_data.get('dxy', 0)
        vix = btc_data.get('vix', 0)
        spx = btc_data.get('spx', 0)
        ndq = btc_data.get('ndq', 0)
        tnx = btc_data.get('tnx', 0)
        
        c1.metric("🇺🇸 DXY", f"{dxy:.2f}" if dxy > 0 else "Loading...")
        c2.metric("😨 VIX", f"{vix:.2f}" if vix > 0 else "Loading...")
        c3.metric("📈 SPX", f"{spx:.0f}" if spx > 0 else "Loading...")
        c4.metric("💻 NASDAQ", f"{ndq:.0f}" if ndq > 0 else "Loading...")
        c5.metric("🏦 US10Y", f"{tnx:.2f}%" if tnx > 0 else "Loading...")
        c6.metric(f"₿ {btc_data.get('symbol')}", f"${btc_data.get('price', 0):,.2f}")
        
        st.markdown("---")
        
        # --- AI KARARI ---
        dec = btc_data.get('ai_decision', 'NEUTRAL')
        conf = btc_data.get('ai_confidence', 0)
        regime = btc_data.get('regime', 'UNKNOWN')
        
        c_ai, c_reg = st.columns([1, 3])
        color = "off"
        if dec == "BUY": color = "normal"
        elif dec == "SELL": color = "inverse"
        
        c_ai.metric("🧠 AI Decision (LSTM)", dec, f"{conf:.1f}% Conf.", delta_color=color)
        c_reg.info(f"📡 Market Regime: **{regime}** (Strategy Adapted)")

        # --- TABLO ---
        st.markdown("### 📊 Asset Analysis Board")
        df_display = pd.DataFrame(data.values())
        cols = ['symbol', 'price', 'ai_decision', 'ai_confidence', 'regime', 'rsi', 'trend']
        valid_cols = [c for c in cols if c in df_display.columns]
        
        st.dataframe(
            df_display[valid_cols],
            use_container_width=True,
            column_config={
                "ai_confidence": st.column_config.ProgressColumn("AI Conviction", format="%.1f%%", min_value=0, max_value=100),
            }
        )

# ==========================================
# 2. SANAL CÜZDAN
# ==========================================
elif page == "💼 Live Portfolio (Paper)":
    st.header("💼 Active Paper Trading Portfolio")
    if st.button('🔄 Refresh'): st.rerun()
    
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
                current_price = pos['entry_price']
                if market_data and sym in market_data:
                    current_price = market_data[sym]['price']
                
                market_val = pos['amount'] * current_price
                equity += market_val
                unrealized_pnl = market_val - pos['cost']
                pnl_pct = (unrealized_pnl / pos['cost']) * 100 if pos['cost'] > 0 else 0
                
                positions_data.append({
                    "symbol": sym, "entry": pos['entry_price'], "current": current_price,
                    "amount": pos['amount'], "pnl": unrealized_pnl, "pnl_pct": pnl_pct
                })
        
        pnl_total = equity - PaperTrader.INITIAL_BALANCE
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Equity", f"${equity:,.2f}")
        m2.metric("Available Cash", f"${balance:,.2f}")
        m3.metric("Total PnL", f"${pnl_total:,.2f}", delta_color="normal" if pnl_total >= 0 else "inverse")
        
        st.markdown("---")
        st.subheader("🔓 Open Positions")
        if positions_data:
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
        else: st.info("No open positions.")
            
        st.subheader("📜 Trade History")
        if portfolio.get('history'):
            st.dataframe(pd.DataFrame(portfolio['history']).iloc[::-1], use_container_width=True)

# ==========================================
# 3. BACKTEST LAB
# ==========================================
elif page == "🧪 Backtest Lab":
    st.header("⏳ Time Machine Simulation")
    c1, c2, c3 = st.columns(3)
    with c1: symbol = st.selectbox("Select Asset", Config.TARGET_COINS)
    with c2: days = st.slider("Lookback Period (Days)", 7, 60, 30)
    with c3: start_bal = st.number_input("Start Balance ($)", value=10000)
    
    if st.button("🚀 START SIMULATION"):
        with st.spinner(f"AI is traveling {days} days back in time..."):
            async def run_test():
                bt = Backtester(initial_balance=start_bal)
                # Backtest artık DataFrame de döndürmeli ki grafik çizebilelim
                # (Mevcut backtester sadece dict dönüyor, onu güncelledik varsayıyoruz veya trade log kullanıyoruz)
                return await bt.run_backtest(symbol, days)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_test())
                
                if "error" in result:
                    st.error(f"Backtest Failed: {result['error']}")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ROI", f"{result['roi']:.2f}%")
                    m2.metric("Win Rate", f"{result['win_rate']:.1f}%")
                    m3.metric("Trades", result['total_trades'])
                    m4.metric("Profit", f"${result['final_balance'] - start_bal:,.2f}")
                    
                    if result['trades']:
                        trades_df = pd.DataFrame(result['trades'])
                        # Profesyonel Grafik
                        st.subheader("📈 Equity Curve")
                        st.line_chart(trades_df[trades_df['action']=='SELL'].set_index('time')['balance'])
                        st.dataframe(trades_df, use_container_width=True)
                    else:
                        st.warning("No trades executed.")
            except Exception as e: st.error(f"Error: {e}")

# ==========================================
# 4. STRATEGY OPTIMIZER
# ==========================================
elif page == "⚙️ Strategy Optimizer":
    st.header("🧬 Genetic Strategy Optimizer")
    st.markdown("Yapay zeka, geçmiş verileri tarayarak en karlı ayarları (Stop Loss, Güven Eşiği) bulmaya çalışacak.")
    
    target_sym = st.selectbox("Target Asset", Config.TARGET_COINS, key="opt_sym")
    opt_days = st.slider("Optimization Period", 15, 60, 30, key="opt_days")
    
    if st.button("🧬 FIND BEST SETTINGS"):
        with st.spinner("Running Grid Search... This may take a minute..."):
            async def run_opt():
                opt = StrategyOptimizer()
                return await opt.optimize(target_sym, opt_days)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(run_opt())
            
            best = results['best_config']
            if best:
                st.success(f"✅ Optimization Complete! Best ROI: {best['roi']:.2f}%")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Best Stop Loss", f"{best['params']['sl_mul']}x ATR")
                c2.metric("Best Take Profit", f"{best['params']['tp_mul']}x ATR")
                c3.metric("AI Threshold", f"{best['params']['threshold']*100:.0f}%")
                
                st.subheader("📊 All Results")
                df_res = pd.DataFrame(results['all_results'])
                df_res['params'] = df_res['params'].astype(str)
                st.dataframe(df_res.sort_values('roi', ascending=False))
            else:
                st.error("Optimization failed.")
