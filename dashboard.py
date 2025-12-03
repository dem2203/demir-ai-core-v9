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
        
        # Güvenli veri çekme (Veri yoksa ilkini al)
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
        
        # Sütunların varlığını kontrol et
        available_cols = ['symbol', 'price', 'ai_decision', 'ai_confidence', 'regime', 'rsi', 'trend']
        valid_cols = [c for c in available_cols if c in df_display.columns]
        
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
# 2. SANAL CÜZDAN (DÜZELTİLDİ)
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
                current_price = pos.get('entry_price', 0) # Varsayılan
                
                # Eğer market verisi varsa oradan güncel fiyatı al
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
                
                # Tablo için veri hazırla (Sözlük yapısı)
                positions_data.append({
                    "Symbol": sym, 
                    "Entry Price": pos.get('entry_price', 0), 
                    "Current Price": current_price,
                    "Amount": amount, 
                    "PnL ($)": unrealized_pnl, 
                    "PnL (%)": pnl_pct
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
            # Sütun formatlama
            st.dataframe(
                df_pos, 
                use_container_width=True,
                column_config={
                    "Entry Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                    "PnL ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "PnL (%)": st.column_config.NumberColumn(format="%.2f%%"),
                }
            )
        else: 
            st.info("No open positions.")
            
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
    c1, c2, c3 = st.columns(3)
    with c1: symbol = st.selectbox("Asset", Config.TARGET_COINS)
    with c2: days = st.slider("Days", 7, 60, 30)
    with c3: start_bal = st.number_input("Start Balance", value=10000)
    
    if st.button("🚀 Run Backtest"):
        with st.spinner("Simulating..."):
            async def run():
                bt = Backtester(initial_balance=start_bal)
                return await bt.run_backtest(symbol, days)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(run())
                
                if "error" in res:
                    st.error(f"Backtest Failed: {result['error']}")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ROI", f"{res['roi']:.2f}%")
                    m2.metric("Win Rate", f"{res['win_rate']:.1f}%")
                    m3.metric("Trades", res['total_trades'])
                    m4.metric("Profit", f"${res['final_balance'] - start_bal:,.2f}")
                    
                    if res['trades']:
                        trades_df = pd.DataFrame(res['trades'])
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
    target_sym = st.selectbox("Target Asset", Config.TARGET_COINS, key="opt_sym")
    opt_days = st.slider("Optimization Period", 15, 60, 30, key="opt_days")
    
    if st.button("🧬 FIND BEST SETTINGS"):
        with st.spinner("Running Grid Search..."):
            async def run_opt():
                opt = StrategyOptimizer()
                return await opt.optimize(target_sym, opt_days)
            
            try:
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
                    
                    st.subheader("📊 Leaderboard")
                    df_res = pd.DataFrame(results['all_results'])
                    df_res['params'] = df_res['params'].astype(str)
                    st.dataframe(df_res.sort_values('roi', ascending=False))
                else:
                    st.error("Optimization failed.")
            except Exception as e: st.error(f"Error: {e}")
