import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
from src.backtest.backtester import Backtester
from src.config.settings import Config
from src.execution.paper_trader import PaperTrader
from src.brain.optimizer import StrategyOptimizer # <-- Optimizer Geri Geldi
from src.utils.visualizer import MarketVisualizer # <-- Görselleştirici

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
        else:
            first_key = list(data.keys())[0]
            main_info = data[first_key]
            display_symbol = main_info['symbol']
        
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
# 2. SANAL CÜZDAN
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
        
        if portfolio.get('positions'):
            for sym, pos in portfolio['positions'].items():
                current_price = pos['entry_price']
                if market_data and sym in market_data:
                    current_price = market_data[sym]['price']
                
                market_val = pos['amount'] * current_price
                equity += market_val
                unrealized_pnl = market_val - pos['cost']
                pnl_pct = (unrealized_pnl / pos['cost']) * 100
                
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
# 3. BACKTEST LAB (GRAFİK GÜNCELLENDİ)
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
                # Backtester'ın run_backtest metodunun hem raporu hem de DataFrame'i döndürdüğünden emin olmalıyız.
                # Şu anki Backtester yapısı sadece rapor sözlüğü döndürüyor.
                # Görselleştirme için trade loglarını kullanacağız.
                return await bt.run_backtest(symbol, days)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_test())
                
                if "error" in result:
                    st.error(f"Backtest Failed: {result['error']}")
                else:
                    # Sonuçlar
                    m1, m2, m3, m4 = st.columns(4)
                    roi = result['roi']
                    pnl = result['final_balance'] - start_bal
                    m1.metric("Net ROI", f"{roi:.2f}%", delta_color="normal" if roi >= 0 else "inverse")
                    m2.metric("Win Rate", f"{result['win_rate']:.1f}%")
                    m3.metric("Trades", result['total_trades'])
                    m4.metric("Net Profit", f"${pnl:,.2f}")
                    
                    st.metric("Final Wallet Balance", f"${result['final_balance']:,.2f}")
                    
                    if result['trades']:
                        trades_df = pd.DataFrame(result['trades'])
                        
                        # --- YENİ: GELİŞMİŞ GRAFİK ---
                        st.markdown("### 📈 Advanced Performance Chart")
                        # Not: Backtester şu an tam mum verisini döndürmüyor, sadece trade loglarını döndürüyor.
                        # Tam mum grafiği için Backtester'ın yapısını değiştirmemiz gerekir.
                        # Şimdilik Bakiye Eğrisini Plotly ile çizelim.
                        
                        import plotly.express as px
                        fig = px.line(trades_df[trades_df['action']=='SELL'], x='time', y='balance', title='Portfolio Growth Equity Curve')
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("📜 Transaction Logs")
                        st.dataframe(trades_df, use_container_width=True)
                    else:
                        st.warning("No trades executed.")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 4. STRATEGY OPTIMIZER (GERİ GELDİ!)
# ==========================================
elif page == "⚙️ Strategy Optimizer":
    st.header("🧬 Genetic Strategy Optimizer")
    st.markdown("Yapay zeka, geçmiş verileri tarayarak en karlı ayarları bulur.")
    
    target_sym = st.selectbox("Target Asset", Config.TARGET_COINS, key="opt_sym")
    opt_days = st.slider("Optimization Period", 15, 60, 30, key="opt_days")
    
    if st.button("🧬 FIND BEST SETTINGS"):
        with st.spinner("Running Grid Search..."):
            async def run_opt():
                opt = StrategyOptimizer()
                return await opt.optimize(target_sym, opt_days)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(run_opt())
            
            best = results['best_config']
            if best:
                st.success(f"✅ Optimization Complete! Best ROI: {best['roi']:.2f}%")
                st.json(best['params'])
                
                st.subheader("📊 Leaderboard")
                df_res = pd.DataFrame(results['all_results'])
                df_res['params'] = df_res['params'].astype(str)
                st.dataframe(df_res.sort_values('roi', ascending=False))
            else:
                st.error("Optimization failed.")
