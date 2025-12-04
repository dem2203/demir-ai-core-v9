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
st.caption("v23.0 | Zero-Mock | On-Chain Intel | Liquidation Hunter | Wyckoff | Adaptive AI")

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
        
        # Veriyi zenginleştir (Kelly Size + Whale Walls ekle)
        display_data = []
        for sym, info in data.items():
            info_copy = info.copy()
            conf = info.get('ai_confidence', 0)
            # Kelly Hesapla
            kelly_size = risk_manager.calculate_kelly_size(conf) if info.get('ai_decision') != 'NEUTRAL' else 0
            info_copy['kelly_size'] = kelly_size
            
            # Whale Walls (Order Book)
            info_copy['whale_support'] = info.get('whale_support', 0)
            info_copy['whale_resistance'] = info.get('whale_resistance', 0)
            
            display_data.append(info_copy)
            
        df_display = pd.DataFrame(display_data)
        
        cols = ['symbol', 'price', 'ai_decision', 'ai_confidence', 'kelly_size', 'fractal_score', 
                'whale_support', 'whale_resistance', 'orderbook_imbalance', 'hurst',
                'wyckoff_phase', 'pattern_bias', 'onchain_signal', 'adaptive_strategy']
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
                    "whale_support": st.column_config.NumberColumn("Whale Support", format="$%.0f"),
                    "whale_resistance": st.column_config.NumberColumn("Whale Resistance", format="$%.0f"),
                    "orderbook_imbalance": st.column_config.NumberColumn("OB Imbalance", format="%.2f"),
                    "wyckoff_phase": st.column_config.TextColumn("Wyckoff"),
                    "pattern_bias": st.column_config.TextColumn("Bias"),
                    "onchain_signal": st.column_config.TextColumn("On-Chain"),
                    "adaptive_strategy": st.column_config.TextColumn("Strategy"),
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
                    
                    # PHASE 4A: Whale Walls
                    whale_sup = info.get('whale_support', 0)
                    whale_res = info.get('whale_resistance', 0)
                    if whale_sup > 0:
                        st.info(f"🐋 **Whale Support:** ${whale_sup:,.0f}")
                    if whale_res > 0:
                        st.warning(f"🐋 **Whale Resistance:** ${whale_res:,.0f}")
                    
                    # Order Book Imbalance
                    imbalance = info.get('orderbook_imbalance', 0)
                    if abs(imbalance) > 0.1:
                        st.caption(f"Order Book Imbalance: {imbalance*100:.1f}% ({'BULLISH' if imbalance > 0 else 'BEARISH'})")
                    
                    # --- PHASE 8: AI Superpowers Display ---
                    st.markdown("---")
                    st.markdown("**🧠 AI Superpowers**")
                    
                    # On-Chain Intelligence
                    onchain_sig = info.get('onchain_signal', 'N/A')
                    onchain_score = info.get('onchain_score', 0)
                    if onchain_sig != 'N/A':
                        color = "🟢" if "BUY" in onchain_sig or "STRONG" in onchain_sig else "🔴" if "SELL" in onchain_sig else "🟡"
                        st.write(f"🐋 On-Chain: {color} **{onchain_sig}** (Score: {onchain_score})")
                    
                    # Liquidation Data
                    liq_sig = info.get('liq_signal', 'N/A')
                    magnet = info.get('magnet_price', 0)
                    if magnet > 0:
                        st.write(f"🎯 Liquidation: **{liq_sig}** | Magnet: ${magnet:,.0f}")
                    
                    # Pattern Analysis
                    wyckoff = info.get('wyckoff_phase', 'N/A')
                    pattern = info.get('pattern_bias', 'NEUTRAL')
                    structure = info.get('market_structure', 'N/A')
                    st.write(f"📊 Wyckoff: **{wyckoff}** | Bias: **{pattern}**")
                    st.write(f"📈 Structure: **{structure}**")
                    
                    # Adaptive Strategy
                    adaptive = info.get('adaptive_strategy', 'N/A')
                    risk_mult = info.get('risk_multiplier', 1.0)
                    st.write(f"🧠 Strategy: **{adaptive}** | Risk Mult: **{risk_mult:.1f}x**")
                    
                    # --- PHASE 9: Technical Analysis Display ---
                    st.markdown("---")
                    st.markdown("**📐 Technical Analysis**")
                    
                    tech_bias = info.get('tech_bias', 'N/A')
                    bias_color = "🟢" if "BULLISH" in tech_bias else "🔴" if "BEARISH" in tech_bias else "🟡"
                    st.write(f"**Technical Bias:** {bias_color} **{tech_bias}**")
                    
                    # Candlestick Patterns
                    candle_count = info.get('candlestick_count', 0)
                    candle_latest = info.get('candlestick_latest', None)
                    if candle_count > 0:
                        st.write(f"🕯️ Candlestick: **{candle_latest}** ({candle_count} pattern)")
                    
                    # Chart Patterns
                    chart_count = info.get('chart_pattern_count', 0)
                    chart_latest = info.get('chart_pattern_latest', None)
                    if chart_count > 0:
                        st.write(f"📐 Chart Pattern: **{chart_latest}**")
                    
                    # Divergence
                    div_count = info.get('divergence_count', 0)
                    div_latest = info.get('divergence_latest', None)
                    if div_count > 0:
                        st.warning(f"⚠️ **Divergence:** {div_latest}")
                    
                    # Fibonacci
                    fib_sup = info.get('fib_support', 0)
                    fib_res = info.get('fib_resistance', 0)
                    if fib_sup > 0:
                        st.write(f"📏 Fib Support: **${fib_sup:,.0f}** | Resistance: **${fib_res:,.0f}**")
                    
                    # Pivot Points
                    pivot = info.get('pivot', 0)
                    pivot_sup = info.get('pivot_support', 0)
                    pivot_res = info.get('pivot_resistance', 0)
                    if pivot > 0:
                        st.write(f"📍 Pivot: **${pivot:,.0f}** | S: ${pivot_sup:,.0f} | R: ${pivot_res:,.0f}")
                    
                    # Volume Signal
                    vol_sig = info.get('volume_signal', 'N/A')
                    if vol_sig != 'N/A':
                        vol_color = "🟢" if "BULLISH" in vol_sig else "🔴" if "DISTRIBUTION" in vol_sig else "🟡"
                        st.write(f"📊 Volume: {vol_color} **{vol_sig}**")

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
