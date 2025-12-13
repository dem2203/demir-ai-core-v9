import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
from src.config.settings import Config
from src.execution.paper_trader import PaperTrader
from src.core.risk_manager import RiskManager # Yeni
from src.utils.translator import Translator  # Turkish explanations (Türkçe açıklamalar)
from src.brain.turkish_narrative import TurkishNarrativeEngine  # AI Reasoning (AI Yorumları)

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
    "🧠 AI Reasoning",
    "🧠 Neural Brain Monitor",
    "📈 Live Trading Chart",
    "💼 Advisory Portfolio", 
    "🧪 Backtest Lab",
    "⚙️ Strategy Optimizer",
    "🔧 Debug"
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

        # ======================================
        # EARLY WARNINGS SECTION (Proactive Alerts)
        # ======================================
        early_warnings = main_info.get('early_warnings', [])
        if early_warnings:
            st.markdown("---")
            st.markdown("### ⚡ Early Warnings (Erken Uyarılar)")
            st.caption("_(Hareket OLMADAN ÖNCE tespit edilen fırsatlar ve riskler)_")
            
            for w in early_warnings[:4]:  # Max 4 warnings
                priority = w.get('priority', 'LOW')
                priority_emoji = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '⚪'
                }.get(priority, '⚪')
                
                with st.expander(f"{priority_emoji} {w.get('title', 'Warning')}", expanded=(priority in ['CRITICAL', 'HIGH'])):
                    st.write(w.get('message', ''))
                    st.info(f"➡️ **Tavsiye:** {w.get('action', '')}")
        
        # ======================================
        # MARKET CORRELATIONS SECTION (PHASE 22)
        # ======================================
        st.markdown("---")
        st.markdown("### 🌐 Market Correlations & Derivatives")
        
        cor_col1, cor_col2, cor_col3, cor_col4, cor_col5 = st.columns(5)
        
        # Correlation Data (from JSON if available)
        corr_data = main_info.get('correlations', {})
        deriv_data = main_info.get('derivatives', {})
        
        # Gold
        gold = corr_data.get('gold', 0)
        gold_chg = corr_data.get('gold_change', 0)
        cor_col1.metric("🥇 Gold", f"${gold:,.0f}" if gold else "N/A", f"{gold_chg:+.1f}%" if gold else None)
        
        # Nasdaq 
        nasdaq = corr_data.get('nasdaq', 0)
        nasdaq_chg = corr_data.get('nasdaq_change', 0)
        cor_col2.metric("📈 Nasdaq", f"{nasdaq:,.0f}" if nasdaq else "N/A", f"{nasdaq_chg:+.1f}%" if nasdaq else None)
        
        # BTC Dominance
        btc_d = corr_data.get('btc_dominance', 0)
        cor_col3.metric("₿ BTC.D", f"{btc_d:.1f}%" if btc_d else "N/A")
        
        # Open Interest
        oi = deriv_data.get('open_interest', 0)
        cor_col4.metric("📊 Open Interest", f"{oi:,.0f}" if oi else "N/A")
        
        # Long/Short Ratio
        ls_ratio = deriv_data.get('long_short_ratio', 0)
        ls_color = "normal" if ls_ratio > 1 else "inverse" if ls_ratio < 1 else "off"
        cor_col5.metric("📊 L/S Ratio", f"{ls_ratio:.2f}" if ls_ratio else "N/A", delta_color=ls_color)
        
        
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
                    "price": st.column_config.NumberColumn(Translator.t("Price"), format="$%.2f"),
                    "ai_confidence": st.column_config.ProgressColumn(Translator.t("Confidence"), format="%.1f%%", min_value=0, max_value=100),
                    "kelly_size": st.column_config.NumberColumn(Translator.t("Kelly Risk (%)"), format="%.2f%%"),
                    "fractal_score": st.column_config.NumberColumn(Translator.t("Fractal Score"), format="%.1f"),
                    "hurst": st.column_config.NumberColumn(Translator.t("Hurst Exp"), format="%.2f"),
                    "whale_support": st.column_config.NumberColumn(Translator.t("Whale Support"), format="$%.0f"),
                    "whale_resistance": st.column_config.NumberColumn(Translator.t("Whale Resistance"), format="$%.0f"),
                    "orderbook_imbalance": st.column_config.NumberColumn(Translator.t("OB Imbalance"), format="%.2f"),
                    "wyckoff_phase": st.column_config.TextColumn(Translator.t("Wyckoff")),
                    "pattern_bias": st.column_config.TextColumn(Translator.t("Bias")),
                    "onchain_signal": st.column_config.TextColumn(Translator.t("On-Chain")),
                    "adaptive_strategy": st.column_config.TextColumn(Translator.t("Strategy")),
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
                    
                    # --- PHASE 13: Sentiment Analysis Display ---
                    st.markdown("---")
                    st.markdown("**📰 Market Sentiment**")
                    
                    sentiment_data = info.get('sentiment_data', {})
                    if sentiment_data:
                        sentiment = sentiment_data.get('sentiment', 'NEUTRAL')
                        fg_index = sentiment_data.get('fear_greed_index', 50)
                        comp_score = sentiment_data.get('composite_score', 0)
                        
                        # Color coding
                        sent_color = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "🟡"
                        
                        # Fear & Greed visualization
                        if fg_index >= 75:
                            fg_label = "Extreme Greed"
                            fg_emoji = "🔥"
                        elif fg_index >= 55:
                            fg_label = "Greed"
                            fg_emoji = "😁"
                        elif fg_index >= 45:
                            fg_label = "Neutral"
                            fg_emoji = "😐"
                        elif fg_index >= 25:
                            fg_label = "Fear"
                            fg_emoji = "😰"
                        else:
                            fg_label = "Extreme Fear"
                            fg_emoji = "😱"
                        
                        st.write(f"**Sentiment:** {sent_color} **{sentiment}** (Score: {comp_score:.2f})")
                        st.write(f"**Fear & Greed:** {fg_emoji} **{fg_label}** ({fg_index}/100)")


# ==========================================
# 2. AI REASONING (Turkish Narrative Analysis)
# ==========================================
elif page == "🧠 AI Reasoning":
    st.header("🧠 AI Reasoning - Detaylı Türkçe Analiz")
    st.caption("Yapay zekanın her kararının detaylı açıklaması")
    
    if st.button('🔄 Yenile / Refresh'): st.rerun()
    
    data = load_json("dashboard_data.json")
    
    if not data:
        st.warning("📡 Veri bekleniyor... / Waiting for data...")
        time.sleep(2)
        st.rerun()
    else:
        # Symbol selector
        symbols = list(data.keys())
        selected_symbol = st.selectbox("📊 Sembol Seçin / Select Symbol", symbols, index=0)
        
        if selected_symbol and selected_symbol in data:
            st.markdown("---")
            
            # Generate narrative report
            snapshot = data[selected_symbol]
            report = TurkishNarrativeEngine.generate_full_report(selected_symbol, snapshot)
            
            # Display report in markdown
            st.markdown(report)
        else:
            st.info("Sembol seçin / Select a symbol")

# ==========================================
# 3. NEURAL BRAIN MONITOR (Visual Intelligence)
# ==========================================
elif page == "🧠 Neural Brain Monitor":
    st.header("🧠 Neural Brain Monitor")
    st.caption("Visualizing the internal state of the Reinforcement Learning Agent.")
    
    if st.button('🔄 Refresh Brain State'): st.rerun()
    
    data = load_json("dashboard_data.json")
    if not data:
        st.warning("Waiting for Brain Data...")
        st.stop()
        
    import plotly.graph_objects as go
    
    # Ana sembolü al
    main_symbol = Config.TARGET_COINS[0]
    info = data.get(main_symbol, {})
    brain_state = info.get('brain_state', {})
    
    if not brain_state:
        st.info("Brain State not initialized yet. Waiting for first RL decision...")
    else:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("### 🤖 Agent Status")
            # FIX: Fallback to ai_decision if rl_action is missing/sleeping
            ai_decision = info.get('ai_decision', 'NEUTRAL')
            decision_map = {"SELL": 0, "HOLD": 1, "BUY": 2, "NEUTRAL": 1}
            rl_action = brain_state.get('rl_action', -1)
            
            if rl_action == -1 and ai_decision in decision_map:
                 rl_action = decision_map[ai_decision]
            
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY", -1: "SLEEPING"}
            action_color = {0: "red", 1: "gray", 2: "green", -1: "gray"}
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #161b22; border-radius: 10px; border: 2px solid {action_color.get(rl_action, 'gray')};">
                <h1 style="color: {action_color.get(rl_action, 'gray')}; font-size: 48px; margin: 0;">{action_map.get(rl_action, 'UNKNOWN')}</h1>
                <p style="color: #8b949e; margin-top: 10px;">Current RL Action</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.metric(Translator.t("Confidence"), f"{info.get('ai_confidence', 0):.1f}%")
            
            # PHASE 12: Visual Cortex Display
            visual_data = info.get('visual_analysis', {})
            v_score = visual_data.get('visual_score', 50)
            v_trend = visual_data.get('trend', 'NEUTRAL')
            dual_vision = visual_data.get('dual_vision', False)
            agreement = visual_data.get('agreement', 'N/A')
            
            st.markdown(f"### 👁️ Visual Cortex - {main_symbol}")
            st.caption("(Grafiksel Yapay Zeka Analizi - Chart'a bakarak görsel tahmin yapar)")
            
            if dual_vision:
                st.success("🔥 DUAL VISION ACTIVE - Cross-validation enabled!")
                c_v1, c_v2, c_v3 = st.columns(3)
                c_v1.metric(Translator.t("Consensus Score"), f"{v_score}/100", delta=v_score-50)
                c_v2.metric(Translator.t("Consensus Trend"), v_trend, delta="normal" if v_trend=="BULLISH" else "inverse" if v_trend=="BEARISH" else "off")
                
                # Agreement Level
                if agreement == "STRONG":
                    c_v3.metric(Translator.t("Agreement"), "✅ STRONG", delta_color="normal")
                elif agreement == "MODERATE":
                    c_v3.metric(Translator.t("Agreement"), "⚠️ MODERATE", delta_color="off")
                else:
                    c_v3.metric(Translator.t("Agreement"), "❌ CONFLICT", delta_color="inverse")
                
                # Individual AI Scores
                with st.expander("👁️ Individual AI Opinions"):
                    gem_score = visual_data.get('gemini_score', 50)
                    gpt_score = visual_data.get('gpt_score', 50)
                    st.write(f"🟢 **Gemini:** {gem_score}/100")
                    st.write(f"🔵 **GPT-4o:** {gpt_score}/100")
            else:
                c_v1, c_v2 = st.columns(2)
                c_v1.metric(Translator.t("Visual Score"), f"{v_score}/100", delta=v_score-50)
                c_v2.metric(Translator.t("Visual Trend"), v_trend, delta="normal" if v_trend=="BULLISH" else "inverse" if v_trend=="BEARISH" else "off")
            
            if visual_data.get('pattern') and visual_data['pattern'] != 'None':
                st.info(f"📐 Pattern Detected: **{visual_data['pattern']}**")
            
            with st.expander("👁️ Visual Analysis Reasoning (Görsel Analiz Açıklaması)"):
                reasoning = visual_data.get('reasoning', '')
                if '429' in reasoning or 'quota' in reasoning.lower():
                    st.error("⚠️ **API Limit Aşıldı (Quota Exceeded)**")
                    st.caption("Google Gemini API günlük ücretsiz limiti dolmuş. Çözümler:")
                    st.markdown("""
                    1. **Bekleyin:** Limit 24 saat sonra sıfırlanır
                    2. **Ücretli Plan:** [Google AI Studio](https://ai.google.dev/pricing)'dan kota artırın
                    3. **Alternatif:** OpenAI API key ekleyerek GPT-4o'yu aktif edin
                    """)
                else:
                    st.caption(reasoning if reasoning else 'Görsel analiz şu an aktif değil.')
            
            st.markdown("---")
            
            # HTF Trend Göstergesi
            htf_trend = "NEUTRAL"
            if "4H Trend: BULL" in info.get('reason', ''): htf_trend = "BULLISH"
            elif "4H Trend: BEAR" in info.get('reason', ''): htf_trend = "BEARISH"
            
            delta_color = "normal" if htf_trend == "BULLISH" else "inverse" if htf_trend == "BEARISH" else "off"
            st.metric("4H Trend (Eagle Eye)", htf_trend, delta_color=delta_color)
            
        with c2:
            st.markdown("### 🧠 Attention Map")
            
            # Radar Chart Verisi
            categories = ['Technical', 'Pattern', 'LSTM Model', 'HTF Trend', 'On-Chain']
            values = [
                brain_state.get('tech_attention', 0),
                brain_state.get('pattern_attention', 0),
                brain_state.get('lstm_attention', 0),
                brain_state.get('htf_attention', 0),
                brain_state.get('onchain_attention', 0)
            ]
            
            # Normalize (0-1 arası) - Görsel güzellik için mutlak değerlerin toplamına bölünebilir veya max'a
            # Burada basitçe mutlak değerleri gösteriyoruz, zaten 0-0.5 arası genelde.
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='AI Focus',
                line_color='#00ff00' if info.get('ai_decision') == 'BUY' else '#ff0000'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 0.5] # Tahmini max ağırlık
                    ),
                    bgcolor='#0e1117'
                ),
                paper_bgcolor='#0e1117',
                font_color='#e0e0e0',
                showlegend=False,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("### 📝 Decision Logic Breakdown")
        st.json(brain_state)
        
        # PHASE 15: Portfolio Analytics
        st.markdown("---")
        st.subheader("🎯 Portfolio Analytics")
        
        try:
            from src.core.portfolio_optimizer import PortfolioOptimizer
            from src.data_ingestion.connectors.binance_connector import BinanceConnector
            import plotly.express as px
            
            optimizer = PortfolioOptimizer()
            connector = BinanceConnector()
            
            # Fetch price data
            symbols_list = ["BTC/USDT", "ETH/USDT", "LTC/USDT"]
            price_data = {}
            
            for sym in symbols_list:
                df_price = connector.fetch_ohlcv(sym, timeframe='1d', limit=30)
                if not df_price.empty:
                    price_data[sym.replace("/USDT", "")] = df_price
            
            if len(price_data) >= 2:
                corr_matrix = optimizer.calculate_correlation_matrix(price_data, period=30)
                
                if not corr_matrix.empty:
                    # Correlation Heatmap
                    fig_corr = px.imshow(
                        corr_matrix,
                        labels=dict(x="Symbol", y="Symbol", color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        color_continuous_scale='RdYlGn',
                        zmin=-1, zmax=1,
                        text_auto='.2f'
                    )
                    fig_corr.update_layout(
                        title="30-Day Correlation Matrix",
                        height=350,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Portfolio Metrics
                    portfolio_data = load_json("portfolio.json")
                    if portfolio_data and portfolio_data.get('positions'):
                        analytics = optimizer.get_portfolio_analytics(
                            portfolio_data['positions'], 
                            corr_matrix
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Concentration", f"{analytics['concentration']:.2f}")
                        col2.metric("Diversification", f"{analytics['diversification_score']:.2f}")
                        
                        risk = analytics['correlation_risk']
                        risk_emoji = "🔴" if risk in ["HIGH", "CRITICAL"] else "🟡" if risk == "MEDIUM" else "🟢"
                        col3.metric("Correlation Risk", f"{risk_emoji} {risk}")
                else:
                    st.info("Insufficient data for correlation analysis")
            else:
                st.info("Need at least 2 symbols for correlation")
        except Exception as e:
            st.error(f"Portfolio Analytics Error: {str(e)}")

# ==========================================
# 2. SANAL CÜZDAN (Advisory Portfolio)
# ==========================================

# ==========================================
# 3. LIVE TRADING CHART (Phase 14)
# ==========================================
elif page == "📈 Live Trading Chart":
    st.title("📈 Live Paper Trading Chart")
    
    # Import chart visualizer
    from src.core.chart_visualizer import ChartVisualizer
    from src.data_ingestion.connectors.binance_connector import BinanceConnector
    import json
    import os
    
    visualizer = ChartVisualizer()
    connector = BinanceConnector()
    
    # Symbol and Timeframe Selectors
    col1, col2 = st.columns(2)
    with col1:
        symbol_select = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "LTC/USDT"])
    with col2:
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
    
    try:
        # Fetch OHLCV data
        df = connector.fetch_ohlcv(symbol_select, timeframe=timeframe, limit=200)
        
        if df.empty:
            st.error("No data available")
        else:
            # Load paper trades
            portfolio_path = "src/execution/portfolio.json"
            trades = []
            current_position = None
            
            if os.path.exists(portfolio_path):
                with open(portfolio_path, 'r') as f:
                    portfolio_data = json.load(f)
                    
                # Get trades for this symbol
                symbol_upper = symbol_select.replace("/", "")  # BTC/USDT -> BTCUSDT
                if symbol_upper in portfolio_data.get('trades', {}):
                    trades = portfolio_data['trades'][symbol_upper]
                
                # Get current position
                positions = portfolio_data.get('positions', {})
                if symbol_upper in positions:
                    pos = positions[symbol_upper]
                    if pos.get('size', 0) != 0:  # Position is open
                        current_position = pos
            
            # Create chart
            fig = visualizer.create_trading_chart(
                df=df,
                symbol=symbol_select,
                trades=trades,
                current_position=current_position
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show position summary
            if current_position:
                st.success("📊 **ACTIVE POSITION**")
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                
                side = current_position.get('side', 'LONG')
                entry = current_position.get('entry_price', 0)
                size = current_position.get('size', 0)
                current_price = df['close'].iloc[-1]
                
                # Calculate unrealized P&L
                if side == 'LONG':
                    pnl = (current_price - entry) * size
                else:
                    pnl = (entry - current_price) * size
                
                col_p1.metric("Side", side, delta_color="normal" if side == "LONG" else "inverse")
                col_p2.metric("Entry", f"${entry:,.2f}")
                col_p3.metric("Size", f"{size:.4f}")
                col_p4.metric("Unrealized P&L", f"${pnl:,.2f}", delta=pnl, delta_color="normal" if pnl > 0 else "inverse")
            else:
                st.info("No active position")
                
    except Exception as e:
        st.error(f"Error loading chart: {e}")
        import traceback
        st.code(traceback.format_exc())

# ==========================================
# 4. ADVISORY PORTFOLIO
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
        m1.metric(Translator.t("Total Equity"), f"${equity:,.2f}")
        m2.metric(Translator.t("Cash Balance"), f"${balance:,.2f}")
        m3.metric(Translator.t("PnL"), f"${pnl_total:,.2f}", delta_color="normal" if pnl_total >= 0 else "inverse")
        
        st.markdown("---")
        st.subheader("🔓 Open Positions")
        
        if positions_data:
            df_pos = pd.DataFrame(positions_data)
            st.dataframe(
                df_pos, 
                use_container_width=True,
                column_config={
                    "entry": st.column_config.NumberColumn(Translator.t("Entry Price"), format="$%.2f"),
                    "current": st.column_config.NumberColumn(Translator.t("Current Price"), format="$%.2f"),
                    "pnl": st.column_config.NumberColumn(Translator.t("PnL"), format="$%.2f"),
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
                from src.backtest.backtester import Backtester
                bt = Backtester()
                return await bt.run_backtest(symbol, days)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(run())
            
            if "error" in res: st.error(res['error'])
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric(Translator.t("ROI"), f"{res['roi']:.2f}%")
                c2.metric(Translator.t("Win Rate"), f"{res['win_rate']:.1f}%")
                c3.metric(Translator.t("Total Trades"), res['total_trades'])
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
                from src.brain.optimizer import StrategyOptimizer
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

# ==========================================
# 5. DEBUG PANEL
# ==========================================
elif page == "🔧 Debug":
    st.header("🔧 System Debug Panel")
    
    st.subheader("🔑 Environment Check")
    keys = {
        "BINANCE_API_KEY": bool(Config.BINANCE_API_KEY),
        "GOOGLE_API_KEY": bool(Config.GEMINI_API_KEY),
        "OPENAI_API_KEY": bool(Config.OPENAI_API_KEY),
        "FRED_API_KEY": bool(Config.FRED_API_KEY),
        "TELEGRAM_TOKEN": bool(Config.TELEGRAM_TOKEN)
    }
    st.json(keys)
    
    st.subheader("⚙️ Configuration")
    st.write(f"Target Coins: {Config.TARGET_COINS}")
    
    st.subheader("📄 Dashboard Data (Raw)")
    data = load_json("dashboard_data.json")
    st.json(data)
    
    st.subheader("💼 Portfolio Data (Raw)")
    port = load_json("portfolio.json")
    st.json(port)
