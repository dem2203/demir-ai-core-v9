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
# HELPER: Per-Coin Collapsible Section
# ==========================================
def render_coin_section(symbol: str, coin_data: dict, expanded: bool = False):
    """Renders a collapsible section for a single coin with all features."""
    
    price = coin_data.get('price', 0)
    dec = coin_data.get('ai_decision', 'NEUTRAL')
    conf = coin_data.get('ai_confidence', 0)
    
    # Signal emoji
    signal_emoji = "🟢" if dec == "BUY" else "🔴" if dec == "SELL" else "⚪"
    
    # Expander with signal info in title
    with st.expander(f"{signal_emoji} **{symbol}** | ${price:,.2f} | {dec} ({conf:.0f}%)", expanded=expanded):
        
        # Row 1: Basic metrics with Turkish explanations
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("💰 Fiyat", f"${price:,.2f}")
            st.caption('_"Güncel piyasa fiyatı"_')
        
        with m2:
            signal_tr = "AL" if dec == "BUY" else "SAT" if dec == "SELL" else "BEKLE"
            st.metric("🎯 Sinyal", signal_tr, f"{conf:.0f}%")
            signal_explain = "Alım zamanı" if dec == "BUY" else "Satış zamanı" if dec == "SELL" else "Bekle, sinyal yok"
            st.caption(f'_"{signal_explain}"_')
        
        with m3:
            regime = coin_data.get('regime', 'N/A')
            if "BULL" in regime:
                regime_tr = "📈 BOĞA PİYASASI"
                regime_explain = "Fiyatlar yükseliyor, alım avantajlı"
            elif "BEAR" in regime:
                regime_tr = "📉 AYI PİYASASI"
                regime_explain = "Fiyatlar düşüyor, satış düşün"
            elif "RANGE" in regime:
                regime_tr = "↔️ YATAY"
                regime_explain = "Fiyat dar aralıkta, kırılım bekle"
            else:
                regime_tr = "❓ BELİRSİZ"
                regime_explain = "Trend belirlenemedi"
            st.metric("📊 Piyasa Durumu", regime_tr)
            st.caption(f'_"{regime_explain}"_')
        
        with m4:
            fractal = coin_data.get('fractal_score', 0)
            st.metric("📈 Fraktal Uyumu", f"{fractal:.0f}%")
            if fractal >= 80:
                frac_explain = "Mükemmel! Tüm zaman dilimleri aynı yönde"
            elif fractal >= 60:
                frac_explain = "İyi uyum, güvenilir sinyal"
            elif fractal >= 40:
                frac_explain = "Orta uyum, dikkatli ol"
            else:
                frac_explain = "Zayıf uyum, sinyal güvenilmez"
            st.caption(f'_"{frac_explain}"_')
        
        # SMC Section - Enhanced with visible explanations
        st.markdown("#### 🎯 Smart Money Concepts (Akıllı Para Konseptleri)")
        st.caption("_Büyük oyuncuların (bankalar, hedge fund'lar) hareket ettikleri bölgeler_")
        smc = coin_data.get('smc', {})
        smc_signal = smc.get('smc_signal', {})
        
        s1, s2, s3, s4 = st.columns(4)
        
        # SMC Bias
        with s1:
            smc_bias = smc.get('smc_bias', 'N/A')
            bias_emoji = "🟢" if smc_bias == "BULLISH" else "🔴" if smc_bias == "BEARISH" else "⚪"
            bias_tr = "YÜKSELİŞ" if smc_bias == "BULLISH" else "DÜŞÜŞ" if smc_bias == "BEARISH" else "NÖTR"
            st.metric("SMC Yön", f"{bias_emoji} {bias_tr}")
            bias_explain = "Alım bölgesi" if smc_bias == "BULLISH" else "Satış bölgesi" if smc_bias == "BEARISH" else "Belirsiz bölge"
            st.caption(f'_"{bias_explain}"_')
        
        # Order Blocks with price levels
        with s2:
            obs = smc.get('order_blocks', [])
            st.metric("Emir Blokları", f"{len(obs)} aktif" if obs else "0")
            if obs:
                # SMC analyzer uses 'top' and 'bottom' for OB ranges
                ob_prices = ", ".join([f"${ob.get('bottom', 0):,.0f}-${ob.get('top', 0):,.0f}" for ob in obs[:2]])
                st.caption(f'_"{ob_prices}"_')
            else:
                st.caption('_"Destek/direnç bölgesi yok"_')
        
        # FVGs with price ranges
        with s3:
            fvgs = smc.get('fvgs', [])
            st.metric("Boşluklar (FVG)", f"{len(fvgs)} adet" if fvgs else "0")
            if fvgs:
                # SMC analyzer uses 'bottom' and 'top' for FVG ranges
                fvg_ranges = ", ".join([f"${fvg.get('bottom', 0):,.0f}-${fvg.get('top', 0):,.0f}" for fvg in fvgs[:2]])
                st.caption(f'_"{fvg_ranges}"_')
            else:
                st.caption('_"Fiyat boşluğu yok"_')
        
        # Strength
        with s4:
            strength = smc_signal.get('strength', 0)
            st.metric("Güç", f"{strength}%")
            strength_text = "Çok güçlü sinyal" if strength >= 70 else "Orta sinyal" if strength >= 40 else "Zayıf sinyal"
            st.caption(f'_"{strength_text}"_')
        
        
        # MTF Section with detailed Turkish explanations
        st.markdown("#### 📊 Çoklu Zaman Dilimi Analizi")
        st.caption("_1 saatlik, 4 saatlik ve günlük grafiklerin aynı yönü gösterip göstermediği_")
        mtf = coin_data.get('mtf', {})
        trends = mtf.get('trends', {})
        
        t1, t2, t3, t4, t5 = st.columns(5)
        
        tf_labels = {
            '1h': ('1 Saat', 'Kısa vadeli yön'),
            '4h': ('4 Saat', 'Orta vadeli yön'),
            '1d': ('1 Gün', 'Uzun vadeli yön')
        }
        
        for col, (tf, (label, desc)) in zip([t1, t2, t3], tf_labels.items()):
            trend = trends.get(tf, {}).get('trend', 'N/A')
            with col:
                if trend == "BULLISH":
                    st.metric(label, "🟢 YÜKSELİŞ")
                    st.caption(f'_"{desc}: Alım fırsatı"_')
                elif trend == "BEARISH":
                    st.metric(label, "🔴 DÜŞÜŞ")
                    st.caption(f'_"{desc}: Satış düşün"_')
                else:
                    st.metric(label, "⚪ KARARSIZ")
                    st.caption(f'_"{desc}: Bekle"_')
        
        with t4:
            confluence = mtf.get('confluence_score', 0)
            st.metric("Uyum Skoru", f"{confluence}%")
            if confluence >= 80:
                conf_explain = "Mükemmel! Tüm zaman dilimleri hemfikir"
            elif confluence >= 60:
                conf_explain = "İyi uyum, sinyal güçlü"
            elif confluence >= 40:
                conf_explain = "Orta uyum, dikkatli ol"
            else:
                conf_explain = "Zayıf uyum, bekle"
            st.caption(f'_"{conf_explain}"_')
        
        with t5:
            entry_qual = mtf.get('entry_quality', {})
            rating = entry_qual.get('rating', 'N/A')
            rating_map = {
                'EXCELLENT': ('⭐⭐⭐', 'Harika giriş zamanı'),
                'GOOD': ('⭐⭐', 'İyi giriş zamanı'),
                'FAIR': ('⭐', 'Vasat, riskli'),
                'POOR': ('❌', 'Kötü, girme')
            }
            stars, explain = rating_map.get(rating, ('❓', 'Belirsiz'))
            st.metric("Giriş Kalitesi", stars)
            st.caption(f'_"{explain}"_')
        
        # Volume Profile Section - Enhanced with visible explanations
        st.markdown("#### 📈 Volume Profile (Hacim Profili)")
        st.caption("_İşlem hacminin en yoğun olduğu fiyat bölgeleri - Fiyat bu seviyelere çekilme eğilimindedir_")
        vp = coin_data.get('volume_profile', {})
        current_price = price  # Mevcut fiyat
        
        v1, v2, v3, v4 = st.columns(4)
        
        # VPOC - Volume Point of Control
        with v1:
            vpoc = vp.get('vpoc', 0)
            st.metric("VPOC (Mıknatıs Fiyat)", f"${vpoc:,.0f}")
            if vpoc > 0:
                vpoc_diff = ((current_price - vpoc) / vpoc * 100)
                vpoc_text = f"Fiyat {abs(vpoc_diff):.1f}% {'üstünde' if vpoc_diff > 0 else 'altında'}"
            else:
                vpoc_text = "Veri yok"
            st.caption(f'_"{vpoc_text}"_')
        
        # VAH - Value Area High
        with v2:
            vah = vp.get('vah', 0)
            st.metric("VAH (Pahalı Bölge)", f"${vah:,.0f}")
            st.caption('_"Bu üstü = satış baskısı"_')
        
        # VAL - Value Area Low
        with v3:
            val = vp.get('val', 0)
            st.metric("VAL (Ucuz Bölge)", f"${val:,.0f}")
            st.caption('_"Bu altı = alım fırsatı"_')
        
        # Position with detailed explanation
        with v4:
            pos = vp.get('price_position', 'N/A')
            if "ABOVE_VAH" in pos:
                pos_emoji = "🔴"
                pos_tr = "PAHALIDA"
                pos_explain = "Satış baskısı olabilir"
            elif "BELOW_VAL" in pos:
                pos_emoji = "🟢"
                pos_tr = "UCUZDA"
                pos_explain = "Alım fırsatı olabilir"
            elif "ABOVE" in pos:
                pos_emoji = "🟡"
                pos_tr = "ÜSTTE"
                pos_explain = "Değer bölgesi üstünde"
            elif "BELOW" in pos:
                pos_emoji = "🟡"
                pos_tr = "ALTTA"
                pos_explain = "Değer bölgesi altında"
            else:
                pos_emoji = "⚪"
                pos_tr = "DEĞER BÖLGESİNDE"
                pos_explain = "Normal işlem bölgesi"
            st.metric("Konum", f"{pos_emoji} {pos_tr}")
            st.caption(f'_"{pos_explain}"_')
        
        
        # Smart SL/TP Section with Turkish
        sltp = coin_data.get('smart_sltp', {})
        if sltp.get('valid', False) or sltp.get('stop_loss', 0) > 0:
            st.markdown("#### 🎯 Giriş ve Çıkış Seviyeleri")
            st.caption("_Stop Loss: Zarar kes noktası | TP: Kar al seviyeleri | R:R: Risk/Kazanç oranı_")
            
            direction = sltp.get('direction', dec)
            
            e1, e2, e3, e4, e5 = st.columns(5)
            
            with e1:
                if direction in ["LONG", "BUY"]:
                    st.metric("Yön", "🟢 AL")
                    st.caption('_"Yükseliş bekleniyor"_')
                elif direction in ["SHORT", "SELL"]:
                    st.metric("Yön", "🔴 SAT")
                    st.caption('_"Düşüş bekleniyor"_')
                else:
                    st.metric("Yön", "⚪ BEKLE")
                    st.caption('_"Sinyal yok"_')
            
            with e2:
                sl = sltp.get('stop_loss', 0)
                risk_pct = sltp.get('risk_pct', 0)
                st.metric("Zarar Kes", f"${sl:,.0f}", f"-{risk_pct:.1f}%")
                st.caption('_"Bu fiyata düşerse sat"_')
            
            with e3:
                tp1 = sltp.get('take_profit_1', 0)
                rr1 = sltp.get('risk_reward_1', 0)
                st.metric("Hedef 1", f"${tp1:,.0f}", f"R:R {rr1}")
                st.caption('_"İlk kar al seviyesi"_')
            
            with e4:
                tp2 = sltp.get('take_profit_2', 0)
                rr2 = sltp.get('risk_reward_2', 0)
                st.metric("Hedef 2", f"${tp2:,.0f}", f"R:R {rr2}")
                st.caption('_"İkinci kar al"_')
            
            with e5:
                tp3 = sltp.get('take_profit_3', 0)
                rr3 = sltp.get('risk_reward_3', 0)
                st.metric("Hedef 3", f"${tp3:,.0f}", f"R:R {rr3}")
                st.caption('_"Tam hedef"_')
            
            # Quality indicator in Turkish
            quality = sltp.get('quality', 'UNKNOWN')
            quality_map = {
                'EXCELLENT': ('✅ MÜKEMMEL', 'success', 'Güçlü sinyal, yüksek başarı olasılığı!'),
                'GOOD': ('👍 İYİ', 'info', 'Güvenilir setup, normal risk'),
                'FAIR': ('⚠️ ORTA', 'warning', 'Dikkatli ol, risk yüksek'),
                'POOR': ('❌ ZAYIF', 'error', 'Girme, sinyal zayıf')
            }
            label, func, msg = quality_map.get(quality, ('❓ BİLİNMİYOR', 'info', 'Kalite belirlenemedi'))
            getattr(st, func)(f"**{label}** - {msg}")
        
        # Technical summary with Turkish explanations
        st.markdown("#### 📐 Teknik Özet")
        st.caption("_Farklı analiz yöntemlerinin özeti_")
        
        tech1, tech2, tech3, tech4 = st.columns(4)
        
        with tech1:
            tech_bias = coin_data.get('tech_bias', 'N/A')
            if "BULL" in tech_bias:
                st.metric("Teknik Yön", "🟢 YÜKSELİŞ")
                st.caption('_"İndikatörler alım diyor"_')
            elif "BEAR" in tech_bias:
                st.metric("Teknik Yön", "🔴 DÜŞÜŞ")
                st.caption('_"İndikatörler satış diyor"_')
            else:
                st.metric("Teknik Yön", "⚪ NÖTR")
                st.caption('_"İndikatörler kararsız"_')
        
        with tech2:
            pattern = coin_data.get('pattern_bias', 'N/A')
            if "BULL" in pattern:
                st.metric("Grafik Deseni", "🟢 YÜKSELİŞ")
                st.caption('_"Formasyonlar alım gösteriyor"_')
            elif "BEAR" in pattern:
                st.metric("Grafik Deseni", "🔴 DÜŞÜŞ")
                st.caption('_"Formasyonlar satış gösteriyor"_')
            else:
                st.metric("Grafik Deseni", "⚪ BELİRSİZ")
                st.caption('_"Net formasyon yok"_')
        
        with tech3:
            onchain = coin_data.get('onchain_signal', 'N/A')
            if "BUY" in onchain or "STRONG" in onchain:
                st.metric("Zincir Verisi", "🟢 ALIŞ")
                st.caption('_"Balinalar alıyor"_')
            elif "SELL" in onchain:
                st.metric("Zincir Verisi", "🔴 SATIŞ")
                st.caption('_"Balinalar satıyor"_')
            else:
                st.metric("Zincir Verisi", "⚪ NÖTR")
                st.caption('_"Balina aktivitesi normal"_')
        
        with tech4:
            wyckoff = coin_data.get('wyckoff_phase', 'N/A')
            wyckoff_map = {
                'ACCUMULATION': ('🟢 BİRİKİM', 'Akıllı para alım yapıyor'),
                'MARKUP': ('🚀 YÜKSELİŞ', 'Ralli başladı'),
                'DISTRIBUTION': ('🔴 DAĞITIM', 'Akıllı para satıyor'),
                'MARKDOWN': ('📉 DÜŞÜŞ', 'Düşüş rallisi')
            }
            label, explain = wyckoff_map.get(wyckoff, ('❓ BİLİNMİYOR', 'Faz belirlenemedi'))
            st.metric("Wyckoff Fazı", label)
            st.caption(f'_"{explain}"_')

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
        # PER-COIN COLLAPSIBLE SECTIONS (PHASE 25)
        # ======================================
        st.markdown("---")
        st.markdown("### 🎯 Coin-by-Coin Analysis")
        st.caption("_Click to expand each coin for full SMC, MTF, Volume Profile, and SL/TP details_")
        
        # Render each coin with collapsible section
        for i, (symbol, coin_data) in enumerate(data.items()):
            # First coin expanded by default
            render_coin_section(symbol, coin_data, expanded=(i == 0))

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
        
        # ======================================
        # NOTE: SMC/MTF/VP/SL-TP moved to Coin-by-Coin Analysis above
        # Each coin now has its own collapsible section with all features
        # ======================================
        
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

        # AI Reasoning Summary (Simplified - details in coin sections above)
        st.markdown("### 🤖 AI Reasoning Summary")
        st.caption("_Detaylı analiz için yukarıdaki coin bölümlerini açın_")
        
        # Simple 2-column summary
        summary_left, summary_right = st.columns(2)
        
        for i, (symbol, info) in enumerate(data.items()):
            col = summary_left if i % 2 == 0 else summary_right
            with col:
                dec = info.get('ai_decision', 'NEUTRAL')
                conf = info.get('ai_confidence', 0)
                reason = info.get('reason', 'N/A')[:100]  # Truncate
                
                signal_emoji = "🟢" if dec == "BUY" else "🔴" if dec == "SELL" else "⚪"
                st.info(f"{signal_emoji} **{symbol}**: {dec} ({conf:.0f}%)\n\n_{reason}..._")


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
