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
st.caption("v35.0 | 35 AI Modules | CoinGlass | WebSocket | Win Rate Tracking")

# --- Yan Menü ---
page = st.sidebar.radio("System Modules", [
    "📡 Live Market Intelligence", 
    "🔮 AI Predictions",  # NEW: Markov, LSTM, Whale Intel
    "🎯 AI Module Monitor",  # NEW: 35 module status
    "🌐 Web Intelligence",  # All web scraping data
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

# ==========================================
# PHASE 43: TradingView Migration
# Replace yfinance (fails in production) with TradingView scraping
# ==========================================

def fetch_live_market_data():
    """
    Dashboard için canlı piyasa verisi - TradingView scraper kullanır
    
    PHASE 43 UPDATE:
    - ❌ Removed yfinance (production errors)
    - ✅ Added TradingView scraper (reliable real-time data)
    - ✅ Added USDT/USDC dominance (critical missing indicators)
    - ✅ Added ETH dominance
    - ✅ Added DXY, VIX, SPY
    """
    import requests
    from src.brain.tradingview_scraper import TradingViewScraper
    
    result = {
        # Macro data
        'gold': 0, 'gold_change': 0,
        'nasdaq': 0, 'nasdaq_change': 0,
        'dxy': 0, 'dxy_change': 0,
        'vix': 0, 'vix_change': 0,
        'spy': 0, 'spy_change': 0,
        
        # Dominance metrics
        'btc_dominance': 0, 'btc_dominance_change': 0,
        'eth_dominance': 0, 'eth_dominance_change': 0,
        'usdt_dominance': 0, 'usdt_change': 0,
        'usdc_dominance': 0, 'usdc_change': 0,
        'total_stablecoin_dominance': 0,
        'stablecoin_signal': 'NEUTRAL',
        'stablecoin_interpretation': '',
        
        # Derivatives
        'open_interest': 0, 
        'long_short_ratio': 0
    }
    
    try:
        # Initialize TradingView scraper
        tv_scraper = TradingViewScraper()
        
        # Get all macro data
        macro_data = tv_scraper.get_all_macro_data()
        
        # Gold
        gold_data = macro_data.get('gold', {})
        result['gold'] = gold_data.get('price', 0)
        result['gold_change'] = gold_data.get('change', 0)
        
        # Nasdaq
        nasdaq_data = macro_data.get('nasdaq', {})
        result['nasdaq'] = nasdaq_data.get('price', 0)
        result['nasdaq_change'] = nasdaq_data.get('change', 0)
        
        # DXY (US Dollar Index)
        dxy_data = macro_data.get('dxy', {})
        result['dxy'] = dxy_data.get('price', 0)
        result['dxy_change'] = dxy_data.get('change', 0)
        
        # VIX (Volatility Index)
        vix_data = macro_data.get('vix', {})
        result['vix'] = vix_data.get('price', 0)
        result['vix_change'] = vix_data.get('change', 0)
        
        # SPY
        spy_data = macro_data.get('spy', {})
        result['spy'] = spy_data.get('price', 0)
        result['spy_change'] = spy_data.get('change', 0)
        
        # BTC Dominance
        btc_d_data = macro_data.get('btc_dominance', {})
        result['btc_dominance'] = btc_d_data.get('price', 0)
        result['btc_dominance_change'] = btc_d_data.get('change', 0)
        
        # ETH Dominance
        eth_d_data = macro_data.get('eth_dominance', {})
        result['eth_dominance'] = eth_d_data.get('price', 0)
        result['eth_dominance_change'] = eth_d_data.get('change', 0)
        
        # Stablecoin Summary
        stable_summary = tv_scraper.get_stablecoin_summary()
        result['usdt_dominance'] = stable_summary.get('usdt_dominance', 0)
        result['usdc_dominance'] = stable_summary.get('usdc_dominance', 0)
        result['total_stablecoin_dominance'] = stable_summary.get('total_stablecoin_dominance', 0)
        result['stablecoin_signal'] = stable_summary.get('signal', 'NEUTRAL')
        result['stablecoin_interpretation'] = stable_summary.get('interpretation', '')
        
    except Exception as e:
        # Fallback if TradingView fails - still better than yfinance
        import logging
        logging.warning(f"TradingView scraper error: {e}")
    
    # Derivatives data (Binance - unchanged, works well)
    try:
        oi = requests.get("https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT", timeout=5)
        if oi.status_code == 200:
            result['open_interest'] = float(oi.json()['openInterest']) * 100000  # Approximate USD value
    except: pass
    
    try:
        ls = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=1", timeout=5)
        if ls.status_code == 200:
            result['long_short_ratio'] = float(ls.json()[0]['longShortRatio'])
    except: pass
    
    return result

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
        
        # Global Metrics - USE LIVE DATA from browser scraper!
        c1, c2, c3, c4 = st.columns(4)
        
        # DXY/VIX from live browser scraper (not stale engine data)
        live_data_header = fetch_live_market_data()
        dxy = live_data_header.get('dxy', 0) or main_info.get('dxy', 0)
        vix = live_data_header.get('vix', 0) or main_info.get('vix', 0)
        price = main_info.get('price', 0)
        
        c1.metric("🇺🇸 DXY Index", f"{dxy:.2f}" if dxy > 0 else "N/A")
        c2.metric("😨 VIX Index", f"{vix:.2f}" if vix > 0 else "N/A")
        c3.metric(f"₿ {display_symbol}", f"${price:,.2f}" if price > 0 else "N/A")
        
        # AI Signal - Türkçe ve coin ismi belirtilmiş
        dec = main_info.get('ai_decision', 'NEUTRAL')
        conf = main_info.get('ai_confidence', 0)
        
        signal_tr = "AL" if dec == "BUY" else "SAT" if dec == "SELL" else "BEKLE"
        delta_color = "off"
        if dec == "BUY": delta_color = "normal"
        elif dec == "SELL": delta_color = "inverse"
        
        c4.metric(f"🧠 {display_symbol} Sinyal", signal_tr, f"{conf:.1f}% Güven", delta_color=delta_color)
        
        # ======================================
        # DATA VALIDATION STATUS
        # ======================================
        try:
            from src.validation.data_validator import DataValidator
            validator = DataValidator()
            validation = validator.validate_all()
            
            val_col1, val_col2 = st.columns([3, 1])
            with val_col1:
                if validation['is_valid']:
                    st.success(f"✅ Veri Doğrulama: {validation['passed']}/{validation['total_checks']} kontrol geçti")
                elif validation['failed'] == 1:
                    st.warning(f"⚠️ Veri Doğrulama: {validation['passed']}/{validation['total_checks']} kontrol geçti, 1 uyumsuzluk")
                else:
                    st.error(f"❌ Veri Doğrulama: {validation['failed']} uyumsuzluk tespit edildi")
            
            with val_col2:
                with st.expander("Detaylar"):
                    for r in validation['results']:
                        status = "✅" if r.is_valid else "❌"
                        st.text(f"{status} {r.metric}: {r.our_value:.2f} vs {r.reference_value:.2f} ({r.reference_source})")
        except Exception as e:
            st.caption(f"_Veri doğrulama geçici olarak kullanılamıyor_")
        
        # ======================================
        # MARKET CORRELATIONS & DERIVATIVES (DXY/VIX'in hemen altında)
        # ======================================
        st.markdown("---")
        st.markdown("### 🌐 Piyasa Korelasyonları & Türevler")
        st.caption("_🔴 CANLI VERİ - Her 60 saniyede güncellenir_")
        
        cor_col1, cor_col2, cor_col3, cor_col4, cor_col5 = st.columns(5)
        
        # ✅ CANLI VERİ FETCH - Engine'den bağımsız!
        live_data = fetch_live_market_data()
        
        # Snapshot'tan veya canlı veriden al (canlı öncelikli)
        corr_data = {
            'gold': live_data.get('gold') or main_info.get('correlations', {}).get('gold', 0),
            'gold_change': live_data.get('gold_change') or main_info.get('correlations', {}).get('gold_change', 0),
            'nasdaq': live_data.get('nasdaq') or main_info.get('correlations', {}).get('nasdaq', 0),
            'nasdaq_change': live_data.get('nasdaq_change') or main_info.get('correlations', {}).get('nasdaq_change', 0),
            'btc_dominance': live_data.get('btc_dominance') or main_info.get('correlations', {}).get('btc_dominance', 0),
            'btc_dominance_change': live_data.get('btc_dominance_change') or main_info.get('correlations', {}).get('btc_dominance_change', 0),
        }
        deriv_data = {
            # Derivatives
            'open_interest': live_data.get('open_interest') or main_info.get('derivatives', {}).get('open_interest', 0),
            'long_short_ratio': live_data.get('long_short_ratio') or main_info.get('derivatives', {}).get('long_short_ratio', 0),
            # Dominance metrics (from browser scraper)
            'btc_dominance': live_data.get('btc_dominance', 0),
            'btc_dominance_change': live_data.get('btc_dominance_change', 0),
            'eth_dominance': live_data.get('eth_dominance', 0),
            'eth_dominance_change': live_data.get('eth_dominance_change', 0),
            'usdt_dominance': live_data.get('usdt_dominance', 0),
            'usdc_dominance': live_data.get('usdc_dominance', 0),
            'total_stablecoin_dominance': live_data.get('total_stablecoin_dominance', 0),
            'stablecoin_signal': live_data.get('stablecoin_signal', 'NEUTRAL'),
            'stablecoin_interpretation': live_data.get('stablecoin_interpretation', ''),
            # Macro indicators
            'dxy': live_data.get('dxy', 0),
            'dxy_change': live_data.get('dxy_change', 0),
            'vix': live_data.get('vix', 0),
            'vix_change': live_data.get('vix_change', 0),
            'spy': live_data.get('spy', 0),
            'spy_change': live_data.get('spy_change', 0),
        }
        
        # Gold - Dynamic Analysis
        with cor_col1:
            gold = corr_data.get('gold', 0)
            gold_chg = corr_data.get('gold_change', 0)
            if gold and gold > 0:
                st.metric("🥇 Altın", f"${gold:,.0f}", f"{gold_chg:+.1f}%" if gold_chg else None)
                if gold_chg > 1:
                    st.caption('_"📈 Altın yükseliyor → BTC için pozitif"_')
                elif gold_chg < -1:
                    st.caption('_"📉 Altın düşüyor → Risk iştahı azalıyor"_')
                else:
                    st.caption('_"↔️ Altın stabil"_')
            else:
                st.metric("🥇 Altın", "N/A")
                st.caption('_"Piyasa kapalı veya veri yok"_')
        
        # Nasdaq - Dynamic Analysis
        with cor_col2:
            nasdaq = corr_data.get('nasdaq', 0)
            nasdaq_chg = corr_data.get('nasdaq_change', 0)
            if nasdaq and nasdaq > 0:
                st.metric("📈 Nasdaq", f"{nasdaq:,.0f}", f"{nasdaq_chg:+.1f}%" if nasdaq_chg else None)
                if nasdaq_chg > 1:
                    st.caption('_"📈 Risk iştahı yüksek → Kripto için pozitif"_')
                elif nasdaq_chg < -1:
                    st.caption('_"📉 Hisseler düşüyor → Kriptoda dikkat"_')
                else:
                    st.caption('_"↔️ Piyasa kararsız"_')
            else:
                st.metric("📈 Nasdaq", "N/A")
                st.caption('_"Hafta sonu veya piyasa kapalı"_')
        
        # BTC Dominance - Dynamic Analysis
        with cor_col3:
            btc_d = corr_data.get('btc_dominance', 0)
            btc_d_chg = corr_data.get('btc_dominance_change', 0)
            if btc_d and btc_d > 0:
                st.metric("₿ BTC Dominans", f"{btc_d:.1f}%", f"{btc_d_chg:+.1f}%" if btc_d_chg else None)
                if btc_d_chg > 0.5:
                    st.caption('_"📈 BTC güçleniyor → Altcoinler zayıf"_')
                elif btc_d_chg < -0.5:
                    st.caption('_"📉 Altcoin sezonu başlıyor!"_')
                else:
                    st.caption('_"↔️ Denge durumu"_')
            else:
                st.metric("₿ BTC Dominans", "N/A")
                st.caption('_"CoinGecko verisi yok"_')
        
        # Open Interest - Dynamic Analysis
        with cor_col4:
            oi = deriv_data.get('open_interest', 0)
            if oi and oi > 0:
                oi_display = f"${oi/1e9:.2f}B" if oi > 1e9 else f"${oi/1e6:.0f}M"
                st.metric("📊 Açık Pozisyon", oi_display)
                if oi > 30e9:  # 30B threshold
                    st.caption('_"🔴 Çok yüksek! Volatilite riski"_')
                elif oi > 20e9:
                    st.caption('_"🟡 Yüksek kaldıraç - dikkatli ol"_')
                else:
                    st.caption('_"🟢 Normal seviye"_')
            else:
                st.metric("📊 Açık Pozisyon", "N/A")
                st.caption('_"Binance Futures verisi bekleniyor"_')
        
        # Long/Short Ratio - Dynamic Analysis
        with cor_col5:
            ls_ratio = deriv_data.get('long_short_ratio', 0)
            if ls_ratio and ls_ratio > 0:
                if ls_ratio > 1.5:
                    st.metric("📊 L/S Oranı", f"🔴 {ls_ratio:.2f}")
                    st.caption('_"⚠️ Herkes long! Düşüş riski yüksek"_')
                elif ls_ratio < 0.7:
                    st.metric("📊 L/S Oranı", f"🟢 {ls_ratio:.2f}")
                    st.caption('_"✅ Herkes short! Sıkışma fırsatı"_')
                elif ls_ratio > 1.2:
                    st.metric("📊 L/S Oranı", f"🟡 {ls_ratio:.2f}")
                    st.caption('_"Long ağırlıklı - dikkatli ol"_')
                else:
                    st.metric("📊 L/S Oranı", f"⚪ {ls_ratio:.2f}")
                    st.caption('_"↔️ Dengeli piyasa"_')
            else:
                st.metric("📊 L/S Oranı", "N/A")
                st.caption('_"Binance Futures verisi bekleniyor"_')

        # ======================================
        # PHASE 122: FUNDING RATE & FEAR GREED
        # New panels with dynamic Turkish comments
        # ======================================
        st.markdown("---")
        st.markdown("### 💰 Funding Rate & Piyasa Duygusu")
        st.caption("_Contrarian trading sinyalleri - aşırı kalabalığın tersi kazandırır_")
        
        fg_col1, fg_col2 = st.columns(2)
        
        # Funding Rate Panel
        with fg_col1:
            try:
                funding = deriv_data.get('funding_rate', 0)
                funding_pct = funding * 100 if funding else 0
                
                if funding_pct > 0.05:
                    st.metric("💰 Funding Rate", f"🔴 {funding_pct:.4f}%", "EXTREME LONG")
                    st.caption('_"⚠️ Herkes LONG! Counter-trade: SHORT düşün"_')
                elif funding_pct > 0.01:
                    st.metric("💰 Funding Rate", f"🟡 {funding_pct:.4f}%", "Bullish")
                    st.caption('_"📈 Long ağırlıklı piyasa"_')
                elif funding_pct < -0.05:
                    st.metric("💰 Funding Rate", f"🟢 {funding_pct:.4f}%", "EXTREME SHORT")
                    st.caption('_"✅ Herkes SHORT! Counter-trade: LONG düşün"_')
                elif funding_pct < -0.01:
                    st.metric("💰 Funding Rate", f"🟡 {funding_pct:.4f}%", "Bearish")
                    st.caption('_"📉 Short ağırlıklı piyasa"_')
                else:
                    st.metric("💰 Funding Rate", f"⚪ {funding_pct:.4f}%", "Neutral")
                    st.caption('_"↔️ Dengeli piyasa - net sinyal yok"_')
            except:
                st.metric("💰 Funding Rate", "N/A")
                st.caption('_"Binance Futures verisi bekleniyor"_')
        
        # Fear & Greed Panel
        with fg_col2:
            try:
                from src.brain.web_scrapers import get_fear_greed_index
                fng = get_fear_greed_index()
                fng_value = fng.get('value', 50)
                fng_class = fng.get('classification', 'Neutral')
                
                if fng_value <= 25:
                    st.metric("😱 Fear & Greed", f"🟢 {fng_value}", "EXTREME FEAR")
                    st.caption('_"✅ Aşırı korku = ALIM FIRSATI! Kalabalık satıyor, sen al."_')
                elif fng_value <= 40:
                    st.metric("😱 Fear & Greed", f"🟡 {fng_value}", "Fear")
                    st.caption('_"📉 Korku var - dikkatli alım yapılabilir"_')
                elif fng_value >= 75:
                    st.metric("😱 Fear & Greed", f"🔴 {fng_value}", "EXTREME GREED")
                    st.caption('_"⚠️ Aşırı açgözlülük = RİSK! Herkes alıyor, sen dikkatli ol."_')
                elif fng_value >= 60:
                    st.metric("😱 Fear & Greed", f"🟡 {fng_value}", "Greed")
                    st.caption('_"📈 Açgözlülük var - kar realizasyonu düşün"_')
                else:
                    st.metric("😱 Fear & Greed", f"⚪ {fng_value}", "Neutral")
                    st.caption('_"↔️ Nötr piyasa - yön bekleniyor"_')
                
                st.progress(fng_value / 100)
            except Exception as e:
                st.metric("😱 Fear & Greed", "N/A")
                st.caption(f'_"Alternative.me verisi bekleniyor"_')

        # ======================================
        # PHASE 122: FIBONACCI LEVELS
        # Technical analysis support/resistance
        # ======================================
        st.markdown("---")
        st.markdown("### 📐 Fibonacci Seviyeleri")
        st.caption("_Önemli destek ve direnç noktaları - Fib 0.382/0.5/0.618_")
        
        try:
            from src.brain.fibonacci_analyzer import get_fibonacci
            import asyncio
            
            fib = get_fibonacci()
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            fib_data = loop.run_until_complete(fib.analyze(symbol='BTCUSDT', timeframe='4h'))
            loop.close()
            
            if fib_data and 'error' not in fib_data:
                fib_col1, fib_col2, fib_col3 = st.columns(3)
                
                with fib_col1:
                    trend = fib_data.get('trend', 'UNKNOWN')
                    trend_emoji = "📈" if trend == 'UPTREND' else "📉" if trend == 'DOWNTREND' else "↔️"
                    st.metric("📐 Trend", f"{trend_emoji} {trend}")
                    if trend == 'UPTREND':
                        st.caption('_"Yükseliş trendi - fib desteklerinden al"_')
                    elif trend == 'DOWNTREND':
                        st.caption('_"Düşüş trendi - fib dirençlerinden sat"_')
                
                with fib_col2:
                    support = fib_data.get('nearest_support')
                    if support:
                        st.metric("🟢 En Yakın Destek", f"${support['price']:,.0f}", f"Fib {support['ratio']}")
                        st.caption(f'_"Bu seviye altına düşerse {support["distance_pct"]:.1f}% kayıp"_')
                    else:
                        st.metric("🟢 En Yakın Destek", "N/A")
                
                with fib_col3:
                    resistance = fib_data.get('nearest_resistance')
                    if resistance:
                        st.metric("🔴 En Yakın Direnç", f"${resistance['price']:,.0f}", f"Fib {resistance['ratio']}")
                        st.caption(f'_"Bu seviyeyi geçerse {abs(resistance["distance_pct"]):.1f}% potansiyel"_')
                    else:
                        st.metric("🔴 En Yakın Direnç", "N/A")
                
                # Fib signal
                signal = fib_data.get('signal', {})
                if signal.get('direction') != 'NEUTRAL':
                    dir_emoji = "📈" if signal['direction'] == 'LONG' else "📉"
                    st.info(f"{dir_emoji} **Fib Sinyal:** {signal['direction']} - {signal.get('reason', '')} ({signal.get('confidence', 0)}% güven)")
            else:
                st.warning("📐 Fibonacci analizi yapılamadı")
                
        except Exception as e:
            st.warning(f"Fibonacci analizi kullanılamıyor: {e}")

        # ======================================
        # PHASE 43: DOMINANCE METRICS
        # Comprehensive market dominance tracking
        # ======================================
        st.markdown("---")
        st.markdown("### 📊 Market Dominance & Macro Indicators")
        st.caption("_Real-time data from TradingView (replaces yfinance)_")
        
        # Row 1: Crypto Dominance
        dom_col1, dom_col2, dom_col3 = st.columns(3)
        
        with dom_col1:
            btc_d = deriv_data.get('btc_dominance', 0)
            btc_d_change = deriv_data.get('btc_dominance_change', 0)
            if btc_d > 0:
                change_emoji = "🟢" if btc_d_change > 0 else "🔴" if btc_d_change < 0 else "⚪"
                st.metric("🪙 BTC Dominance", f"{btc_d:.1f}%", f"{btc_d_change:+.2f}%")
                if btc_d > 60:
                    st.caption(f'_{change_emoji} "BTC çok dominant - altcoin sezonu uzak"_')
                elif btc_d < 40:
                    st.caption(f'_{change_emoji} "BTC zayıf - altcoin sezonu!"_')
                else:
                    st.caption(f'_{change_emoji} "BTC dengeli dominance"_')
            else:
                st.metric("🪙 BTC Dominance", "N/A")
        
        with dom_col2:
            eth_d = deriv_data.get('eth_dominance', 0)
            eth_d_change = deriv_data.get('eth_dominance_change', 0)
            if eth_d > 0:
                change_emoji = "🟢" if eth_d_change > 0 else "🔴" if eth_d_change < 0 else "⚪"
                st.metric("💎 ETH Dominance", f"{eth_d:.1f}%", f"{eth_d_change:+.2f}%")
                st.caption(f'_{change_emoji} "ETH market share"_')
            else:
                st.metric("💎 ETH Dominance", "N/A")
        
        with dom_col3:
            total_stable = deriv_data.get('total_stablecoin_dominance', 0)
            stable_signal = deriv_data.get('stablecoin_signal', 'NEUTRAL')
            if total_stable > 0:
                if stable_signal == 'EXTREME_FEAR':
                    st.metric("💵 Stablecoin Dominance", f"🔴 {total_stable:.1f}%", "EXTREME FEAR")
                elif stable_signal == 'CAUTION':
                    st.metric("💵 Stablecoin Dominance", f"🟡 {total_stable:.1f}%", "CAUTION")
                elif stable_signal == 'GREED':
                    st.metric("💵 Stablecoin Dominance", f"🟢 {total_stable:.1f}%", "GREED")
                else:
                    st.metric("💵 Stablecoin Dominance", f"⚪ {total_stable:.1f}%", "NEUTRAL")
                
                st.caption(f'_"{deriv_data.get("stablecoin_interpretation", "")}"_')
            else:
                st.metric("💵 Stablecoin Dominance", "N/A")
        
        # Row 2: Stablecoin Breakdown
        stable_col1, stable_col2, stable_col3 = st.columns(3)
        
        with stable_col1:
            usdt_d = deriv_data.get('usdt_dominance', 0)
            if usdt_d > 0:
                st.metric("💰 USDT Dominance", f"{usdt_d:.2f}%")
                st.caption('_"Tether market share"_')
            else:
                st.metric("💰 USDT Dominance", "N/A")
        
        with stable_col2:
            usdc_d = deriv_data.get('usdc_dominance', 0)
            if usdc_d > 0:
                st.metric("🏦 USDC Dominance", f"{usdc_d:.2f}%")
                st.caption('_"Circle (institutional) market share"_')
            else:
                st.metric("🏦 USDC Dominance", "N/A")
        
        with stable_col3:
            # Interpretation summary
            if total_stable > 8:
                st.warning("⚠️ **HIGH FEAR:** Para stablecoin'lere kaçıyor!")
            elif total_stable > 6:
                st.info("ℹ️ **CAUTION:** Stablecoin akışı artıyor")
            elif total_stable < 4:
                st.success("✅ **GREED:** Para crypto'ya geri dönüyor")
            else:
                st.caption("↔️ Normal stablecoin aktivitesi")
        
        # Row 3: Traditional Market Indicators
        st.markdown("#### Traditional Markets (Correlation Indicators)")
        macro_col1, macro_col2, macro_col3, macro_col4 = st.columns(4)
        
        with macro_col1:
            dxy = deriv_data.get('dxy', 0)
            dxy_change = deriv_data.get('dxy_change', 0)
            if dxy > 0:
                change_emoji = "🟢" if dxy_change > 0 else "🔴" if dxy_change < 0 else "⚪"
                st.metric("💵 DXY (Dollar Index)", f"{dxy:.2f}", f"{dxy_change:+.2f}%")
                st.caption(f'_{change_emoji} "DXY ↑ = Crypto usually ↓"_')
            else:
                st.metric("💵 DXY", "N/A")
        
        with macro_col2:
            vix = deriv_data.get('vix', 0)
            vix_change = deriv_data.get('vix_change', 0)
            if vix > 0:
                if vix > 30:
                    st.metric("😱 VIX (Fear Index)", f"🔴 {vix:.2f}", f"{vix_change:+.2f}%")
                    st.caption('_"Extreme fear in markets"_')
                elif vix > 20:
                    st.metric("😱 VIX (Fear Index)", f"🟡 {vix:.2f}", f"{vix_change:+.2f}%")
                    st.caption('_"Elevated fear"_')
                else:
                    st.metric("😱 VIX (Fear Index)", f"🟢 {vix:.2f}", f"{vix_change:+.2f}%")
                    st.caption('_"Low fear = risk-on"_')
            else:
                st.metric("😱 VIX", "N/A")
        
        with macro_col3:
            spy = deriv_data.get('spy', 0)
            spy_change = deriv_data.get('spy_change', 0)
            if spy > 0:
                change_emoji = "🟢" if spy_change > 0 else "🔴" if spy_change < 0 else "⚪"
                st.metric("📈 SPY (S&P 500)", f"${spy:.2f}", f"{spy_change:+.2f}%")
                st.caption(f'_{change_emoji} "SPY ↑ = Crypto usually ↑"_')
            else:
                st.metric("📈 SPY", "N/A")
        
        with macro_col4:
            # Summary indicator
            if dxy > 0 and spy > 0:
                if dxy_change > 0.5 and spy_change < -0.5:
                    st.warning("⚠️ **Risk-Off Mode**")
                    st.caption("_Dollar strong, stocks weak_")
                elif dxy_change < -0.5 and spy_change > 0.5:
                    st.success("✅ **Risk-On Mode**")
                    st.caption("_Dollar weak, stocks strong_")
                else:
                    st.info("↔️ **Mixed Signals**")
                    st.caption("_No clear macro trend_")

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

        # AI Reasoning Summary - Detaylı Türkçe Analiz
        st.markdown("### 🤖 AI Karar Özeti")
        st.caption("_Her coin için detaylı analiz, Entry/SL/TP ve karar gerekçesi_")
        
        for symbol, info in data.items():
            dec = info.get('ai_decision', 'NEUTRAL')
            conf = info.get('ai_confidence', 0)
            price = info.get('price', 0)
            
            # Signal styling
            if dec == "BUY":
                signal_emoji = "🟢"
                signal_tr = "ALIŞ"
                bg_color = "#1a4731"
            elif dec == "SELL":
                signal_emoji = "🔴"
                signal_tr = "SATIŞ"
                bg_color = "#4a1a1a"
            else:
                signal_emoji = "⚪"
                signal_tr = "BEKLE"
                bg_color = "#2d2d2d"
            
            with st.expander(f"{signal_emoji} **{symbol}**: {signal_tr} ({conf:.0f}%) - ${price:,.2f}", expanded=False):
                
                # === KARAR GEREKÇESİ ===
                st.markdown("#### 📋 Karar Gerekçesi")
                
                # Collect all reasoning factors
                reasons = []
                
                # 1. SMC Analysis
                smc = info.get('smc', {})
                if smc:
                    smc_bias = smc.get('bias', 'N/A')
                    smc_strength = smc.get('strength', 0)
                    if smc_bias == 'BULLISH':
                        reasons.append(f"✅ **SMC**: Yükseliş eğilimi tespit edildi (Güç: %{smc_strength:.0f})")
                    elif smc_bias == 'BEARISH':
                        reasons.append(f"❌ **SMC**: Düşüş eğilimi tespit edildi (Güç: %{smc_strength:.0f})")
                    else:
                        reasons.append(f"↔️ **SMC**: Kararsız piyasa")
                
                # 2. MTF Confluence
                mtf = info.get('mtf', {})
                if mtf:
                    confluence = mtf.get('confluence_score', 0)
                    trend_1h = mtf.get('trend_1h', 'N/A')
                    trend_4h = mtf.get('trend_4h', 'N/A')
                    if confluence > 70:
                        reasons.append(f"✅ **MTF**: Tüm zaman dilimleri uyumlu (%{confluence:.0f}) - 1h:{trend_1h}, 4h:{trend_4h}")
                    elif confluence > 50:
                        reasons.append(f"🟡 **MTF**: Kısmi uyum (%{confluence:.0f})")
                    else:
                        reasons.append(f"❌ **MTF**: Zaman dilimleri çelişkili (%{confluence:.0f})")
                
                # 3. Wyckoff Phase
                wyckoff = info.get('wyckoff_phase', '')
                if wyckoff:
                    wyckoff_tr = {
                        'ACCUMULATION': '📈 BİRİKİM - Akıllı para alım yapıyor',
                        'MARKUP': '🚀 YÜKSELİŞ - Ralli başladı',
                        'DISTRIBUTION': '📉 DAĞITIM - Akıllı para satıyor',
                        'MARKDOWN': '💀 DÜŞÜŞ - Satış rallisi'
                    }.get(wyckoff, wyckoff)
                    reasons.append(f"📊 **Wyckoff**: {wyckoff_tr}")
                
                # 4. On-Chain
                onchain = info.get('onchain_signal', '')
                if onchain and 'BUY' in onchain:
                    reasons.append("🐋 **On-Chain**: Balinalar alım yapıyor")
                elif onchain and 'SELL' in onchain:
                    reasons.append("🐋 **On-Chain**: Balinalar satış yapıyor")
                
                # 5. Regime
                regime = info.get('regime', 'UNKNOWN')
                regime_tr = {
                    'TRENDING_BULL': '📈 Güçlü Yükseliş Trendi',
                    'TRENDING_BEAR': '📉 Güçlü Düşüş Trendi',
                    'RANGING': '↔️ Yatay Piyasa',
                    'VOLATILE': '⚡ Yüksek Volatilite'
                }.get(regime, regime)
                reasons.append(f"🌍 **Piyasa Durumu**: {regime_tr}")
                
                for reason in reasons:
                    st.markdown(f"• {reason}")
                
                # === ENTRY / SL / TP ===
                st.markdown("---")
                st.markdown("#### 🎯 İşlem Seviyeleri")
                
                sltp = info.get('smart_sltp', {})
                if sltp and dec != 'NEUTRAL':
                    direction = sltp.get('direction', 'NONE')
                    sl = sltp.get('stop_loss', 0)
                    tp1 = sltp.get('take_profit_1', 0)
                    tp2 = sltp.get('take_profit_2', 0)
                    tp3 = sltp.get('take_profit_3', 0)
                    quality = sltp.get('quality', 'N/A')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📍 Giriş (Entry)", f"${price:,.2f}")
                    with col2:
                        sl_pct = ((price - sl) / price * 100) if sl and price else 0
                        st.metric("🛑 Zarar Kes (SL)", f"${sl:,.2f}" if sl else "N/A", f"{sl_pct:+.1f}%" if sl else None, delta_color="inverse")
                    with col3:
                        tp1_pct = ((tp1 - price) / price * 100) if tp1 and price else 0
                        st.metric("🎯 Hedef 1 (TP1)", f"${tp1:,.2f}" if tp1 else "N/A", f"{tp1_pct:+.1f}%" if tp1 else None)
                    
                    if tp2 or tp3:
                        col4, col5, col6 = st.columns(3)
                        with col4:
                            st.metric("🎯 Hedef 2 (TP2)", f"${tp2:,.2f}" if tp2 else "N/A")
                        with col5:
                            st.metric("🎯 Hedef 3 (TP3)", f"${tp3:,.2f}" if tp3 else "N/A")
                        with col6:
                            quality_emoji = "🟢" if quality == "HIGH" else "🟡" if quality == "MEDIUM" else "🔴"
                            st.metric("📊 Sinyal Kalitesi", f"{quality_emoji} {quality}")
                    
                    # Risk/Reward calculation
                    if sl and tp1 and price:
                        risk = abs(price - sl)
                        reward = abs(tp1 - price)
                        rr_ratio = reward / risk if risk > 0 else 0
                        st.caption(f"_Risk/Ödül Oranı: 1:{rr_ratio:.1f}_")
                else:
                    st.info("⏸️ BEKLE sinyali - şu an işlem önerilmiyor. Fırsat oluştuğunda Entry/SL/TP hesaplanacak.")


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

# ==========================================
# 🌐 WEB INTELLIGENCE - All Web Scraping Data
# ==========================================
elif page == "🌐 Web Intelligence":
    st.header("🌐 Web Intelligence Dashboard")
    st.caption("Tüm web scraping verileri - Fear&Greed, TradingView, CME Gap, DeFi TVL, Stablecoins, News")
    
    # Import scrapers
    try:
        from src.brain.advanced_scrapers import AdvancedMarketScrapers
        from src.brain.news_scraper import CryptoNewsScraper
        from src.brain.signal_combiner import SignalCombinerModel
        # PHASE 42: Critical Scrapers
        from src.brain.liquidation_tracker import LiquidationTracker
        from src.brain.whale_tracker import WhaleTracker
        from src.brain.reddit_scraper import RedditScraper
        
        scrapers = AdvancedMarketScrapers()
        news_scraper = CryptoNewsScraper()
        signal_combiner = SignalCombinerModel()
        # Initialize Phase 42 scrapers
        liquidation_tracker = LiquidationTracker()
        whale_tracker = WhaleTracker()
        reddit_scraper = RedditScraper()
        
        scrapers_available = True
    except Exception as e:
        st.warning(f"⚠️ Scrapers yüklenemedi: {e}")
        scrapers_available = False
    
    if scrapers_available:
        # Row 1: Fear & Greed + TradingView Signals
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("😱 Fear & Greed Index")
            try:
                fng = scrapers.get_fear_greed_index()
                fng_value = fng.get('value', 50)
                fng_class = fng.get('classification', 'Neutral')
                
                # Color based on value
                if fng_value <= 25:
                    color = "#00ff00"  # Green - Extreme Fear = Buy
                    emoji = "🟢"
                elif fng_value >= 75:
                    color = "#ff0000"  # Red - Extreme Greed = Caution
                    emoji = "🔴"
                else:
                    color = "#ffff00"  # Yellow
                    emoji = "🟡"
                
                st.metric("Fear & Greed", f"{emoji} {fng_value}", fng_class)
                st.caption(fng.get('action', ''))
                
                # Progress bar
                st.progress(fng_value / 100)
            except Exception as e:
                st.error(f"Veri alınamadı: {e}")
        
        with col2:
            st.subheader("📊 TradingView Sinyalleri")
            st.caption("_Kaynak: TradingView Technicals (MA + Oscillators)_")
            coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT']
            
            tv_cols = st.columns(4)
            for i, coin in enumerate(coins):
                with tv_cols[i]:
                    try:
                        tv = scrapers.get_tradingview_signals(coin)
                        action = tv.get('action', 'NEUTRAL')
                        emoji = tv.get('emoji', '⚪')
                        rsi = tv.get('rsi', 50)
                        
                        st.metric(coin.replace('USDT', ''), f"{emoji} {action}", f"RSI: {rsi:.0f}")
                    except:
                        st.metric(coin.replace('USDT', ''), "⚪ N/A", "")
            
            # Explanation
            st.markdown("""
            <small>
            📌 <b>Sinyal Kaynağı:</b> TradingView'den 10+ teknik indikatör (EMA, SMA, MACD, RSI, Stochastic, CCI, ADX) birleştirilir.<br>
            • <b>BUY/STRONG_BUY:</b> Çoğu indikatör yükseliş sinyali veriyor<br>
            • <b>SELL/STRONG_SELL:</b> Çoğu indikatör düşüş sinyali veriyor<br>
            • <b>NEUTRAL:</b> İndikatörler karışık
            </small>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Row 2: CME Gap + DeFi TVL + Stablecoin Flow
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 CME Gap")
            try:
                cme = scrapers.get_cme_gap()
                if cme.get('has_gap', False):
                    gap_pct = cme.get('gap_percent', 0)
                    gap_price = cme.get('gap_price', 0)
                    emoji = "🔴" if gap_pct > 0 else "🟢"
                    st.metric("CME Gap", f"{emoji} {gap_pct:+.1f}%", f"Gap: ${gap_price:,.0f}")
                    st.caption(cme.get('action', ''))
                else:
                    st.metric("CME Gap", "✅ Yok", "Gap kapatılmış")
            except Exception as e:
                st.error(f"Veri alınamadı: {e}")
        
        with col2:
            st.subheader("💧 DeFi TVL")
            try:
                tvl = scrapers.get_defi_tvl()
                tvl_formatted = tvl.get('total_tvl_formatted', 'N/A')
                change = tvl.get('change_24h', 0)
                emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                st.metric("Total TVL", tvl_formatted, f"{emoji} {change:+.1f}%")
                # Better explanation
                if change > 3:
                    st.caption("🟢 TVL artıyor = Para DeFi'ye giriyor = Risk-on sentiment = BULLISH")
                elif change < -3:
                    st.caption("🔴 TVL düşüyor = Para DeFi'den çıkıyor = Risk-off sentiment = BEARISH")
                else:
                    st.caption("⚪ TVL stabil = Piyasa dengeli, net sinyal yok")
            except Exception as e:
                st.error(f"Veri alınamadı: {e}")
        
        with col3:
            st.subheader("💵 Stablecoin Flow")
            try:
                stable = scrapers.get_stablecoin_flow()
                supply = stable.get('usdt_supply_formatted', 'N/A')
                change = stable.get('change_7d_formatted', 'N/A')
                change_value = stable.get('change_7d', 0)
                direction = stable.get('direction', 'NEUTRAL')
                emoji = "🟢" if direction == 'BULLISH' else "🔴" if direction == 'BEARISH' else "⚪"
                st.metric("USDT Supply", supply, f"{emoji} {change} (7d)")
                # Better explanation
                if change_value > 500_000_000:
                    st.caption(f"🟢 ${change_value/1e6:.0f}M yeni USDT mint edildi = Alım hazırlığı = BULLISH")
                elif change_value < -500_000_000:
                    st.caption(f"🔴 ${abs(change_value)/1e6:.0f}M USDT yakıldı = Çıkış yapılıyor = BEARISH")
                else:
                    st.caption("⚪ Stablecoin akışı normal seviyede")
            except Exception as e:
                st.error(f"Veri alınamadı: {e}")
        
        st.divider()
        
        # Row 3: News Sentiment
        st.subheader("📰 Haber Sentimenti")
        try:
            news_sentiment = news_scraper.get_market_sentiment()
            score = news_sentiment.get('score', 50)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                mood = "🟢 BULLISH" if score > 60 else "🔴 BEARISH" if score < 40 else "⚪ NEUTRAL"
                st.metric("Genel Mood", mood, f"Skor: {score:.0f}/100")
            with col2:
                st.metric("Pozitif Haberler", f"🟢 {news_sentiment.get('bullish_count', 0)}", "")
            with col3:
                st.metric("Negatif Haberler", f"🔴 {news_sentiment.get('bearish_count', 0)}", "")
            with col4:
                st.metric("Toplam Haber", news_sentiment.get('news_count', 0), "")
            
            # Show important news
            important = news_scraper.get_important_news(5)
            if important:
                st.markdown("**Son Önemli Haberler:**")
                for news in important:
                    emoji = "🟢" if news.sentiment == 'BULLISH' else "🔴" if news.sentiment == 'BEARISH' else "⚪"
                    impact = "⚡" if news.impact == 'HIGH' else ""
                    st.markdown(f"- {emoji}{impact} **{news.title[:70]}...** _{news.source}_")
        except Exception as e:
            st.error(f"Haber verisi alınamadı: {e}")
        
        st.divider()
        
        # Row 4: Signal Combiner (Unified AI Signal)
        st.subheader("🤖 AI Birleşik Sinyal")
        st.caption("Tüm sinyaller ML modeli ile birleştirilir")
        
        try:
            # Collect all signals for prediction
            fng_data = scrapers.get_fear_greed_index()
            tv_btc = scrapers.get_tradingview_signals('BTCUSDT')
            stable_data = scrapers.get_stablecoin_flow()
            tvl_data = scrapers.get_defi_tvl()
            cme_data = scrapers.get_cme_gap()
            news_data = news_scraper.get_market_sentiment()
            
            # Prepare input for Signal Combiner
            raw_data = {
                'fear_greed': fng_data.get('value', 50),
                'tradingview': tv_btc.get('overall', 0),
                'stablecoin_flow': stable_data.get('change_7d', 0),
                'defi_tvl_change': tvl_data.get('change_24h', 0),
                'cme_gap': cme_data.get('gap_percent', 0),
                'news_sentiment': news_data.get('score', 50),
                'bullish_patterns': 0,  # Would come from chart_patterns
                'bearish_patterns': 0,
                'funding_rate': 0,
                'oi_velocity': 0,
                'whale_ratio': 0.5,
                'rsi': tv_btc.get('rsi', 50) if isinstance(tv_btc.get('rsi'), (int, float)) else 50
            }
            
            signal = signal_combiner.predict(raw_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                action_emoji = "🟢🟢" if signal.action == 'STRONG_BUY' else "🟢" if signal.action == 'BUY' else "🔴🔴" if signal.action == 'STRONG_SELL' else "🔴" if signal.action == 'SELL' else "⚪"
                st.metric("AI Sinyal", f"{action_emoji} {signal.action}", f"Score: {signal.raw_score:+.2f}")
            with col2:
                conf_bar = "█" * int(signal.confidence / 20) + "░" * (5 - int(signal.confidence / 20))
                st.metric("Güven", f"[{conf_bar}]", f"{signal.confidence:.0f}%")
            with col3:
                st.metric("Model", "Signal Combiner v1", "ML Trained")
            
            # Show reasoning
            st.info(f"💡 **AI Reasoning:** {signal.reasoning}")
            
            # Show top factors
            if signal.top_bullish:
                st.success(f"🟢 Bullish faktörler: {', '.join(signal.top_bullish)}")
            if signal.top_bearish:
                st.error(f"🔴 Bearish faktörler: {', '.join(signal.top_bearish)}")
                
        except Exception as e:
            st.error(f"Signal Combiner hatası: {e}")
        
        st.divider()
        
        # Row 5: Confluence Analyzer
        st.subheader("🎯 Confluence Analyzer")
        st.caption("MTF, Volatilite, Seans ve Exchange Flow analizi")
        
        try:
            from src.brain.confluence_analyzer import ConfluenceAnalyzer
            confluence = ConfluenceAnalyzer()
            
            # Analyze all coins
            coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT']
            
            for coin in coins:
                conf_signal = confluence.calculate_confluence(coin)
                
                # Score bar visualization
                score = conf_signal.confluence_score
                score_bar = "█" * score + "░" * (10 - score)
                
                # Direction emoji
                dir_emoji = "🟢" if conf_signal.direction == 'BULLISH' else "🔴" if conf_signal.direction == 'BEARISH' else "⚪"
                
                with st.expander(f"{dir_emoji} **{coin.replace('USDT', '')}** - Confluence: [{score_bar}] {score}/10"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("**📊 MTF:**")
                        for tf, sig in conf_signal.mtf_agreement.items():
                            tf_emoji = "🟢" if sig['direction'] == 'BULLISH' else "🔴" if sig['direction'] == 'BEARISH' else "⚪"
                            st.markdown(f"• {tf}: {tf_emoji}")
                    
                    with col2:
                        vol_emoji = "💥" if conf_signal.volatility_state == 'COMPRESSED' else "📊"
                        st.markdown(f"**{vol_emoji} Volatilite:**")
                        st.markdown(conf_signal.volatility_state)
                    
                    with col3:
                        st.markdown("**🕐 Session:**")
                        st.markdown(conf_signal.session)
                    
                    with col4:
                        flow_emoji = "🟢" if conf_signal.exchange_flow == 'OUTFLOW' else "🔴" if conf_signal.exchange_flow == 'INFLOW' else "⚪"
                        st.markdown(f"**{flow_emoji} Exchange:**")
                        st.markdown(conf_signal.exchange_flow)
                    
                    # Action
                    if score >= 6:
                        st.success(conf_signal.action)
                    elif score >= 4:
                        st.info(conf_signal.action)
                    else:
                        st.warning(conf_signal.action)
                        
        except Exception as e:
            st.error(f"Confluence Analyzer hatası: {e}")
        
        st.divider()
        
        # ===================================
        # PHASE 42: CRITICAL SCRAPERS
        # ===================================
        
        # Row 6: Liquidation Heatmap
        st.subheader("⚡ Liquidation Heatmap")
        st.caption("Binance Open Interest analizi - Cascade risk detection")
        
        try:
            liq_summary = liquidation_tracker.get_liquidation_summary('BTCUSDT')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk = liq_summary['cascade_risk']
                risk_emoji = "🔴" if risk == 'HIGH' else "🟡" if risk == 'MEDIUM' else "🟢"
                st.metric("Cascade Risk", f"{risk_emoji} {risk}", f"{liq_summary['zone_count']} zones")
            
            with col2:
                nearby_size = liq_summary['nearby_liq_size']
                st.metric("Nearby Liquidations", f"${nearby_size/1e9:.2f}B", "Within 3% of price")
            
            with col3:
                nl = liq_summary.get('nearest_long_liq')
                if nl:
                    st.metric("Nearest Long Liq", f"${nl.price_level:,.0f}", f"{nl.distance_pct:.1f}% below")
            
            # Explanation
            st.caption(liq_summary['summary'])
            
        except Exception as e:
            st.error(f"Liquidation Heatmap hatası: {e}")
        
        st.divider()
        
        # Row 7: Whale Alert
        st.subheader("🐋 Whale Alert")
        st.caption("Bitcoin $1M+ transactions - Exchange flow analysis")
        
        try:
            whale_summary = whale_tracker.get_whale_summary('BTC', hours=24)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Whales Tracked", whale_summary['whale_count'], "Last 24h")
            
            with col2:
                inflow = whale_summary['total_inflow']
                st.metric("Exchange Inflow", f"${inflow/1e6:.1f}M", "🔴 Selling pressure")
            
            with col3:
                outflow = whale_summary['total_outflow']
                st.metric("Exchange Outflow", f"${outflow/1e6:.1f}M", "🟢 Accumulation")
            
            with col4:
                net_flow = whale_summary['net_flow']
                direction = whale_summary['direction']
                dir_emoji = "🟢" if direction == 'BULLISH' else "🔴" if direction == 'BEARISH' else "⚪"
                st.metric("Net Flow", f"${net_flow/1e6:+.0f}M", f"{dir_emoji} {direction}")
            
            # Recent whales
            if whale_summary.get('recent_whales'):
                st.markdown("**Son Büyük Transferler:**")
                for whale in whale_summary['recent_whales'][:5]:
                    tx_emoji = "🔴" if whale.tx_type == 'EXCHANGE_INFLOW' else "🟢"
                    st.markdown(f"• {tx_emoji} ${whale.amount_usd/1e6:.1f}M {whale.tx_type} - {whale.timestamp.strftime('%H:%M')}")
            
            # Explanation
            st.info(whale_summary['summary'])
            
        except Exception as e:
            st.error(f"Whale Alert hatası: {e}")
        
        st.divider()
        
        # Row 8: Reddit Sentiment
        st.subheader("💬 Reddit Sentiment")
        st.caption("r/cryptocurrency + r/bitcoin + r/ethtrader sentiment analysis")
        
        try:
            reddit_sentiment = reddit_scraper.get_sentiment(hours=24)
            
            score = reddit_sentiment['score']
            mood = reddit_sentiment['sentiment']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Score bar
                bar_length = int(score / 10)
                score_bar = "█" * bar_length + "░" * (10 - bar_length)
                mood_emoji = "🟢" if mood == 'BULLISH' else "🔴" if mood == 'BEARISH' else "⚪"
                st.metric("Sentiment Score", f"[{score_bar}]", f"{mood_emoji} {score}/100")
            
            with col2:
                st.metric("Total Posts", reddit_sentiment['post_count'], "Last 24h")
            
            with col3:
                st.metric("Bullish Posts", f"🟢 {reddit_sentiment['bullish_count']}", "")
            
            with col4:
                st.metric("Bearish Posts", f"🔴 {reddit_sentiment['bearish_count']}", "")
            
            # Top posts
            if reddit_sentiment.get('top_posts'):
                st.markdown("**En Popüler Postlar:**")
                for i, post in enumerate(reddit_sentiment['top_posts'][:5], 1):
                    emoji = "🟢" if post.sentiment == 'BULLISH' else "🔴" if post.sentiment == 'BEARISH' else "⚪"
                    st.markdown(f"{i}. {emoji} [{post.subreddit}] {post.title[:70]}...")
                    st.caption(f"   ↑{post.score} ({post.upvote_ratio:.0%} upvoted)")
            
            # Summary
            st.success(reddit_sentiment['summary'])
            
        except Exception as e:
            st.error(f"Reddit Sentiment hatası: {e}")
        
        st.divider()
        
        # ===================================
        # PHASE 45: HIGH-VALUE SCRAPERS
        # ===================================
        
        # Row 9: DEX Volume Spikes
        st.subheader("🔥 DEX Volume Spikes")
        st.caption("DexScreener - Pump detection & new pairs")
        
        try:
            from src.brain.dex_volume_tracker import DEXVolumeTracker
            dex_tracker = DEXVolumeTracker()
            
            spike_data = dex_tracker.detect_volume_spikes(min_volume_24h=100000)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🚀 Mega Pumps", len(spike_data['mega_pumps']), ">1000% volume")
            
            with col2:
                st.metric("⚡ Volume Spikes", len(spike_data['pump_signals']), ">300% volume")
            
            with col3:
                st.metric("🆕 New Pairs", len(spike_data['new_pairs']), "Last 24h")
            
            with col4:
                st.metric("📊 Total Tracked", spike_data['total_tracked'], "DEX pairs")
            
            # Mega pumps list
            if spike_data['mega_pumps']:
                st.warning("⚠️ **MEGA PUMP ALERT:**")
                for pair in spike_data['mega_pumps'][:3]:
                    st.markdown(f"• **{pair.base_token}/{pair.quote_token}** ({pair.chain})")
                    st.caption(f"   📈 +{pair.volume_change_24h:.0f}% volume | ${pair.volume_24h/1e6:.1f}M | {pair.dex}")
            
            # Regular pumps
            elif spike_data['pump_signals']:
                st.info("ℹ️ **Volume Spikes Detected:**")
                for pair in spike_data['pump_signals'][:5]:
                    st.markdown(f"• {pair.base_token}: +{pair.volume_change_24h:.0f}%")
            
            # Summary
            st.caption(spike_data['summary'])
            
        except Exception as e:
            st.error(f"❌ DEX Volume Tracker hatası: {e}")
            import traceback
            with st.expander("Debug Info"):
                st.code(traceback.format_exc())
        
        st.divider()
        
        # Row 10: Twitter Sentiment (Influencers)
        st.subheader("🐦 Twitter Sentiment")
        st.caption("Top crypto influencer analysis via Nitter")
        
        try:
            from src.brain.twitter_sentiment import TwitterSentimentScraper
            twitter = TwitterSentimentScraper()
            
            sentiment_data = twitter.get_influencer_sentiment(hours=24)
            
            score = sentiment_data['score']
            mood = sentiment_data['sentiment']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Score visualization
                bar_length = int(score / 10)
                score_bar = "█" * bar_length + "░" * (10 - bar_length)
                mood_emoji = "🟢" if mood == 'BULLISH' else "🔴" if mood == 'BEARISH' else "⚪"
                st.metric("Sentiment", f"[{score_bar}]", f"{mood_emoji} {score:.0f}/100")
            
            with col2:
                st.metric("Tweets Analyzed", sentiment_data['tweet_count'], "Last 24h")
            
            with col3:
                st.metric("Bullish", f"🟢 {sentiment_data['bullish_count']}", "")
            
            with col4:
                st.metric("Bearish", f"🔴 {sentiment_data['bearish_count']}", "")
            
            # Top influencer tweet
            if sentiment_data['top_tweets']:
                top_tweet = sentiment_data['top_tweets'][0]
                st.markdown(f"**Top Tweet** (@{top_tweet.username}):")
                st.caption(f'"{top_tweet.text[:150]}..."')
                st.caption(f"❤️ {top_tweet.likes} | 🔁 {top_tweet.retweets}")
            
            # Summary
            if score >= 70:
                st.success(sentiment_data['summary'])
            elif score <= 30:
                st.warning(sentiment_data['summary'])
            else:
                st.info(sentiment_data['summary'])
            
        except Exception as e:
            st.error(f"❌ Twitter Sentiment hatası: {e}")
            import traceback
            with st.expander("Debug Info"):
                st.code(traceback.format_exc())
        
        st.divider()
        
        # Row 11: Enhanced Funding Rate
        st.subheader("💰 Enhanced Funding Rate")
        st.caption("Multi-exchange comparison (Binance, Bybit, OKX)")
        
        try:
            from src.brain.enhanced_funding import EnhancedFundingTracker
            funding = EnhancedFundingTracker()
            
            funding_data = funding.get_multi_exchange_funding('BTCUSDT')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Exchange rates
                st.markdown("**Exchange Rates:**")
                for exchange in ['binance', 'bybit', 'okx']:
                    fr = funding_data.get(exchange)
                    if fr and fr.rate is not None:
                        rate_pct = fr.rate * 100
                        emoji = "🔴" if fr.rate > 0.1 else "🟡" if fr.rate > 0.05 else "🟢"
                        st.markdown(f"{emoji} {exchange.capitalize()}: {rate_pct:.4f}%")
            
            with col2:
                avg_rate = funding_data['average'] * 100
                st.metric("Average Rate", f"{avg_rate:.4f}%", "")
                
                divergence = funding_data['divergence'] * 100
                if divergence > 0.05:
                    st.metric("Divergence", f"⚠️ {divergence:.4f}%", "Cross-exchange")
                else:
                    st.metric("Divergence", f"{divergence:.4f}%", "Normal")
            
            with col3:
                signal = funding_data['signal']
                if signal == 'EXTREME_LONG':
                    st.error("🔴 EXTREME LONG BIAS")
                    st.caption("Contrarian SHORT signal!")
                elif signal == 'EXTREME_SHORT':
                    st.success("🟢 EXTREME SHORT BIAS")
                    st.caption("Contrarian LONG signal!")
                elif signal == 'DIVERGENCE':
                    st.warning("⚠️ DIVERGENCE DETECTED")
                    st.caption("Possible manipulation")
                else:
                    st.info("↔️ Normal Funding")
            
            # Summary
            st.caption(funding_data['summary'])
            
        except Exception as e:
            st.error(f"❌ Enhanced Funding hatası: {e}")
            import traceback
            with st.expander("Debug Info"):
                st.code(traceback.format_exc())
        
        st.divider()
        
        # Refresh button
        if st.button("🔄 Verileri Yenile"):
            st.cache_data.clear()
            st.rerun()


# ==========================================
# NEW: AI PREDICTIONS MODULE
# Markov Chain, LSTM, Whale Intel, Liquidation Hunter
# ==========================================
elif page == "🔮 AI Predictions":
    st.header("🔮 AI Tahmin Merkezi")
    st.caption("_Markov Zinciri + LSTM Deep Learning + Whale Intelligence_")
    
    # Get current BTC data for predictions
    data = load_json("dashboard_data.json")
    btc_data = data.get("BTCUSDT", {}) if data else {}
    current_price = btc_data.get('price', 85000)
    
    # Calculate recent price change for Markov
    price_history = btc_data.get('price_history', [current_price, current_price])
    if len(price_history) >= 2:
        recent_change = ((price_history[-1] / price_history[-2]) - 1) * 100
    else:
        recent_change = 0
    
    # ==========================================
    # 1. MARKOV CHAIN PREDICTOR
    # ==========================================
    st.markdown("### 📊 Markov Zinciri Tahmin (1-2 Saat)")
    st.caption("_Durum geçiş olasılıklarına dayalı matematiksel model_")
    
    try:
        from src.brain.markov_predictor import MarkovPredictor
        
        markov = MarkovPredictor()
        prediction = markov.predict_1_2_hours(recent_change)
        
        # Visual columns
        m1, m2, m3, m4 = st.columns(4)
        
        # Signal badge
        signal = prediction['combined_signal']
        signal_color = "🟢" if signal == "LONG" else "🔴" if signal == "SHORT" else "⚪"
        
        with m1:
            st.metric("🎯 Sinyal", f"{signal_color} {signal}")
            st.caption(f"Güç: {prediction['signal_strength']:.0f}%")
        
        with m2:
            st.metric("⏰ 1 Saat Tahmin", prediction['1_hour']['direction'])
            st.caption(f"Olasılık: {prediction['1_hour']['probability']:.0f}%")
        
        with m3:
            st.metric("⏰ 2 Saat Tahmin", prediction['2_hour']['direction'])
            st.caption(f"Olasılık: {prediction['2_hour']['probability']:.0f}%")
        
        with m4:
            st.metric("📈 Bullish / 📉 Bearish", 
                     f"{prediction['1_hour']['bullish_probability']:.0f}% / {prediction['1_hour']['bearish_probability']:.0f}%")
            st.caption(f"Mevcut durum: {prediction['current_state']}")
        
        # Probability distribution
        with st.expander("📊 Olasılık Dağılımı"):
            probs = prediction['1_hour']['all_probabilities']
            cols = st.columns(5)
            states = ['STRONG_UP', 'UP', 'NEUTRAL', 'DOWN', 'STRONG_DOWN']
            emojis = ['🚀', '📈', '➡️', '📉', '💥']
            for i, (state, emoji) in enumerate(zip(states, emojis)):
                with cols[i]:
                    st.metric(f"{emoji}", f"{probs[state]:.0f}%")
                    st.caption(state.replace('_', ' '))
        
        # Trend duration estimate
        duration = markov.get_trend_duration_estimate(prediction['current_state'])
        st.info(f"⏱️ **Trend Süresi Tahmini:** {duration['expected_duration_hours']:.1f} saat | "
               f"Tersine Dönüş: {duration['reversal_probability']:.0f}% | "
               f"Momentum: {duration['momentum_strength']}")
        
    except Exception as e:
        st.warning(f"Markov tahmin geçici olarak kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 2. LSTM DEEP LEARNING PREDICTION
    # ==========================================
    st.markdown("### 🧠 LSTM Neural Network Tahmin")
    st.caption("_Deep Learning ile fiyat yönü tahmini_")
    
    try:
        from src.brain.models.lstm_trend import LSTMTrendPredictor
        import pandas as pd
        
        lstm = LSTMTrendPredictor()
        
        # Create simple price dataframe for prediction
        if price_history and len(price_history) >= 24:
            df = pd.DataFrame({'close': price_history[-24:]})
            lstm_pred = lstm.predict(df)
        else:
            lstm_pred = lstm._fallback_prediction(pd.DataFrame({'close': [current_price]}))
        
        l1, l2, l3 = st.columns(3)
        
        with l1:
            dir_emoji = "🟢" if lstm_pred['direction'] == 'UP' else "🔴" if lstm_pred['direction'] == 'DOWN' else "⚪"
            st.metric("🎯 LSTM Yön", f"{dir_emoji} {lstm_pred['direction']}")
        
        with l2:
            st.metric("💪 Güven", f"{lstm_pred['confidence']:.1f}%")
        
        with l3:
            st.metric("🔧 Model", lstm_pred.get('model', 'N/A'))
        
        if 'probabilities' in lstm_pred:
            with st.expander("📊 Yön Olasılıkları"):
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric("📈 UP", f"{lstm_pred['probabilities'].get('UP', 0):.0f}%")
                with p2:
                    st.metric("➡️ NEUTRAL", f"{lstm_pred['probabilities'].get('NEUTRAL', 0):.0f}%")
                with p3:
                    st.metric("📉 DOWN", f"{lstm_pred['probabilities'].get('DOWN', 0):.0f}%")
        
    except Exception as e:
        st.warning(f"LSTM tahmin kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 3. WHALE INTELLIGENCE (Coinglass)
    # ==========================================
    st.markdown("### 🐋 Whale Intelligence")
    st.caption("_Coinglass Hyperliquid - Büyük trader pozisyonları_")
    
    try:
        from src.brain.coinglass_scraper import CoinglassScraper
        
        scraper = CoinglassScraper()
        enhancement = scraper.get_signal_enhancement(current_price)
        
        w1, w2, w3, w4 = st.columns(4)
        
        with w1:
            bias_emoji = "🟢" if enhancement['whale_bias'] == 'LONG' else "🔴" if enhancement['whale_bias'] == 'SHORT' else "⚪"
            st.metric("🐋 Whale Yönelimi", f"{bias_emoji} {enhancement['whale_bias']}")
        
        with w2:
            st.metric("💪 Güven Boost", f"+{enhancement['confidence_boost']}%")
        
        with w3:
            st.metric("👥 Whale Sayısı", enhancement.get('whale_count', 0))
        
        with w4:
            warning = enhancement.get('liquidation_warning', 'NONE')
            warn_emoji = "⚠️" if warning != 'NONE' else "✅"
            st.metric("⚠️ Likidasyon Riski", f"{warn_emoji} {warning.replace('_', ' ')}")
        
        st.caption("_Veriler Coinglass Hyperliquid'den alınmaktadır_")
        
    except Exception as e:
        st.warning(f"Whale intel kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 4. LIQUIDATION HUNTER
    # ==========================================
    st.markdown("### 🎯 Liquidation Hunter")
    st.caption("_Fiyatın çekileceği tasfiye seviyeleri_")
    
    try:
        from src.brain.liquidation_hunter import LiquidationHunter
        
        hunter = LiquidationHunter()
        
        # Run async function sync
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        liq_data = loop.run_until_complete(hunter.calculate_liquidation_levels())
        loop.close()
        
        if liq_data:
            lh1, lh2, lh3 = st.columns(3)
            
            with lh1:
                nearest_long = liq_data.get('nearest_long_liq', 0)
                if nearest_long:
                    st.metric("📈 En Yakın Long Liq", f"${nearest_long:,.0f}")
                else:
                    st.metric("📈 En Yakın Long Liq", "N/A")
            
            with lh2:
                nearest_short = liq_data.get('nearest_short_liq', 0)
                if nearest_short:
                    st.metric("📉 En Yakın Short Liq", f"${nearest_short:,.0f}")
                else:
                    st.metric("📉 En Yakın Short Liq", "N/A")
            
            with lh3:
                total_liq = liq_data.get('total_liquidation_value', 0)
                st.metric("💰 Toplam Liq Değeri", f"${total_liq/1e6:.1f}M" if total_liq else "N/A")
            
            # Liquidation interpretation
            interp = liq_data.get('interpretation', '')
            if interp:
                st.info(f"🎯 **Analiz:** {interp}")
        
        # Close hunter (sync)
        if hasattr(hunter, 'close'):
            try:
                import asyncio
                asyncio.get_event_loop().run_until_complete(hunter.close())
            except:
                pass
        
    except Exception as e:
        st.warning(f"Liquidation hunter kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 5. COMBINED AI SIGNAL
    # ==========================================
    st.markdown("### 🎯 Kombine AI Sinyal")
    st.caption("_Tüm modellerin ağırlıklı kombinasyonu_")
    
    # Combine all signals
    signals = []
    weights = []
    
    # Markov signal
    try:
        if prediction['combined_signal'] == 'LONG':
            signals.append(1)
            weights.append(prediction['signal_strength'] / 100)
        elif prediction['combined_signal'] == 'SHORT':
            signals.append(-1)
            weights.append(prediction['signal_strength'] / 100)
        else:
            signals.append(0)
            weights.append(0.3)
    except:
        pass
    
    # LSTM signal
    try:
        if lstm_pred['direction'] == 'UP':
            signals.append(1)
            weights.append(lstm_pred['confidence'] / 100)
        elif lstm_pred['direction'] == 'DOWN':
            signals.append(-1)
            weights.append(lstm_pred['confidence'] / 100)
        else:
            signals.append(0)
            weights.append(0.3)
    except:
        pass
    
    # Whale signal
    try:
        if enhancement['whale_bias'] == 'LONG':
            signals.append(1)
            weights.append(0.5)
        elif enhancement['whale_bias'] == 'SHORT':
            signals.append(-1)
            weights.append(0.5)
        else:
            signals.append(0)
            weights.append(0.2)
    except:
        pass
    
    # Calculate combined score
    if signals and weights:
        weighted_sum = sum(s * w for s, w in zip(signals, weights))
        total_weight = sum(weights)
        combined_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        if combined_score > 0.3:
            combined_signal = "🟢 LONG"
            combined_confidence = min(100, combined_score * 100)
        elif combined_score < -0.3:
            combined_signal = "🔴 SHORT"
            combined_confidence = min(100, abs(combined_score) * 100)
        else:
            combined_signal = "⚪ BEKLE"
            combined_confidence = 50
        
        cs1, cs2, cs3 = st.columns(3)
        
        with cs1:
            st.metric("🎯 KOMBİNE SİNYAL", combined_signal)
        
        with cs2:
            st.metric("💪 Toplam Güven", f"{combined_confidence:.0f}%")
        
        with cs3:
            st.metric("📊 Model Sayısı", len(signals))
        
        # Signal breakdown
        with st.expander("📊 Sinyal Detayları"):
            st.write("**Model Katkıları:**")
            model_names = ["Markov Chain", "LSTM Neural", "Whale Intel"]
            for i, (name, sig, wgt) in enumerate(zip(model_names, signals, weights)):
                dir_text = "LONG" if sig > 0 else "SHORT" if sig < 0 else "NEUTRAL"
                st.write(f"- **{name}:** {dir_text} (ağırlık: {wgt:.2f})")
    
    st.divider()
    
    # ==========================================
    # 6. SIGNAL ORCHESTRATOR (MERKEZ ORKESTRATÖR)
    # ==========================================
    st.markdown("### 🎯 Signal Orchestrator - Merkez Karar Motoru")
    st.caption("_Tüm modülleri birleştirir, konsensüs sinyali üretir_")
    
    try:
        from src.brain.signal_orchestrator import SignalOrchestrator
        import asyncio
        
        orchestrator = SignalOrchestrator()
        
        # Collect signals
        with st.spinner("Tüm modüllerden sinyal toplanıyor..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_signal = loop.run_until_complete(orchestrator.orchestrate('BTCUSDT'))
            loop.close()
        
        if final_signal:
            so1, so2, so3, so4 = st.columns(4)
            
            with so1:
                dir_emoji = "🟢" if final_signal.direction == 'LONG' else "🔴" if final_signal.direction == 'SHORT' else "⚪"
                st.metric("🎯 ORKESTRE SİNYAL", f"{dir_emoji} {final_signal.direction}")
            
            with so2:
                st.metric("💪 Güven", f"{final_signal.confidence:.0f}%")
            
            with so3:
                st.metric("📊 Konsensüs", f"{final_signal.consensus_ratio:.0f}%")
            
            with so4:
                strength_emoji = "💪💪💪" if final_signal.strength == 'STRONG' else "💪💪" if final_signal.strength == 'MODERATE' else "💪"
                st.metric("⚡ Güç", f"{strength_emoji} {final_signal.strength}")
            
            # Entry/SL/TP
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("📈 Entry", f"${final_signal.entry_price:,.2f}")
            with e2:
                st.metric("🛑 Stop Loss", f"${final_signal.stop_loss:,.2f}")
            with e3:
                st.metric("🎯 Take Profit", f"${final_signal.take_profit:,.2f}")
            with e4:
                st.metric("📊 Risk:Reward", f"{final_signal.risk_reward:.1f}:1")
            
            st.success(f"✅ **{len(final_signal.contributing_modules)} modül** aynı fikirde: {', '.join(final_signal.contributing_modules)}")
        else:
            st.info("⏸️ Şu an güçlü sinyal yok - tüm kriterler karşılanmadı (konsensüs, güven, R:R)")
        
        # Breakdown
        breakdown = orchestrator.get_signal_breakdown()
        with st.expander("📊 Modül Sinyalleri Detayı"):
            for mod in breakdown.get('modules', []):
                emoji = "🟢" if mod['direction'] == 'LONG' else "🔴" if mod['direction'] == 'SHORT' else "⚪"
                st.write(f"{emoji} **{mod['name']}:** {mod['direction']} ({mod['confidence']:.0f}%) - {mod['reasoning']}")
    
    except Exception as e:
        st.warning(f"Signal Orchestrator kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 7. RESEARCH AGENT STATUS
    # ==========================================
    st.markdown("### 🔬 Otonom Araştırma Durumu")
    st.caption("_4 coin için otomatik TradingView grafik analizi_")
    
    try:
        from src.brain.research_agent import ResearchAgent
        
        agent = ResearchAgent()
        
        with st.expander("🔍 Araştırma Başlat"):
            if st.button("🚀 4 Coin İçin Otonom Araştırma Yap"):
                with st.spinner("TradingView, teknik analiz ve on-chain verileri taranıyor..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(agent.conduct_full_research())
                    loop.close()
                
                summary = agent.get_research_summary()
                
                st.success(f"✅ {summary['coins_analyzed']} coin analiz edildi | Genel: {summary.get('overall_market', 'N/A')}")
                
                for symbol, data in summary.get('coins', {}).items():
                    emoji = "🟢" if data['bias'] == 'BULLISH' else "🔴" if data['bias'] == 'BEARISH' else "⚪"
                    st.write(f"{emoji} **{symbol}:** {data['bias']} ({data['confidence']:.0f}%) - {data['findings_count']} bulgu")
                    if data.get('supports'):
                        st.caption(f"  📉 Destek: ${data['supports'][0]:,.0f} | 📈 Direnç: ${data.get('resistances', [0])[0]:,.0f}")
    
    except Exception as e:
        st.warning(f"Research Agent kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 8. SMART TIMING FILTER STATUS
    # ==========================================
    st.markdown("### ⏱️ Sinyal Zamanlama Durumu")
    st.caption("_Spam önleme, kalite kontrolü_")
    
    try:
        from src.brain.smart_timing_filter import SmartTimingFilter
        
        timing = SmartTimingFilter()
        stats = timing.get_statistics()
        cooldown = timing.get_cooldown_status()
        
        t1, t2, t3, t4 = st.columns(4)
        
        with t1:
            st.metric("📊 Toplam Sinyal", stats['total'])
        
        with t2:
            st.metric("📈 Kazanç Oranı", f"{stats['win_rate']:.0f}%")
        
        with t3:
            st.metric("📅 Bugün", f"{stats['signals_today']}/{8}")
        
        with t4:
            st.metric("🔄 Kalan", stats['remaining_today'])
        
        # Cooldown status per coin
        with st.expander("⏰ Coin Bekleme Süreleri"):
            for coin, status in cooldown.items():
                if status['can_signal']:
                    st.write(f"✅ **{coin}:** Sinyal gönderilebilir")
                else:
                    st.write(f"⏳ **{coin}:** {status['wait_minutes']:.0f} dakika bekle")
    
    except Exception as e:
        st.warning(f"Timing Filter kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 9. NEWS SENTIMENT PANEL
    # ==========================================
    st.markdown("### 📰 Haber Sentiment Analizi")
    st.caption("_Kripto haberlerinden piyasa duygu durumu_")
    
    try:
        from src.brain.news_scraper import CryptoNewsScraper
        
        scraper = CryptoNewsScraper()
        scraper.fetch_all_news(max_age_hours=4)
        sentiment = scraper.get_market_sentiment()
        important = scraper.get_important_news(max_items=3)
        
        n1, n2, n3, n4 = st.columns(4)
        
        with n1:
            overall_emoji = "🟢" if sentiment['overall'] == 'BULLISH' else "🔴" if sentiment['overall'] == 'BEARISH' else "⚪"
            st.metric("📊 Genel Sentiment", f"{overall_emoji} {sentiment['overall']}")
        
        with n2:
            st.metric("🟢 Bullish", sentiment['bullish_count'])
        
        with n3:
            st.metric("🔴 Bearish", sentiment['bearish_count'])
        
        with n4:
            st.metric("📰 Toplam Haber", sentiment['total_count'])
        
        if important:
            with st.expander("🔥 Önemli Haberler"):
                for news in important:
                    emoji = "🟢" if news.sentiment == 'BULLISH' else "🔴" if news.sentiment == 'BEARISH' else "⚪"
                    st.write(f"{emoji} **{news.title[:80]}...**")
                    st.caption(f"_{news.source} | {news.impact} impact_")
    
    except Exception as e:
        st.warning(f"News Scraper kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 10. CME GAP TRACKER PANEL
    # ==========================================
    st.markdown("### 📉 CME Gap Tracker")
    st.caption("_Bitcoin CME Futures hafta sonu gap analizi_")
    
    try:
        from src.brain.cme_gap_tracker import CMEGapTracker
        
        tracker = CMEGapTracker()
        gap_status = tracker.get_gap_status()
        
        if gap_status['status'] == 'NO_GAP':
            st.info("📊 Şu an önemli CME gap yok")
        else:
            g1, g2, g3, g4 = st.columns(4)
            
            with g1:
                gap_emoji = "🟢" if gap_status['type'] == 'BULLISH' else "🔴"
                st.metric("📊 Gap Tipi", f"{gap_emoji} {gap_status['type']}")
            
            with g2:
                st.metric("💰 Gap Boyutu", f"${gap_status['size_usd']:,.0f}")
            
            with g3:
                st.metric("📈 Gap %", f"{gap_status['size_pct']:.2f}%")
            
            with g4:
                st.metric("🎯 Doluluk", f"{gap_status['fill_pct']:.0f}%")
            
            # Gap details
            st.write(f"**Cuma Kapanış:** ${gap_status['friday_close']:,.0f} | **Pazartesi Açılış:** ${gap_status['monday_open']:,.0f}")
            st.write(f"**Hedef (Gap Kapanma):** ${gap_status['target']:,.0f}")
            
            signal_emoji = "📈" if gap_status['signal'] == 'LONG' else "📉"
            if gap_status['status'] == 'GAP_FILLED':
                st.success("✅ Gap kapandı!")
            else:
                st.warning(f"{signal_emoji} **Sinyal:** {gap_status['signal']} (gap kapanması bekleniyor)")
    
    except Exception as e:
        st.warning(f"CME Gap Tracker kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 11. OPTIONS FLOW PANEL
    # ==========================================
    st.markdown("### 📈 Options Flow (Deribit)")
    st.caption("_Opsiyon piyasası akıllı para analizi_")
    
    try:
        from src.brain.options_flow import OptionsFlowAnalyzer
        
        analyzer = OptionsFlowAnalyzer()
        options = analyzer.analyze()
        
        if options.get('available'):
            o1, o2, o3, o4 = st.columns(4)
            
            with o1:
                bias_emoji = "🟢" if options['bias'] == 'BULLISH' else "🔴" if options['bias'] == 'BEARISH' else "⚪"
                st.metric("📊 Bias", f"{bias_emoji} {options['bias']}")
            
            with o2:
                st.metric("📈 Call/Put", f"{options['call_put_ratio']:.2f}")
            
            with o3:
                st.metric("🎯 Max Pain", f"${options['max_pain']:,.0f}")
            
            with o4:
                iv_emoji = "🔥" if options['iv_status'] == 'VERY_HIGH' else "❄️" if options['iv_status'] == 'VERY_LOW' else "📊"
                st.metric("📉 IV Rank", f"{iv_emoji} {options['iv_rank']:.0f}%")
            
            # Details
            st.write(f"**Call OI:** {options['call_oi']:,.0f} | **Put OI:** {options['put_oi']:,.0f}")
            
            if options['max_pain_direction'] != 'NEUTRAL':
                dir_emoji = "⬆️" if options['max_pain_direction'] == 'UP' else "⬇️"
                st.write(f"{dir_emoji} Fiyat Max Pain'e doğru çekilebilir ({options['max_pain_distance_pct']:+.1f}%)")
            
            st.caption(f"_{options['iv_note']}_")
        else:
            st.info("📊 Options verisi şu an kullanılamıyor")
    
    except Exception as e:
        st.warning(f"Options Flow kullanılamıyor: {e}")
    
    st.divider()
    
    # ==========================================
    # 12. SIGNAL BACKTEST VALIDATION
    # ==========================================
    st.markdown("### 📊 Sinyal Backtest Doğrulama")
    st.caption("_Geçmiş sinyallerin başarı oranı analizi_")
    
    try:
        from src.notifications.signal_tracker import SignalTracker
        
        tracker = SignalTracker()
        stats = tracker.get_statistics()
        
        b1, b2, b3, b4 = st.columns(4)
        
        with b1:
            st.metric("📊 Toplam Sinyal", stats.get('total_signals', 0))
        
        with b2:
            wr = stats.get('win_rate', 0)
            wr_emoji = "🏆" if wr >= 60 else "📈" if wr >= 50 else "📉"
            st.metric("🎯 Win Rate", f"{wr_emoji} {wr:.1f}%")
        
        with b3:
            st.metric("✅ Kazanç", stats.get('wins', 0))
        
        with b4:
            st.metric("❌ Kayıp", stats.get('losses', 0))
        
        # Profit summary
        net = stats.get('net_profit_pct', 0)
        if net != 0:
            profit_emoji = "🟢" if net > 0 else "🔴"
            st.metric("💰 Net Kar/Zarar", f"{profit_emoji} {net:+.2f}%")
        
        # Active signals
        active = tracker.get_active_signals()
        if active:
            with st.expander(f"📍 Aktif Sinyaller ({len(active)})"):
                for sig in active:
                    emoji = "📈" if sig['direction'] == 'LONG' else "📉"
                    st.write(f"{emoji} **{sig['symbol']}** {sig['direction']} @ ${sig['entry_price']:,.2f}")
                    st.caption(f"SL: ${sig['stop_loss']:,.2f} | TP1: ${sig['take_profit_1']:,.2f}")
    
    except Exception as e:
        st.warning(f"Backtest Validation kullanılamıyor: {e}")
    
    st.divider()
    
    # Refresh button
    if st.button("🔄 Tahminleri Yenile"):
        st.cache_data.clear()
        st.rerun()

# ==========================================
# 🎯 AI MODULE MONITOR (Phase 92)
# 35 AI modülünün durumunu göster
# ==========================================
if page == "🎯 AI Module Monitor":
    st.sidebar.markdown("---")
    st.sidebar.info("**35 AI Modül Monitörü**")
    
    st.markdown("## 🧠 AI Module Status Dashboard")
    st.caption("_SignalOrchestrator'dan canlı modül durumları - 35 aktif modül_")
    
    if st.button("🔄 Modülleri Yeniden Tara"):
        st.cache_data.clear()
        st.rerun()
    
    try:
        from src.brain.signal_orchestrator import SignalOrchestrator
        
        orchestrator = SignalOrchestrator()
        
        # Modül özeti
        st.markdown("### 📊 Modül Özeti")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_modules = len(orchestrator.weights)
        
        with col1:
            st.metric("🧠 Toplam Modül", total_modules)
        
        with col2:
            st.metric("⚡ Aktif", total_modules)
        
        with col3:
            st.metric("📊 Ağırlık Toplamı", f"{sum(orchestrator.weights.values()):.0%}")
        
        with col4:
            st.metric("🔧 Version", "v35.0")
        
        st.markdown("---")
        
        # Modül kategorileri
        st.markdown("### 📋 Modül Listesi (Ağırlıklar)")
        st.caption("""
_**📊 Ağırlık % Ne Demek?**_
• Her modülün yüzdesi = **Toplam sinyale katkı oranı** (güven değil!)
• Örnek: %6 = Bu modül nihai kararda %6 ağırlık taşır
• Yüksek ağırlık (>5%) = Daha etkili modül
• Düşük ağırlık (<3%) = Destekleyici modül
• Toplam ağırlık = %100 (tüm modüller birlikte)
        """)
        
        # Phase bazlı gruplandırma
        phases = {
            "🔮 Core AI (Phase 1-30)": [
                ('MarkovPredictor', 'Markov tahmini'),
                ('LSTMTrend', 'LSTM trend'),
                ('ResearchAgent', 'Araştırma ajanı'),
                ('SMCAnalyzer', 'Smart Money'),
                ('WhaleIntelligence', 'Balina takibi'),
                ('LiquidationHunter', 'Likidasyon avı'),
                ('PredictiveAnalyzer', 'Öncü analiz'),
                ('NewsSentiment', 'Haber duygusu'),
                ('CMEGapTracker', 'CME boşluk'),
                ('OptionsFlow', 'Opsiyon akışı'),
                ('OnChainIntel', 'On-chain'),
                ('TradingViewTA', 'TradingView'),
            ],
            "📊 Phase 61-66": [
                ('TwitterSentiment', 'Twitter'),
                ('OrderBookDepth', 'Orderbook'),
                ('EnsembleModel', 'Ensemble'),
                ('MultiTimeframe', 'MTF'),
                ('GoogleTrends', 'Google Trends'),
            ],
            "🚨 Sudden Movement (Phase 71-75)": [
                ('BollingerSqueeze', 'Bollinger sıkışma'),
                ('LiquidationCascade', 'Likidasyon cascade'),
                ('VolumeSpike', 'Hacim artışı'),
                ('TakerFlowDelta', 'Taker akışı'),
                ('ExchangeDivergence', 'Borsa divergence'),
            ],
            "📈 CoinGlass (Phase 77-84)": [
                ('CGLiquidationMap', 'CG Likidasyon haritası'),
                ('CGWhaleOrders', 'CG Whale emirleri'),
                ('CGWhaleAlerts', 'CG Whale uyarıları'),
                ('CGOIDelta', 'CG OI Delta'),
                ('CGFundingExtreme', 'CG Funding'),
                ('CGTopTraderLS', 'CG Top Trader L/S'),
                ('CGOrderbookDelta', 'CG Orderbook'),
                ('CGExchangeBalance', 'CG Borsa bakiyesi'),
            ],
            "🎯 Advanced (Phase 86-90)": [
                ('CandlePatterns', 'Mum formasyonları'),
                ('VolatilityPredictor', 'Volatilite tahmini'),
                ('CrossAssetCorr', 'Çapraz korelasyon'),
                ('CVDAnalyzer', 'CVD analizi'),
                ('CompositeAlert', 'Birleşik alarm'),
            ],
        }
        
        for phase_name, modules in phases.items():
            with st.expander(phase_name, expanded=True):
                cols = st.columns(4)
                for i, (mod_name, mod_desc) in enumerate(modules):
                    weight = orchestrator.weights.get(mod_name, 0)
                    with cols[i % 4]:
                        if weight > 0:
                            st.success(f"**{mod_name}**\n{weight:.0%}")
                            st.caption(mod_desc)
                        else:
                            st.warning(f"**{mod_name}**\n❌ Kayıtlı değil")
        
        st.markdown("---")
        
        # Win Rate Stats
        st.markdown("### 🏆 Sinyal Performansı")
        
        try:
            from src.brain.signal_performance_tracker import get_tracker
            tracker = get_tracker()
            stats_7d = tracker.get_win_rate(days=7)
            stats_30d = tracker.get_win_rate(days=30)
            
            w1, w2, w3, w4 = st.columns(4)
            
            with w1:
                st.metric("📊 Son 7 Gün", f"{stats_7d['total_signals']} sinyal")
            
            with w2:
                wr = stats_7d['win_rate']
                emoji = "🏆" if wr >= 60 else "📈" if wr >= 50 else "📉"
                st.metric("🎯 Win Rate", f"{emoji} {wr:.1f}%")
            
            with w3:
                st.metric("✅ Kazanan", stats_7d['winners'])
            
            with w4:
                st.metric("❌ Kaybeden", stats_7d['losers'])
            
            st.caption(f"_Son 30 gün: {stats_30d['total_signals']} sinyal, %{stats_30d['win_rate']:.1f} win rate_")
            
        except Exception as e:
            st.info("📊 Henüz sinyal performans verisi yok. İlk sinyaller kaydedildikten sonra görüntülenecek.")
        
    except Exception as e:
        st.error(f"SignalOrchestrator yüklenemedi: {e}")
        st.info("Engine çalışmıyor olabilir. Railway'de deploy edildiğinden emin olun.")
