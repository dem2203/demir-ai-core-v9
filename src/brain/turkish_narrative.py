import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("TURKISH_NARRATIVE")

class TurkishNarrativeEngine:
    """
    Template-Based Turkish AI Narrative Generator (Şablon Tabanlı Türkçe AI Anlatıcısı)
    
    Generates detailed human-readable Turkish explanations of AI trading decisions
    without requiring external API calls (GPT-4o independent).
    
    (Harici API çağrıları gerektirmeden AI ticaret kararlarının detaylı, 
    insan okuyabilir Türkçe açıklamalarını oluşturur)
    """
    
    @staticmethod
    def generate_full_report(symbol: str, snapshot: Dict) -> str:
        """
        Generate comprehensive Turkish AI analysis report
        (Kapsamlı Türkçe AI analiz raporu oluştur)
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            snapshot: Complete market snapshot with brain_state
            
        Returns:
            Formatted markdown report in Turkish
        """
        if not snapshot:
            return "⚠️ Snapshot verisi eksik / Snapshot data missing"
        
        # Extract key data
        decision = snapshot.get('ai_decision', 'NEUTRAL')
        confidence = snapshot.get('ai_confidence', 0)
        price = snapshot.get('price', 0)
        reason = snapshot.get('reason', '')
        brain_state = snapshot.get('brain_state', {})
        visual_data = snapshot.get('visual_analysis', {})
        sentiment_data = snapshot.get('sentiment_data', {})
        
        # Build report sections
        header = TurkishNarrativeEngine._build_header(symbol, decision, confidence, price)
        layers_analysis = TurkishNarrativeEngine._analyze_all_layers(brain_state, snapshot, visual_data, sentiment_data)
        risks = TurkishNarrativeEngine._generate_risk_analysis(snapshot, brain_state)
        recommendation = TurkishNarrativeEngine._build_trade_recommendation(snapshot)
        
        # Combine into full report
        report = f"""{header}

{layers_analysis}

{risks}

{recommendation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_⚠️ Bu analiz AI tarafından otomatik oluşturulmuştur. Yatırım tavsiyesi değildir._
_📅 Oluşturulma: {datetime.now().strftime('%d.%m.%Y %H:%M')} UTC_
"""
        return report
    
    @staticmethod
    def _build_header(symbol: str, decision: str, confidence: float, price: float) -> str:
        """Build report header with decision summary"""
        decision_tr = {
            'BUY': '🟢 **ALIŞ**',
            'SELL': '🔴 **SATIŞ**',
            'NEUTRAL': '⚪ **NÖTR (BEKLEYİŞ)**'
        }.get(decision, decision)
        
        conf_level = ("Çok Yüksek ⭐⭐⭐" if confidence > 80 else
                     "Yüksek ⭐⭐" if confidence > 65 else
                     "Orta ⭐" if confidence > 50 else
                     "Düşük")
        
        return f"""# 📊 DEMIR AI - ANALİZ RAPORU
## {symbol}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 🎯 KARAR: {decision_tr}
- **Güven Seviyesi:** %{confidence:.1f} ({conf_level})
- **Güncel Fiyat:** ${price:,.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    @staticmethod
    def _analyze_all_layers(brain_state: Dict, snapshot: Dict, visual_data: Dict, sentiment_data: Dict) -> str:
        """Analyze all AI layers and their contributions"""
        sections = []
        sections.append("### 🧠 NEDEN BU KARARI VERDİM?\n")
        
        # 1. Technical Indicators
        tech_section = TurkishNarrativeEngine._analyze_technical(brain_state, snapshot)
        if tech_section:
            sections.append(f"#### 1. 📊 Teknik Göstergeler\n{tech_section}")
        
        # 2. TimeNet / LSTM
        lstm_section = TurkishNarrativeEngine._analyze_timenet(brain_state)
        if lstm_section:
            sections.append(f"#### 2. 🧠 TimeNet (Derin Öğrenme Modeli)\n{lstm_section}")
        
        # 3. Visual Cortex
        visual_section = TurkishNarrativeEngine._analyze_visual(visual_data)
        if visual_section:
            sections.append(f"#### 3. 👁️ Visual Cortex (Grafik Analizi)\n{visual_section}")
        
        # 4. On-Chain
        onchain_section = TurkishNarrativeEngine._analyze_onchain(snapshot, brain_state)
        if onchain_section:
            sections.append(f"#### 4. 💰 On-Chain Intelligence (Zincir Üstü Veriler)\n{onchain_section}")
        
        # 5. Sentiment
        sentiment_section = TurkishNarrativeEngine._analyze_sentiment(sentiment_data)
        if sentiment_section:
            sections.append(f"#### 5. 📰 Sentiment Analizi (Piyasa Duygusu)\n{sentiment_section}")
        
        # 6. Macro
        macro_section = TurkishNarrativeEngine._analyze_macro(snapshot)
        if macro_section:
            sections.append(f"#### 6. 🌍 Makroekonomik Durum\n{macro_section}")
        
        return "\n\n".join(sections)
    
    @staticmethod
    def _analyze_technical(brain_state: Dict, snapshot: Dict) -> str:
        """Analyze technical indicators layer"""
        tech_attention = brain_state.get('tech_attention', 0)
        tech_bias = snapshot.get('tech_bias', 'NEUTRAL')
        
        if tech_attention < 0.05:
            return ""
        
        weight_pct = tech_attention * 100
        
        bias_text = {
            'STRONG_BULLISH': '**Güçlü Yükseliş** 📈',
            'BULLISH': '**Yükseliş** ↗️',
            'NEUTRAL': 'Nötr ↔️',
            'BEARISH': '**Düşüş** ↘️',
            'STRONG_BEARISH': '**Güçlü Düşüş** 📉'
        }.get(tech_bias, tech_bias)
        
        # Additional technical details
        patterns = []
        if snapshot.get('candlestick_latest'):
            patterns.append(f"Mum Formasyonu: **{snapshot['candlestick_latest']}**")
        if snapshot.get('chart_pattern_latest'):
            patterns.append(f"Chart Pattern: **{snapshot['chart_pattern_latest']}**")
        if snapshot.get('divergence_latest'):
            patterns.append(f"Divergence: **{snapshot['divergence_latest']}**")
        
        pattern_text = "\n  - " + "\n  - ".join(patterns) if patterns else ""
        
        return f"""- **Ağırlık:** %{weight_pct:.0f}
- **Genel Eğilim:** {bias_text}
- 20+ teknik gösterge (RSI, MACD, Bollinger, vb.) analiz edildi{pattern_text}"""
    
    @staticmethod
    def _analyze_timenet(brain_state: Dict) -> str:
        """Analyze TimeNet/LSTM predictions"""
        lstm_attention = brain_state.get('lstm_attention', 0)
        
        if lstm_attention < 0.05:
            return ""
        
        weight_pct = lstm_attention * 100
        
        return f"""- **Ağırlık:** %{weight_pct:.0f}
- Transformer tabanlı derin öğrenme modeli
- Son 60 saatlik fiyat hareketini analiz etti
- Gizli pattern'leri tespit ediyor"""
    
    @staticmethod
    def _analyze_visual(visual_data: Dict) -> str:
        """Analyze Visual Cortex output"""
        if not visual_data or visual_data.get('visual_score', 0) <= 50:
            return "- ⚠️ Görsel analiz şu an aktif değil veya sonuç belirsiz"
        
        trend = visual_data.get('trend', 'NEUTRAL')
        score = visual_data.get('visual_score', 0)
        pattern = visual_data.get('pattern', 'None')
        
        trend_text = {
            'BULLISH': '**Yükseliş** 📈',
            'BEARISH': '**Düşüş** 📉',
            'NEUTRAL': 'Belirsiz ↔️'
        }.get(trend, trend)
        
        return f"""- **Görsel Trend:** {trend_text} (Skor: {score}/100)
- **Pattern:** {pattern}
- Gemini AI chart'ı görsel olarak analiz etti"""
    
    @staticmethod
    def _analyze_onchain(snapshot: Dict, brain_state: Dict) -> str:
        """Analyze on-chain data"""
        onchain_attention = brain_state.get('onchain_attention', 0)
        onchain_signal = snapshot.get('onchain_signal', 'NEUTRAL')
        onchain_score = snapshot.get('onchain_score', 0)
        
        if onchain_attention < 0.05 and abs(onchain_score) < 5:
            return ""
        
        weight_pct = onchain_attention * 100
        
        signal_text = {
            'BUY': '**Birikim (Accumulation)** ⬆️',
            'SELL': '**Dağıtım (Distribution)** ⬇️',
            'NEUTRAL': 'Nötr'
        }.get(onchain_signal, onchain_signal)
        
        funding = snapshot.get('funding_rate', 0)
        funding_text = ("Pozitif (Long baskısı)" if funding > 0.01 else
                       "Negatif (Short baskısı)" if funding < -0.01 else
                       "Nötr")
        
        return f"""- **Ağırlık:** %{weight_pct:.0f}
- **Sinyal:** {signal_text} (Skor: {onchain_score})
- **Funding Rate:** {funding:.4%} ({funding_text})
- Whale hareketleri ve exchange akışları izlendi"""
    
    @staticmethod
    def _analyze_sentiment(sentiment_data: Dict) -> str:
        """Analyze market sentiment"""
        if not sentiment_data:
            return ""
        
        sentiment = sentiment_data.get('sentiment', 'NEUTRAL')
        score = sentiment_data.get('composite_score', 0)
        fg = sentiment_data.get('fear_greed_index', 50)
        
        sentiment_text = {
            'EXTREME_FEAR': '**Aşırı Korku** 😱 (Fırsat olabilir)',
            'FEAR': '**Korku** 😰',
            'NEUTRAL': 'Nötr 😐',
            'GREED': '**Açgözlülük** 😁',
            'EXTREME_GREED': '**Aşırı Açgözlülük** 🔥 (Dikkatli olun)',
            'BULLISH': '**İyimser** 🟢',
            'BEARISH': '**Kötümser** 🔴'
        }.get(sentiment, sentiment)
        
        fg_label = ("Aşırı Açgözlülük" if fg >= 75 else
                   "Açgözlülük" if fg >= 55 else
                   "Nötr" if fg >= 45 else
                   "Korku" if fg >= 25 else
                   "Aşırı Korku")
        
        return f"""- **Genel Duygu:** {sentiment_text}
- **Fear & Greed Index:** {fg}/100 ({fg_label})
- Reddit ve sosyal medya verileri analiz edildi"""
    
    @staticmethod
    def _analyze_macro(snapshot: Dict) -> str:
        """Analyze macro economic context"""
        dxy = snapshot.get('dxy', 0)
        vix = snapshot.get('vix', 0)
        macro_score = snapshot.get('macro_score', 0)
        
        if dxy == 0 and vix == 0:
            return ""
        
        dxy_effect = ("USD güçleniyor → Crypto için olumsuz 📉" if dxy > 105 else
                     "USD zayıflıyor → Crypto için olumlu 📈" if dxy < 102 else
                     "USD dengeli")
        
        vix_effect = ("Yüksek volatilite → Riskli ortam ⚠️" if vix > 20 else
                     "Düşük volatilite → Sakin piyasa ✅" if vix < 15 else
                     "Normal volatilite")
        
        macro_text = ("Crypto için olumsuz ortam" if macro_score < -10 else
                     "Crypto için olumlu ortam" if macro_score > 10 else
                     "Nötr ortam")
        
        return f"""- **DXY (Dolar Endeksi):** {dxy:.2f} → {dxy_effect}
- **VIX (Korku Endeksi):** {vix:.2f} → {vix_effect}
- **Makro Skor:** {macro_score} ({macro_text})"""
    
    @staticmethod
    def _generate_risk_analysis(snapshot: Dict, brain_state: Dict) -> str:
        """Generate risk warnings"""
        risks = []
        
        # Confidence risk
        conf = snapshot.get('ai_confidence', 0)
        if conf < 60:
            risks.append("⚠️ **Düşük Güven:** Pozisyon büyüklüğünü küçük tutun")
        
        # Macro risk
        vix = snapshot.get('vix', 0)
        if vix > 25:
            risks.append(f"⚠️ **Yüksek VIX ({vix:.1f}):** Piyasa volatil, ani hareketler olabilir")
        
        # Pattern conflict
        pattern_bias = snapshot.get('pattern_bias', '')
        tech_bias = snapshot.get('tech_bias', '')
        if pattern_bias and tech_bias and pattern_bias != tech_bias and 'NEUTRAL' not in pattern_bias:
            risks.append("⚠️ **Pattern Çelişkisi:** Teknik ve Pattern sinyalleri anlaşamadı")
        
        # Visual cortex inactive
        visual_score = brain_state.get('visual_score', 50)
        if visual_score <= 50:
            risks.append("ℹ️ **Visual Cortex:** Chart görsel analizi yapılamadı")
        
        if not risks:
            risks.append("✅ Belirgin risk sinyali tespit edilmedi")
        
        risks_text = "\n".join(f"- {r}" for r in risks)
        
        return f"""### ⚠️ RİSKLER VE UYARILAR

{risks_text}"""
    
    @staticmethod
    def _build_trade_recommendation(snapshot: Dict) -> str:
        """Build trade recommendation section"""
        decision = snapshot.get('ai_decision', 'NEUTRAL')
        price = snapshot.get('price', 0)
        
        # Calculate TP/SL levels (simplified - these should come from snapshot if available)
        fib_resistance = snapshot.get('fib_resistance', price * 1.02)
        fib_support = snapshot.get('fib_support', price * 0.98)
        kelly_size = snapshot.get('kelly_size', '2.0')
        
        if decision == 'BUY':
            entry = price
            tp = fib_resistance
            sl = fib_support
            pnl_tp = ((tp - entry) / entry) * 100
            pnl_sl = ((sl - entry) / entry) * 100
        elif decision == 'SELL':
            entry = price
            tp = fib_support
            sl = fib_resistance
            pnl_tp = ((entry - tp) / entry) * 100
            pnl_sl = ((entry - sl) / entry) * 100
        else:
            return """### 💼 TAVSİYE

**🔵 NÖTR** - Şu an beklemek daha uygun. Net sinyal bekleniyor."""
        
        side_text = "LONG (Alış)" if decision == 'BUY' else "SHORT (Satış)"
        
        return f"""### 💼 İŞLEM TAVSİYESİ

- **Yön:** {side_text}
- **📍 Giriş:** ${entry:,.2f}
- **🎯 Kar Al (TP):** ${tp:,.2f} (+%{abs(pnl_tp):.2f})
- **🛡️ Zarar Durdur (SL):** ${sl:,.2f} (-%{abs(pnl_sl):.2f})
- **💰 Pozisyon Büyüklüğü:** ~{kelly_size}% (Risk yönetimi)

**Risk/Reward Oranı:** {abs(pnl_tp/pnl_sl):.2f}:1"""
