import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("EARLY_WARNING")

class EarlyWarningSystem:
    """
    PROAKTIF ERKEN UYARI SİSTEMİ (Proactive Early Warning System)
    
    Piyasa hareketlerini ÖNCEDEN tespit eder ve trader'ı ERKEN uyarır.
    İndikatörlerden FARKLI olarak:
    - Pattern tamamlanmadan ÖNCE uyarır
    - Breakout olmadan ÖNCE hazırlık sinyali verir
    - Whale hareketi başlamadan ÖNCE alert gönderir
    """
    
    # Alert thresholds
    PATTERN_FORMATION_THRESHOLD = 0.7  # Pattern %70 tamamlandığında uyar
    BREAKOUT_PROBABILITY_MIN = 0.65    # %65+ breakout olasılığında uyar
    WHALE_ACCUMULATION_THRESHOLD = 3   # 3+ whale trade = accumulation alert
    
    @staticmethod
    def analyze_for_early_warnings(symbol: str, snapshot: Dict, visual_data: Dict = None) -> List[Dict]:
        """
        Tüm veriyi analiz edip ERKEN UYARI listesi döndürür.
        
        Returns:
            List of warning dicts: [{'type': 'SETUP_FORMING', 'message': '...', 'priority': 'HIGH'}, ...]
        """
        warnings = []
        
        # 1. Pattern Formation Check (Pattern oluşmaya başladı mı?)
        pattern_warning = EarlyWarningSystem._check_pattern_formation(snapshot, visual_data)
        if pattern_warning:
            warnings.append(pattern_warning)
        
        # 2. Pre-Breakout Detection (Breakout yakın mı?)
        breakout_warning = EarlyWarningSystem._check_pre_breakout(snapshot, visual_data)
        if breakout_warning:
            warnings.append(breakout_warning)
        
        # 3. Whale Activity Warning (Balina hareketi var mı?)
        whale_warning = EarlyWarningSystem._check_whale_activity(snapshot)
        if whale_warning:
            warnings.append(whale_warning)
        
        # 4. Momentum Shift Detection (Momentum değişimi?)
        momentum_warning = EarlyWarningSystem._check_momentum_shift(snapshot)
        if momentum_warning:
            warnings.append(momentum_warning)
        
        # 5. Divergence Warning (Divergence oluşuyor mu?)
        divergence_warning = EarlyWarningSystem._check_divergence_forming(snapshot)
        if divergence_warning:
            warnings.append(divergence_warning)
        
        # 6. Multi-Timeframe Alignment (TF'ler hizalanıyor mu?)
        mtf_warning = EarlyWarningSystem._check_mtf_alignment(snapshot)
        if mtf_warning:
            warnings.append(mtf_warning)
        
        # === PHASE 32: DERIVATIVES-BASED EARLY WARNINGS ===
        
        # 7. Funding Rate Extreme (Aşırı funding rate)
        derivatives = snapshot.get('derivatives', {})
        funding_warning = EarlyWarningSystem._check_funding_rate_extreme(derivatives)
        if funding_warning:
            warnings.append(funding_warning)
        
        # 8. Long/Short Ratio Extreme (Aşırı L/S oranı)
        ls_warning = EarlyWarningSystem._check_long_short_extreme(derivatives)
        if ls_warning:
            warnings.append(ls_warning)
        
        # 9. Open Interest Anomaly (OI anormalliği)
        oi_warning = EarlyWarningSystem._check_oi_anomaly(derivatives)
        if oi_warning:
            warnings.append(oi_warning)
        
        # 10. Fear & Greed Extreme (Aşırı korku/açgözlülük)
        sentiment = snapshot.get('sentiment', {})
        fg_warning = EarlyWarningSystem._check_fear_greed_extreme(sentiment)
        if fg_warning:
            warnings.append(fg_warning)
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        warnings.sort(key=lambda x: priority_order.get(x.get('priority', 'LOW'), 3))
        
        return warnings
    
    @staticmethod
    def _check_pattern_formation(snapshot: Dict, visual_data: Dict) -> Optional[Dict]:
        """Pattern oluşmaya BAŞLADIĞINDA uyar (tamamlanmadan önce)"""
        
        pattern_status = None
        if visual_data:
            pattern_status = visual_data.get('pattern_status')
            pattern = visual_data.get('pattern', 'None')
        else:
            pattern = snapshot.get('chart_pattern_latest')
        
        if not pattern or pattern == 'None':
            return None
        
        # Check if pattern is forming (not complete yet)
        if pattern_status == 'FORMING' or pattern_status == 'BREAKOUT_IMMINENT':
            return {
                'type': 'PATTERN_FORMING',
                'priority': 'HIGH',
                'title': f'⚡ SETUP OLUŞUYOR: {pattern}',
                'message': f'{pattern} formasyonu oluşmakta. Breakout yaklaşıyor, takipte kal!',
                'action': 'Giriş için hazır ol, onay bekle',
                'pattern': pattern,
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    @staticmethod
    def _check_pre_breakout(snapshot: Dict, visual_data: Dict) -> Optional[Dict]:
        """Breakout OLMADAN ÖNCE uyar"""
        
        # Check Gemini prediction for breakout probability
        if visual_data:
            probability = visual_data.get('probability', 0)
            trend = visual_data.get('trend', 'NEUTRAL')
            early_entry = visual_data.get('early_entry_price')
            target = visual_data.get('target_price')
            time_horizon = visual_data.get('time_horizon', 'N/A')
            
            if probability >= 70 and trend != 'NEUTRAL':
                direction = 'YUKARI 📈' if trend == 'BULLISH' else 'AŞAĞI 📉'
                return {
                    'type': 'PRE_BREAKOUT',
                    'priority': 'CRITICAL',
                    'title': f'🚨 BREAKOUT YAKLAŞIYOR: {direction}',
                    'message': f'AI %{probability} olasılıkla {direction} hareket bekliyor.',
                    'early_entry': early_entry,
                    'target': target,
                    'time_horizon': time_horizon,
                    'action': f'Erken giriş: ${early_entry:,.0f}' if early_entry else 'Giriş için bekle',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Fallback: Check technical confluence
        tech_bias = snapshot.get('tech_bias', 'NEUTRAL')
        pattern_bias = snapshot.get('pattern_bias', 'NEUTRAL')
        fractal_score = snapshot.get('fractal_score', 0)
        
        if fractal_score >= 80 and tech_bias == pattern_bias and 'STRONG' in str(tech_bias):
            direction = 'YUKARI' if 'BULL' in tech_bias else 'AŞAĞI'
            return {
                'type': 'CONFLUENCE_ALERT',
                'priority': 'HIGH',
                'title': f'⚡ GÜÇLÜ CONFLUENCE: {direction}',
                'message': f'Teknik + Pattern + MTF hepsi {direction} yönünde. Hareket yakın!',
                'action': 'Giriş hazırlığı yap',
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    @staticmethod
    def _check_whale_activity(snapshot: Dict) -> Optional[Dict]:
        """Whale birikimi veya dağıtımı başladığında uyar"""
        
        whale_support = snapshot.get('whale_support', 0)
        whale_resistance = snapshot.get('whale_resistance', 0)
        onchain_signal = snapshot.get('onchain_signal', 'NEUTRAL')
        onchain_score = snapshot.get('onchain_score', 0)
        
        # Strong whale accumulation
        if onchain_score >= 20 and onchain_signal in ['BUY', 'STRONG_BUY']:
            return {
                'type': 'WHALE_ACCUMULATION',
                'priority': 'HIGH',
                'title': '🐋 WHALE BİRİKİMİ TESPİT EDİLDİ',
                'message': f'Büyük oyuncular alış yapıyor (On-Chain Score: {onchain_score})',
                'action': 'Whale\'lerle birlikte hareket etmeyi düşün',
                'whale_support': whale_support,
                'timestamp': datetime.now().isoformat()
            }
        
        # Strong whale distribution
        if onchain_score <= -20 and onchain_signal in ['SELL', 'STRONG_SELL']:
            return {
                'type': 'WHALE_DISTRIBUTION',
                'priority': 'HIGH',
                'title': '🐋 WHALE DAĞITIMI TESPİT EDİLDİ',
                'message': f'Büyük oyuncular satış yapıyor (On-Chain Score: {onchain_score})',
                'action': 'Dikkatli ol, pozisyonları koru',
                'whale_resistance': whale_resistance,
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    @staticmethod
    def _check_momentum_shift(snapshot: Dict) -> Optional[Dict]:
        """Momentum değişimi başladığında uyar"""
        
        brain_state = snapshot.get('brain_state', {})
        tech_attention = brain_state.get('tech_attention', 0)
        pattern_attention = brain_state.get('pattern_attention', 0)
        
        # High technical attention = momentum shift possible
        if tech_attention >= 0.4:
            volume_signal = snapshot.get('volume_signal', '')
            
            if 'CLIMAX' in volume_signal or 'SPIKE' in volume_signal:
                return {
                    'type': 'MOMENTUM_SHIFT',
                    'priority': 'MEDIUM',
                    'title': '⚡ MOMENTUM DEĞİŞİMİ OLASI',
                    'message': f'Yüksek teknik aktivite ({tech_attention:.0%}) + Volume anomali ({volume_signal})',
                    'action': 'Yön değişimine hazır ol',
                    'timestamp': datetime.now().isoformat()
                }
        
        return None
    
    @staticmethod
    def _check_divergence_forming(snapshot: Dict) -> Optional[Dict]:
        """Divergence oluşmaya başladığında uyar"""
        
        divergence = snapshot.get('divergence_latest')
        
        if divergence and divergence != 'None':
            is_bullish = 'BULLISH' in divergence
            direction = 'YUKARI dönüş' if is_bullish else 'AŞAĞI dönüş'
            
            return {
                'type': 'DIVERGENCE_FORMING',
                'priority': 'MEDIUM',
                'title': f'📊 DIVERGENCE TESPİT: {divergence}',
                'message': f'Fiyat-indikatör uyumsuzluğu. {direction} olasılığı artıyor.',
                'action': 'Trend dönüşü için hazır ol',
                'divergence': divergence,
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    @staticmethod
    def _check_mtf_alignment(snapshot: Dict) -> Optional[Dict]:
        """Multi-timeframe hizalanma oluştuğunda uyar"""
        
        fractal_score = snapshot.get('fractal_score', 0)
        
        if fractal_score >= 85:
            # Check 4H trend
            reason = snapshot.get('reason', '')
            htf_trend = 'BULL' if '4H Trend: BULL' in reason else 'BEAR' if '4H Trend: BEAR' in reason else None
            
            if htf_trend:
                direction = 'YUKARI' if htf_trend == 'BULL' else 'AŞAĞI'
                return {
                    'type': 'MTF_ALIGNMENT',
                    'priority': 'HIGH',
                    'title': f'🎯 TÜM ZAMAN DİLİMLERİ HİZALANDI: {direction}',
                    'message': f'15m + 1H + 4H aynı yöne işaret ediyor (Score: {fractal_score})',
                    'action': f'{direction} pozisyon için ideal zaman',
                    'fractal_score': fractal_score,
                    'timestamp': datetime.now().isoformat()
                }
        
        return None
    
    @staticmethod
    def format_warnings_for_display(warnings: List[Dict]) -> str:
        """Dashboard veya Telegram için uyarıları formatla"""
        
        if not warnings:
            return "✅ Şu an aktif erken uyarı yok."
        
        lines = ["🚨 **ERKEN UYARILAR**\n"]
        
        for w in warnings:
            priority_emoji = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '⚪'
            }.get(w.get('priority'), '⚪')
            
            lines.append(f"{priority_emoji} **{w.get('title', 'Uyarı')}**")
            lines.append(f"   {w.get('message', '')}")
            lines.append(f"   ➡️ {w.get('action', '')}")
            lines.append("")
        
        return "\n".join(lines)
    
    # =====================================================
    # PHASE 32: DERIVATIVES & SENTIMENT EARLY WARNINGS
    # =====================================================
    
    @staticmethod
    def _check_funding_rate_extreme(derivatives: Dict) -> Optional[Dict]:
        """
        Funding Rate extreme değerdeyse uyar.
        
        Extreme Positive (>0.05%) → Çok fazla long → Short squeeze riski
        Extreme Negative (<-0.03%) → Çok fazla short → Long squeeze riski
        """
        funding_rate = derivatives.get('funding_rate', 0)
        
        if not funding_rate:
            return None
        
        # Convert to percentage for easier reading
        fr_pct = funding_rate * 100
        
        if fr_pct >= 0.05:  # Very high positive
            return {
                'type': 'FUNDING_RATE_EXTREME',
                'title': '⚠️ FUNDING RATE ÇOK YÜKSEK',
                'message': f'Funding Rate: %{fr_pct:.3f} - Aşırı long pozisyon birikimi. Likidasyon riski!',
                'action': 'Long pozisyon açmadan önce düzeltme bekle',
                'priority': 'HIGH' if fr_pct >= 0.1 else 'MEDIUM',
                'data': {'funding_rate': fr_pct}
            }
        elif fr_pct <= -0.03:  # Very negative
            return {
                'type': 'FUNDING_RATE_EXTREME',
                'title': '⚠️ FUNDING RATE NEGATİF',
                'message': f'Funding Rate: %{fr_pct:.3f} - Aşırı short pozisyon birikimi. Squeeze riski!',
                'action': 'Short squeeze ihtimali - dikkatli ol',
                'priority': 'MEDIUM',
                'data': {'funding_rate': fr_pct}
            }
        
        return None
    
    @staticmethod
    def _check_long_short_extreme(derivatives: Dict) -> Optional[Dict]:
        """
        Long/Short Ratio aşırı değerlerde uyar.
        
        > 2.0 → Çok fazla long → Tehlike
        < 0.5 → Çok fazla short → Squeeze riski
        """
        ls_ratio = derivatives.get('long_short_ratio', 1.0)
        
        if not ls_ratio or ls_ratio == 1.0:
            return None
        
        if ls_ratio >= 2.0:
            return {
                'type': 'LONG_SHORT_EXTREME',
                'title': '🚨 AŞIRI LONG POZİSYON',
                'message': f'Long/Short: {ls_ratio:.2f} - Piyasa aşırı bullish. Düzeltme riski!',
                'action': 'Long pozisyona dikkat, stop tight tut',
                'priority': 'CRITICAL' if ls_ratio >= 3.0 else 'HIGH',
                'data': {'ls_ratio': ls_ratio}
            }
        elif ls_ratio <= 0.5:
            return {
                'type': 'LONG_SHORT_EXTREME',
                'title': '🔥 AŞIRI SHORT POZİSYON',
                'message': f'Long/Short: {ls_ratio:.2f} - Piyasa aşırı bearish. Short squeeze potansiyeli!',
                'action': 'Agresif short riskli, hedge düşün',
                'priority': 'HIGH',
                'data': {'ls_ratio': ls_ratio}
            }
        
        return None
    
    @staticmethod
    def _check_oi_anomaly(derivatives: Dict) -> Optional[Dict]:
        """
        Open Interest anormal değişimler tespit et.
        
        OI急 artış + Fiyat düşüş → Shortlar birikiyor
        OI急 artış + Fiyat artış → Longlar birikiyor
        OI急 düşüş → Pozisyon kapanışları / Likidasyon
        """
        oi_change_pct = derivatives.get('oi_change_pct', 0)
        
        if not oi_change_pct:
            return None
        
        if oi_change_pct >= 10:  # 10%+ OI artışı
            return {
                'type': 'OI_ANOMALY',
                'title': '📈 OI ANİ ARTIŞ',
                'message': f'Open Interest %{oi_change_pct:.1f} arttı - Yeni pozisyon girişi. Volatilite artabilir!',
                'action': 'Büyük hareket bekleniyor, pozisyon boyutunu küçült',
                'priority': 'HIGH' if oi_change_pct >= 20 else 'MEDIUM',
                'data': {'oi_change': oi_change_pct}
            }
        elif oi_change_pct <= -10:  # 10%+ OI düşüşü
            return {
                'type': 'OI_ANOMALY',
                'title': '📉 OI SERT DÜŞÜŞ',
                'message': f'Open Interest %{abs(oi_change_pct):.1f} düştü - Pozisyon kapanışları veya likidasyon!',
                'action': 'Fiyat stabilize olana kadar bekle',
                'priority': 'HIGH',
                'data': {'oi_change': oi_change_pct}
            }
        
        return None
    
    @staticmethod
    def _check_fear_greed_extreme(sentiment: Dict) -> Optional[Dict]:
        """
        Fear & Greed Index extreme değerlerde uyar.
        
        0-20: Extreme Fear → Alım fırsatı olabilir
        80-100: Extreme Greed → Düzeltme riski
        """
        fg_index = sentiment.get('fear_greed_index', 50)
        fg_label = sentiment.get('fear_greed_label', '')
        
        if not fg_index or fg_index == 50:
            return None
        
        if fg_index <= 20:
            return {
                'type': 'FEAR_GREED_EXTREME',
                'title': '😱 EXTREME FEAR',
                'message': f'Fear & Greed: {fg_index}/100 ({fg_label}) - Piyasa panik modunda!',
                'action': 'Tarihsel olarak alım fırsatı, ama dikkatli ol',
                'priority': 'HIGH',
                'data': {'fear_greed': fg_index, 'label': fg_label}
            }
        elif fg_index >= 80:
            return {
                'type': 'FEAR_GREED_EXTREME',
                'title': '🤑 EXTREME GREED',
                'message': f'Fear & Greed: {fg_index}/100 ({fg_label}) - Piyasa aşırı açgözlü!',
                'action': 'Düzeltme riski yüksek, kar al veya stop sıkılaştır',
                'priority': 'CRITICAL' if fg_index >= 90 else 'HIGH',
                'data': {'fear_greed': fg_index, 'label': fg_label}
            }
        
        return None
