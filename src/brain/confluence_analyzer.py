"""
DEMIR AI - Advanced Confluence Analyzer
Çoklu sinyal uyumunu analiz ederek güçlü giriş noktaları tespit eder.

PHASE 41: Signal Confluence System
1. Multi-Timeframe Confluence (MTF)
2. Volatility Compression Detector
3. Session Analysis (Asia/London/NY)
4. Exchange Flow Analysis
5. Unified Confluence Score
"""
import logging
import requests
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("CONFLUENCE_ANALYZER")


@dataclass
class ConfluenceSignal:
    """Confluence analiz sonucu"""
    symbol: str
    confluence_score: int       # 0-10
    direction: str              # BULLISH, BEARISH, NEUTRAL
    confidence: float           # 0-100
    mtf_agreement: Dict         # {1h: BUY, 4h: BUY, 1d: SELL}
    volatility_state: str       # COMPRESSED, NORMAL, EXPANDED
    session: str                # ASIA, LONDON, NY, OVERLAP
    exchange_flow: str          # INFLOW, OUTFLOW, NEUTRAL
    factors: List[str]          # Contributing factors
    action: str                 # Recommended action
    timestamp: datetime


class ConfluenceAnalyzer:
    """
    Confluence Analizörü
    
    Birden fazla sinyal kaynağını birleştirerek
    yüksek olasılıklı giriş noktaları tespit eder.
    """
    
    # Confluence scoring weights
    WEIGHTS = {
        'mtf_3_agree': 3,        # 3 timeframe aynı yön
        'mtf_2_agree': 2,        # 2 timeframe aynı yön
        'volatility_compressed': 2,  # Sıkışma = patlama yakın
        'session_optimal': 1,    # London/NY open
        'exchange_outflow': 1,   # HODL = bullish
        'exchange_inflow': -1,   # Selling = bearish
        'volume_spike': 1,       # Hacim artışı
        'pattern_present': 1,    # Teknik pattern
    }
    
    # Session times (UTC)
    SESSIONS = {
        'ASIA': (0, 8),          # 00:00 - 08:00 UTC
        'LONDON': (8, 16),       # 08:00 - 16:00 UTC
        'NY': (14, 22),          # 14:00 - 22:00 UTC (overlap 14-16)
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 60  # 1 minute
        
    # =========================================
    # 1. MULTI-TIMEFRAME CONFLUENCE
    # =========================================
    def analyze_mtf(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Çoklu zaman dilimi analizi.
        
        1h, 4h, 1D trend yönlerini karşılaştırır.
        """
        timeframes = ['1h', '4h', '1d']
        signals = {}
        
        for tf in timeframes:
            try:
                signal = self._get_tf_signal(symbol, tf)
                signals[tf] = signal
            except Exception as e:
                logger.debug(f"MTF {tf} error: {e}")
                signals[tf] = {'direction': 'NEUTRAL', 'strength': 0}
        
        # Calculate agreement
        directions = [s['direction'] for s in signals.values()]
        bullish_count = directions.count('BULLISH')
        bearish_count = directions.count('BEARISH')
        
        if bullish_count == 3:
            agreement = 'ALL_BULLISH'
            direction = 'BULLISH'
            score = 3
        elif bearish_count == 3:
            agreement = 'ALL_BEARISH'
            direction = 'BEARISH'
            score = 3
        elif bullish_count >= 2:
            agreement = 'MOSTLY_BULLISH'
            direction = 'BULLISH'
            score = 2
        elif bearish_count >= 2:
            agreement = 'MOSTLY_BEARISH'
            direction = 'BEARISH'
            score = 2
        else:
            agreement = 'MIXED'
            direction = 'NEUTRAL'
            score = 0
        
        return {
            'signals': signals,
            'agreement': agreement,
            'direction': direction,
            'score': score,
            'description': self._get_mtf_description(signals, agreement)
        }
    
    def _get_tf_signal(self, symbol: str, interval: str) -> Dict:
        """Tek timeframe için sinyal hesapla"""
        try:
            # Fetch klines
            url = f"https://api.binance.com/api/v3/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': 50}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                klines = response.json()
                closes = [float(k[4]) for k in klines]
                
                # Calculate EMAs
                ema_9 = self._ema(closes, 9)
                ema_21 = self._ema(closes, 21)
                
                current_price = closes[-1]
                
                # Determine direction
                if ema_9 > ema_21 and current_price > ema_9:
                    direction = 'BULLISH'
                    strength = (current_price - ema_21) / ema_21 * 100
                elif ema_9 < ema_21 and current_price < ema_9:
                    direction = 'BEARISH'
                    strength = (ema_21 - current_price) / ema_21 * 100
                else:
                    direction = 'NEUTRAL'
                    strength = 0
                
                return {'direction': direction, 'strength': abs(strength)}
                
        except Exception as e:
            logger.debug(f"TF signal error: {e}")
        
        return {'direction': 'NEUTRAL', 'strength': 0}
    
    def _ema(self, data: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(data) < period:
            return np.mean(data)
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def _get_mtf_description(self, signals: Dict, agreement: str) -> str:
        """MTF Türkçe açıklama"""
        desc = "📊 MTF Analizi: "
        for tf, sig in signals.items():
            emoji = "🟢" if sig['direction'] == 'BULLISH' else "🔴" if sig['direction'] == 'BEARISH' else "⚪"
            desc += f"{tf}={emoji} "
        
        if 'ALL' in agreement:
            desc += "| 🎯 TÜM ZAMAN DİLİMLERİ UYUMLU!"
        elif 'MOSTLY' in agreement:
            desc += "| ✅ Çoğunluk uyumlu"
        else:
            desc += "| ⚠️ Karışık sinyaller - dikkatli ol!"
        
        return desc
    
    # =========================================
    # 2. VOLATILITY COMPRESSION DETECTOR
    # =========================================
    def detect_volatility_compression(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Volatilite sıkışması tespiti.
        
        Bollinger bantları daralırsa büyük hareket yakın demektir.
        """
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {'symbol': symbol, 'interval': '4h', 'limit': 50}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                klines = response.json()
                closes = np.array([float(k[4]) for k in klines])
                
                # Bollinger Bands
                sma_20 = np.mean(closes[-20:])
                std_20 = np.std(closes[-20:])
                bb_width = (std_20 * 2) / sma_20 * 100  # Percentage width
                
                # Historical average width
                hist_widths = []
                for i in range(20, len(closes) - 1):
                    hist_std = np.std(closes[i-20:i])
                    hist_sma = np.mean(closes[i-20:i])
                    hist_widths.append((hist_std * 2) / hist_sma * 100)
                
                avg_width = np.mean(hist_widths) if hist_widths else bb_width
                
                # Determine state
                if bb_width < avg_width * 0.6:
                    state = 'COMPRESSED'
                    action = "💥 Volatilite sıkışması! Büyük hareket yakın - breakout bekle!"
                    score = 2
                elif bb_width > avg_width * 1.4:
                    state = 'EXPANDED'
                    action = "📈 Yüksek volatilite - trend devam veya geri çekilme olabilir"
                    score = 0
                else:
                    state = 'NORMAL'
                    action = "Volatilite normal seviyede"
                    score = 0
                
                return {
                    'state': state,
                    'bb_width': bb_width,
                    'avg_width': avg_width,
                    'ratio': bb_width / avg_width if avg_width > 0 else 1,
                    'score': score,
                    'action': action
                }
                
        except Exception as e:
            logger.debug(f"Volatility detection error: {e}")
        
        return {
            'state': 'NORMAL',
            'bb_width': 0,
            'avg_width': 0,
            'ratio': 1,
            'score': 0,
            'action': 'Veri alınamadı'
        }
    
    # =========================================
    # 3. SESSION ANALYSIS
    # =========================================
    def analyze_session(self) -> Dict:
        """
        Trading session analizi.
        
        Asia: Düşük hacim, range-bound
        London: Trend başlangıcı, yüksek hacim
        NY: En yüksek volatilite
        Overlap: London-NY overlap = en iyi zaman
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        # Determine current session
        in_asia = self.SESSIONS['ASIA'][0] <= hour < self.SESSIONS['ASIA'][1]
        in_london = self.SESSIONS['LONDON'][0] <= hour < self.SESSIONS['LONDON'][1]
        in_ny = self.SESSIONS['NY'][0] <= hour < self.SESSIONS['NY'][1]
        
        # Check for overlap
        if in_london and in_ny:
            session = 'LONDON_NY_OVERLAP'
            volatility = 'HIGHEST'
            action = "🔥 London-NY overlap! En yüksek volatilite ve hacim. Breakout'lar için ideal!"
            score = 2
        elif in_ny:
            session = 'NEW_YORK'
            volatility = 'HIGH'
            action = "🇺🇸 NY seansı aktif. Yüksek volatilite ve hacim beklenir."
            score = 1
        elif in_london:
            session = 'LONDON'
            volatility = 'HIGH'
            action = "🇬🇧 London seansı aktif. Trendler başlayabilir!"
            score = 1
        elif in_asia:
            session = 'ASIA'
            volatility = 'LOW'
            action = "🌏 Asya seansı. Düşük hacim - range trading olabilir."
            score = 0
        else:
            session = 'TRANSITION'
            volatility = 'MEDIUM'
            action = "Seans geçişi. Yeni trend için hazırlık yapılabilir."
            score = 0
        
        # Day of week analysis
        day = now.weekday()
        day_names = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
        
        if day == 4:  # Friday
            day_note = "⚠️ Cuma - Hafta sonu öncesi pozisyon kapatmaları olabilir"
        elif day in [5, 6]:  # Weekend
            day_note = "📉 Hafta sonu - Düşük hacim ve CME gap riski"
        elif day == 0:  # Monday
            day_note = "📊 Pazartesi - CME gap check et, yeni hafta trendi başlayabilir"
        else:
            day_note = f"{day_names[day]} - Normal işlem günü"
        
        return {
            'session': session,
            'volatility': volatility,
            'hour_utc': hour,
            'day': day_names[day],
            'score': score,
            'action': action,
            'day_note': day_note
        }
    
    # =========================================
    # 4. EXCHANGE FLOW ANALYSIS
    # =========================================
    def analyze_exchange_flow(self, symbol: str = 'BTC') -> Dict:
        """
        Borsa giriş/çıkış analizi.
        
        Borsaya giriş = Satış hazırlığı (bearish)
        Borsadan çıkış = HODL (bullish)
        
        Not: CryptoQuant/Glassnode API gerekir, burada simüle ediyoruz.
        """
        try:
            # Simulated based on volume patterns
            # In production, would use CryptoQuant API
            
            url = f"https://api.binance.com/api/v3/klines"
            params = {'symbol': f'{symbol}USDT', 'interval': '1h', 'limit': 24}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                klines = response.json()
                
                # Use taker buy/sell ratio as proxy
                taker_buy_volume = sum([float(k[9]) for k in klines])  # Taker buy base
                total_volume = sum([float(k[5]) for k in klines])  # Total volume
                
                if total_volume > 0:
                    buy_ratio = taker_buy_volume / total_volume
                    
                    if buy_ratio > 0.55:
                        flow = 'OUTFLOW'  # More buying = accumulation
                        direction = 'BULLISH'
                        action = "🟢 Borsadan çıkış baskın - Birikim (HODL) devam ediyor!"
                        score = 1
                    elif buy_ratio < 0.45:
                        flow = 'INFLOW'  # More selling = distribution
                        direction = 'BEARISH'
                        action = "🔴 Borsaya giriş baskın - Satış baskısı olabilir!"
                        score = -1
                    else:
                        flow = 'NEUTRAL'
                        direction = 'NEUTRAL'
                        action = "Borsa akışı dengeli"
                        score = 0
                    
                    return {
                        'flow': flow,
                        'direction': direction,
                        'buy_ratio': buy_ratio,
                        'score': score,
                        'action': action
                    }
                    
        except Exception as e:
            logger.debug(f"Exchange flow error: {e}")
        
        return {
            'flow': 'NEUTRAL',
            'direction': 'NEUTRAL',
            'buy_ratio': 0.5,
            'score': 0,
            'action': 'Veri alınamadı'
        }
    
    # =========================================
    # 5. UNIFIED CONFLUENCE SCORE
    # =========================================
    def calculate_confluence(self, symbol: str = 'BTCUSDT') -> ConfluenceSignal:
        """
        Tüm sinyalleri birleştirip confluence score hesapla.
        
        Score: 0-10
        - 8-10: 🟢🟢🟢 MÜKEMMEL CONFLUENCE
        - 6-7:  🟢🟢 GÜÇLÜ CONFLUENCE
        - 4-5:  🟢 ORTA CONFLUENCE
        - 0-3:  ⚪ ZAYIF CONFLUENCE
        """
        factors = []
        total_score = 0
        
        # 1. MTF Analysis
        mtf = self.analyze_mtf(symbol)
        total_score += mtf['score']
        if mtf['score'] >= 2:
            factors.append(f"MTF {mtf['agreement']}")
        
        # 2. Volatility
        vol = self.detect_volatility_compression(symbol)
        total_score += vol['score']
        if vol['score'] > 0:
            factors.append(f"Volatilite {vol['state']}")
        
        # 3. Session
        session = self.analyze_session()
        total_score += session['score']
        if session['score'] > 0:
            factors.append(f"Session {session['session']}")
        
        # 4. Exchange Flow
        base_symbol = symbol.replace('USDT', '')
        flow = self.analyze_exchange_flow(base_symbol)
        total_score += flow['score']
        if flow['score'] != 0:
            factors.append(f"Exchange {flow['flow']}")
        
        # Normalize score to 0-10
        # Max possible: MTF(3) + Vol(2) + Session(2) + Flow(1) = 8
        # Min possible: Flow(-1) = -1
        normalized_score = max(0, min(10, total_score + 2))  # Shift to 0-10 range
        
        # Determine overall direction
        directions = [mtf['direction'], flow['direction']]
        bullish = directions.count('BULLISH')
        bearish = directions.count('BEARISH')
        
        if bullish > bearish:
            direction = 'BULLISH'
        elif bearish > bullish:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        # Calculate confidence
        confidence = normalized_score * 10  # 0-100
        
        # Generate action
        if normalized_score >= 8:
            action = f"🟢🟢🟢 MÜKEMMEL CONFLUENCE! {direction} yönünde güçlü giriş noktası!"
        elif normalized_score >= 6:
            action = f"🟢🟢 GÜÇLÜ CONFLUENCE! {direction} sinyali güçlü."
        elif normalized_score >= 4:
            action = f"🟢 ORTA CONFLUENCE. {direction} sinyali var ama dikkatli ol."
        elif normalized_score >= 2:
            action = f"⚪ ZAYIF CONFLUENCE. Sinyaller karışık - bekle."
        else:
            action = "❌ CONFLUENCE YOK. İşlem için uygun değil."
        
        return ConfluenceSignal(
            symbol=symbol,
            confluence_score=normalized_score,
            direction=direction,
            confidence=confidence,
            mtf_agreement=mtf['signals'],
            volatility_state=vol['state'],
            session=session['session'],
            exchange_flow=flow['flow'],
            factors=factors,
            action=action,
            timestamp=datetime.now()
        )
    
    # =========================================
    # FORMAT FOR TELEGRAM
    # =========================================
    def format_for_telegram(self, signal: ConfluenceSignal) -> str:
        """Telegram için formatla"""
        
        # Score visualization
        score_bar = "█" * signal.confluence_score + "░" * (10 - signal.confluence_score)
        
        # Direction emoji
        dir_emoji = "🟢" if signal.direction == 'BULLISH' else "🔴" if signal.direction == 'BEARISH' else "⚪"
        
        msg = f"🎯 *CONFLUENCE ANALİZİ - {signal.symbol}*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        msg += f"📊 *Confluence Score:* [{score_bar}] {signal.confluence_score}/10\n"
        msg += f"{dir_emoji} *Yön:* {signal.direction} (güven: {signal.confidence:.0f}%)\n\n"
        
        # MTF
        msg += "📈 *Multi-Timeframe:*\n"
        for tf, sig in signal.mtf_agreement.items():
            emoji = "🟢" if sig['direction'] == 'BULLISH' else "🔴" if sig['direction'] == 'BEARISH' else "⚪"
            msg += f"  • {tf}: {emoji} {sig['direction']}\n"
        
        # Volatility
        vol_emoji = "💥" if signal.volatility_state == 'COMPRESSED' else "📊"
        msg += f"\n{vol_emoji} *Volatilite:* {signal.volatility_state}\n"
        
        # Session
        msg += f"🕐 *Session:* {signal.session}\n"
        
        # Exchange Flow
        flow_emoji = "🟢" if signal.exchange_flow == 'OUTFLOW' else "🔴" if signal.exchange_flow == 'INFLOW' else "⚪"
        msg += f"{flow_emoji} *Exchange Flow:* {signal.exchange_flow}\n"
        
        # Factors
        if signal.factors:
            msg += f"\n✅ *Aktif Faktörler:* {', '.join(signal.factors)}\n"
        
        # Action
        msg += f"\n💡 *{signal.action}*\n"
        msg += f"\n⏰ _{signal.timestamp.strftime('%H:%M:%S')}_"
        
        return msg
    
    def analyze_all_coins(self, coins: List[str] = None) -> Dict[str, ConfluenceSignal]:
        """Tüm coinler için confluence analizi"""
        if coins is None:
            coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT']
        
        results = {}
        for coin in coins:
            try:
                results[coin] = self.calculate_confluence(coin)
            except Exception as e:
                logger.warning(f"Confluence analysis failed for {coin}: {e}")
        
        return results
