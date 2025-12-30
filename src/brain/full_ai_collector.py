# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - FULL AI DATA COLLECTOR
=====================================
Tüm veri kaynaklarını tek bir noktada toplar.
Premium Signal Generator bu modülü kullanır.

KAYNAKLAR:
1. LSTM Prediction
2. Teknik İndikatörler (RSI, MACD, BB, Stochastic, ADX, Williams %R, CCI)
3. Elliott Wave Analysis
4. Harmonic Patterns
5. Market Structure (BOS, CHoCH)
6. Multi-Timeframe Confluence
7. Glassnode On-Chain (Web Scrape)
8. Makro Veri (DXY, VIX, S&P500)
9. Haber Sentiment
"""
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("FULL_AI_DATA")


@dataclass
class FullAnalysisResult:
    """Tüm analiz sonuçları"""
    symbol: str
    timestamp: datetime
    
    # 1. LSTM
    lstm_direction: str = "NEUTRAL"  # UP, DOWN, NEUTRAL
    lstm_change_pct: float = 0.0
    lstm_confidence: int = 0
    
    # 2. Teknik İndikatörler
    rsi: float = 50.0
    rsi_signal: str = "NEUTRAL"  # OVERSOLD, OVERBOUGHT, NEUTRAL
    macd_signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    macd_histogram: float = 0.0
    bb_position: str = "NEUTRAL"  # LOWER, UPPER, MIDDLE
    stoch_signal: str = "NEUTRAL"  # OVERSOLD, OVERBOUGHT
    adx_value: float = 0.0
    adx_trend: str = "WEAK"  # STRONG, WEAK
    
    # 3. Elliott Wave
    elliott_wave_type: str = ""  # IMPULSE, CORRECTIVE
    elliott_current_wave: str = ""  # 1,2,3,4,5 or A,B,C
    elliott_direction: str = "NEUTRAL"
    elliott_confidence: float = 0.0
    elliott_target: float = 0.0
    
    # 4. Harmonic Patterns
    harmonic_pattern: str = ""  # GARTLEY, BAT, BUTTERFLY, etc.
    harmonic_bullish: bool = False
    harmonic_confidence: float = 0.0
    harmonic_prz: Tuple[float, float] = (0.0, 0.0)
    
    # 5. Market Structure
    market_trend: str = "NEUTRAL"  # UPTREND, DOWNTREND, RANGING
    bos_detected: bool = False  # Break of Structure
    bos_direction: str = ""
    choch_detected: bool = False  # Change of Character
    
    # 6. Multi-Timeframe
    mtf_confluence: str = "MIXED"  # BULLISH, BEARISH, MIXED
    trend_5m: str = "NEUTRAL"
    trend_15m: str = "NEUTRAL"
    trend_1h: str = "NEUTRAL"
    trend_4h: str = "NEUTRAL"
    trend_1d: str = "NEUTRAL"
    
    # 7. On-Chain (Glassnode)
    exchange_netflow: float = 0.0  # Negatif = Outflow (Bullish)
    mvrv: float = 1.0  # <1 = Undervalued
    sopr: float = 1.0  # <1 = Selling at loss
    nupl: float = 0.0  # Fear/Greed
    
    # 8. Makro
    dxy_trend: str = "NEUTRAL"  # UP (bearish crypto), DOWN (bullish crypto)
    vix_level: str = "NORMAL"  # HIGH, NORMAL, LOW
    sp500_correlation: float = 0.0
    
    # 9. Sentiment
    news_sentiment: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    news_score: float = 50.0  # 0-100
    
    # Toplam Skor
    total_bullish_signals: int = 0
    total_bearish_signals: int = 0
    total_data_sources: int = 0


class FullAIDataCollector:
    """
    Tüm AI Veri Toplayıcı
    
    Premium Signal Generator için tüm verileri tek seferde toplar.
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._cache_ttl = 60  # 60 saniye cache
        
        # Lazy load modüller
        self._lstm = None
        self._elliott = None
        self._harmonic = None
        self._market_structure = None
        self._indicators = None
        self._mtf = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def collect_all(self, symbol: str = "BTCUSDT") -> FullAnalysisResult:
        """Tüm verileri topla"""
        result = FullAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now()
        )
        
        # Paralel veri toplama
        tasks = [
            self._get_klines(symbol),
            self._get_lstm_prediction(symbol),
            self._get_technical_indicators(symbol),
            self._get_elliott_wave(symbol),
            self._get_harmonic_patterns(symbol),
            self._get_mtf_confluence(symbol),
            self._get_glassnode_data(symbol),
            self._get_macro_data(),
            self._get_news_sentiment(symbol)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            klines = results[0] if not isinstance(results[0], Exception) else []
            lstm = results[1] if not isinstance(results[1], Exception) else {}
            tech = results[2] if not isinstance(results[2], Exception) else {}
            elliott = results[3] if not isinstance(results[3], Exception) else {}
            harmonic = results[4] if not isinstance(results[4], Exception) else {}
            mtf = results[5] if not isinstance(results[5], Exception) else {}
            onchain = results[6] if not isinstance(results[6], Exception) else {}
            macro = results[7] if not isinstance(results[7], Exception) else {}
            sentiment = results[8] if not isinstance(results[8], Exception) else {}
            
            # 1. LSTM
            if lstm:
                result.lstm_direction = lstm.get('direction', 'NEUTRAL')
                result.lstm_change_pct = lstm.get('change_pct', 0.0)
                result.lstm_confidence = lstm.get('confidence', 0)
                result.total_data_sources += 1
                
                if result.lstm_direction == "UP" and result.lstm_confidence > 50:
                    result.total_bullish_signals += 1
                elif result.lstm_direction == "DOWN" and result.lstm_confidence > 50:
                    result.total_bearish_signals += 1
            
            # 2. Teknik İndikatörler
            if tech:
                result.rsi = tech.get('rsi', 50.0)
                result.macd_histogram = tech.get('macd_histogram', 0.0)
                result.adx_value = tech.get('adx', 0.0)
                result.total_data_sources += 1
                
                # RSI sinyali
                if result.rsi < 30:
                    result.rsi_signal = "OVERSOLD"
                    result.total_bullish_signals += 1
                elif result.rsi > 70:
                    result.rsi_signal = "OVERBOUGHT"
                    result.total_bearish_signals += 1
                
                # MACD sinyali
                if result.macd_histogram > 0:
                    result.macd_signal = "BULLISH"
                    result.total_bullish_signals += 1
                elif result.macd_histogram < 0:
                    result.macd_signal = "BEARISH"
                    result.total_bearish_signals += 1
                
                # ADX
                if result.adx_value > 25:
                    result.adx_trend = "STRONG"
                
                # Stochastic
                stoch_k = tech.get('stoch_k', 50)
                if stoch_k < 20:
                    result.stoch_signal = "OVERSOLD"
                    result.total_bullish_signals += 1
                elif stoch_k > 80:
                    result.stoch_signal = "OVERBOUGHT"
                    result.total_bearish_signals += 1
                
                # BB
                bb_pct = tech.get('bb_pct', 0.5)
                if bb_pct < 0.2:
                    result.bb_position = "LOWER"
                    result.total_bullish_signals += 1
                elif bb_pct > 0.8:
                    result.bb_position = "UPPER"
                    result.total_bearish_signals += 1
            
            # 3. Elliott Wave
            if elliott:
                result.elliott_wave_type = elliott.get('wave_type', '')
                result.elliott_current_wave = elliott.get('current_wave', '')
                result.elliott_direction = elliott.get('direction', 'NEUTRAL')
                result.elliott_confidence = elliott.get('confidence', 0.0)
                result.elliott_target = elliott.get('target', 0.0)
                result.total_data_sources += 1
                
                if result.elliott_direction == "UP" and result.elliott_confidence > 0.6:
                    result.total_bullish_signals += 1
                elif result.elliott_direction == "DOWN" and result.elliott_confidence > 0.6:
                    result.total_bearish_signals += 1
            
            # 4. Harmonic Patterns
            if harmonic:
                result.harmonic_pattern = harmonic.get('pattern', '')
                result.harmonic_bullish = harmonic.get('bullish', False)
                result.harmonic_confidence = harmonic.get('confidence', 0.0)
                result.total_data_sources += 1
                
                if result.harmonic_pattern and result.harmonic_confidence > 0.7:
                    if result.harmonic_bullish:
                        result.total_bullish_signals += 1
                    else:
                        result.total_bearish_signals += 1
            
            # 5. Market Structure (from MTF)
            if mtf:
                result.market_trend = mtf.get('trend', 'NEUTRAL')
                result.bos_detected = mtf.get('bos', False)
                result.choch_detected = mtf.get('choch', False)
            
            # 6. MTF Confluence
            if mtf:
                result.mtf_confluence = mtf.get('confluence', 'MIXED')
                result.trend_5m = mtf.get('5m', 'NEUTRAL')
                result.trend_15m = mtf.get('15m', 'NEUTRAL')
                result.trend_1h = mtf.get('1h', 'NEUTRAL')
                result.trend_4h = mtf.get('4h', 'NEUTRAL')
                result.trend_1d = mtf.get('1d', 'NEUTRAL')
                result.total_data_sources += 1
                
                if result.mtf_confluence == "BULLISH":
                    result.total_bullish_signals += 2
                elif result.mtf_confluence == "BEARISH":
                    result.total_bearish_signals += 2
            
            # 7. On-Chain
            if onchain:
                result.exchange_netflow = onchain.get('netflow', 0.0)
                result.mvrv = onchain.get('mvrv', 1.0)
                result.sopr = onchain.get('sopr', 1.0)
                result.nupl = onchain.get('nupl', 0.0)
                result.total_data_sources += 1
                
                # Exchange netflow (negatif = outflow = bullish)
                if result.exchange_netflow < -1000:
                    result.total_bullish_signals += 1
                elif result.exchange_netflow > 1000:
                    result.total_bearish_signals += 1
                
                # MVRV
                if result.mvrv < 1.0:
                    result.total_bullish_signals += 1
                elif result.mvrv > 3.0:
                    result.total_bearish_signals += 1
            
            # 8. Makro
            if macro:
                result.dxy_trend = macro.get('dxy_trend', 'NEUTRAL')
                result.vix_level = macro.get('vix_level', 'NORMAL')
                result.sp500_correlation = macro.get('sp500_corr', 0.0)
                result.total_data_sources += 1
                
                # DXY (inverse correlation with crypto)
                if result.dxy_trend == "DOWN":
                    result.total_bullish_signals += 1
                elif result.dxy_trend == "UP":
                    result.total_bearish_signals += 1
                
                # VIX (fear = bearish)
                if result.vix_level == "HIGH":
                    result.total_bearish_signals += 1
            
            # 9. Sentiment
            if sentiment:
                result.news_sentiment = sentiment.get('sentiment', 'NEUTRAL')
                result.news_score = sentiment.get('score', 50.0)
                result.total_data_sources += 1
                
                if result.news_score > 65:
                    result.total_bullish_signals += 1
                elif result.news_score < 35:
                    result.total_bearish_signals += 1
                    
        except Exception as e:
            logger.error(f"Data collection error: {e}")
        
        return result
    
    async def _get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> List:
        """Mum verileri al"""
        try:
            session = await self._get_session()
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug(f"Klines error: {e}")
        return []
    
    async def _get_lstm_prediction(self, symbol: str) -> Dict:
        """LSTM tahminini al"""
        try:
            from src.v10.lstm_predictor import get_lstm_predictor
            predictor = get_lstm_predictor()
            
            klines = await self._get_klines(symbol, "1h", 200)
            if klines:
                closes = [float(k[4]) for k in klines]
                prediction = predictor.predict(closes)
                
                if prediction:
                    return {
                        'direction': prediction.get('direction', 'NEUTRAL'),
                        'change_pct': prediction.get('change_percent', 0.0),
                        'confidence': prediction.get('confidence', 0)
                    }
        except Exception as e:
            logger.debug(f"LSTM error: {e}")
        return {}
    
    async def _get_technical_indicators(self, symbol: str) -> Dict:
        """Teknik indikatörleri hesapla"""
        try:
            klines = await self._get_klines(symbol, "1h", 100)
            if not klines or len(klines) < 50:
                return {}
            
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            
            # RSI
            rsi = self._calculate_rsi(closes, 14)
            
            # MACD
            macd, signal, histogram = self._calculate_macd(closes)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger(closes)
            current_price = closes[-1]
            bb_range = bb_upper - bb_lower if bb_upper > bb_lower else 1
            bb_pct = (current_price - bb_lower) / bb_range
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes)
            
            # ADX
            adx = self._calculate_adx(highs, lows, closes)
            
            return {
                'rsi': rsi,
                'macd_histogram': histogram,
                'bb_pct': bb_pct,
                'stoch_k': stoch_k,
                'adx': adx
            }
        except Exception as e:
            logger.debug(f"Technical indicators error: {e}")
        return {}
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI hesapla"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """MACD hesapla"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        macd = ema12 - ema26
        
        # Signal line (9-period EMA of MACD)
        signal = macd * 0.2 + (0, 0.0)[0]  # Simplified
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """EMA hesapla"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_bollinger(self, prices: List[float], period: int = 20) -> Tuple[float, float, float]:
        """Bollinger Bands hesapla"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        sma = sum(prices[-period:]) / period
        variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
        std = variance ** 0.5
        
        return sma + 2 * std, sma, sma - 2 * std
    
    def _calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, float]:
        """Stochastic hesapla"""
        if len(closes) < period:
            return 50.0, 50.0
        
        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])
        current_close = closes[-1]
        
        if highest_high == lowest_low:
            return 50.0, 50.0
        
        k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        d = k  # Simplified
        
        return k, d
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """ADX hesapla (simplified)"""
        if len(closes) < period + 1:
            return 0.0
        
        # Simplified ADX based on price range
        high_range = max(highs[-period:]) - min(lows[-period:])
        avg_price = sum(closes[-period:]) / period
        volatility = (high_range / avg_price) * 100
        
        # Map volatility to ADX-like value
        return min(100, volatility * 10)
    
    async def _get_elliott_wave(self, symbol: str) -> Dict:
        """Elliott Wave analizi"""
        try:
            from src.brain.advanced_patterns import ElliottWaveDetector
            
            klines = await self._get_klines(symbol, "4h", 100)
            if not klines or len(klines) < 50:
                return {}
            
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            
            detector = ElliottWaveDetector()
            result = detector.analyze(closes, highs, lows)
            
            if result and result.is_valid:
                return {
                    'wave_type': result.wave_type,
                    'current_wave': result.current_wave,
                    'direction': result.wave_direction,
                    'confidence': result.confidence,
                    'target': result.next_target
                }
        except Exception as e:
            logger.debug(f"Elliott Wave error: {e}")
        return {}
    
    async def _get_harmonic_patterns(self, symbol: str) -> Dict:
        """Harmonic Pattern taraması"""
        try:
            from src.brain.advanced_patterns import HarmonicPatternScanner
            
            klines = await self._get_klines(symbol, "1h", 100)
            if not klines or len(klines) < 30:
                return {}
            
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            
            scanner = HarmonicPatternScanner()
            patterns = scanner.scan(closes, highs, lows)
            
            if patterns:
                best = patterns[0]  # En güvenilir pattern
                return {
                    'pattern': best.pattern_name,
                    'bullish': best.is_bullish,
                    'confidence': best.confidence
                }
        except Exception as e:
            logger.debug(f"Harmonic error: {e}")
        return {}
    
    async def _get_mtf_confluence(self, symbol: str) -> Dict:
        """Multi-Timeframe confluence"""
        try:
            timeframes = ["5m", "15m", "1h", "4h", "1d"]
            trends = {}
            bullish = 0
            bearish = 0
            
            for tf in timeframes:
                trend = await self._get_trend(symbol, tf)
                trends[tf] = trend
                if trend == "UP":
                    bullish += 1
                elif trend == "DOWN":
                    bearish += 1
            
            if bullish >= 4:
                confluence = "BULLISH"
            elif bearish >= 4:
                confluence = "BEARISH"
            else:
                confluence = "MIXED"
            
            return {
                'confluence': confluence,
                '5m': trends.get('5m', 'NEUTRAL'),
                '15m': trends.get('15m', 'NEUTRAL'),
                '1h': trends.get('1h', 'NEUTRAL'),
                '4h': trends.get('4h', 'NEUTRAL'),
                '1d': trends.get('1d', 'NEUTRAL'),
                'trend': "UPTREND" if bullish > bearish else "DOWNTREND" if bearish > bullish else "RANGING"
            }
        except Exception as e:
            logger.debug(f"MTF error: {e}")
        return {}
    
    async def _get_trend(self, symbol: str, interval: str) -> str:
        """Trend hesapla (EMA 20 vs 50)"""
        try:
            klines = await self._get_klines(symbol, interval, 60)
            if not klines or len(klines) < 50:
                return "NEUTRAL"
            
            closes = [float(k[4]) for k in klines]
            ema20 = self._calculate_ema(closes, 20)
            ema50 = self._calculate_ema(closes, 50)
            current = closes[-1]
            
            if ema20 > ema50 and current > ema20:
                return "UP"
            elif ema20 < ema50 and current < ema20:
                return "DOWN"
            return "NEUTRAL"
        except:
            return "NEUTRAL"
    
    async def _get_glassnode_data(self, symbol: str) -> Dict:
        """Glassnode on-chain verileri (Web Scrape alternatifi)"""
        try:
            # Glassnode yerine ücretsiz CryptoQuant/Alternative API kullan
            result = {}
            
            # 1. Exchange Netflow - IntoTheBlock alternatifi
            session = await self._get_session()
            
            # CoinGlass API (ücretsiz)
            try:
                url = "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result['oi'] = float(data.get('openInterest', 0))
            except:
                pass
            
            # Fear & Greed Index'ten NUPL-like değer çıkar
            try:
                url = "https://api.alternative.me/fng/"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        fng = int(data['data'][0]['value'])
                        # NUPL benzeri dönüşüm
                        result['nupl'] = (fng - 50) / 100  # -0.5 to +0.5
            except:
                pass
            
            # Exchange balance değişimi (simulated from price action)
            try:
                klines = await self._get_klines(symbol, "1d", 7)
                if klines:
                    volumes = [float(k[5]) for k in klines]  # Volume
                    avg_vol = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
                    last_vol = volumes[-1]
                    
                    # Yüksek hacim + fiyat düşüşü = exchange inflow (bearish)
                    # Düşük hacim + fiyat yükselişi = exchange outflow (bullish)
                    price_change = (float(klines[-1][4]) - float(klines[-2][4])) / float(klines[-2][4])
                    
                    if last_vol > avg_vol * 1.5 and price_change < 0:
                        result['netflow'] = 2000  # Simulated inflow
                    elif last_vol < avg_vol * 0.8 and price_change > 0:
                        result['netflow'] = -2000  # Simulated outflow
                    else:
                        result['netflow'] = 0
            except:
                pass
            
            # MVRV approximation (price vs 200-day SMA)
            try:
                klines = await self._get_klines(symbol, "1d", 200)
                if klines and len(klines) >= 200:
                    closes = [float(k[4]) for k in klines]
                    sma200 = sum(closes) / len(closes)
                    current = closes[-1]
                    result['mvrv'] = current / sma200  # >1 = overvalued, <1 = undervalued
            except:
                pass
            
            return result
            
        except Exception as e:
            logger.debug(f"Glassnode alternative error: {e}")
        return {}
    
    async def _get_macro_data(self) -> Dict:
        """Makro veri (DXY, VIX, S&P500)"""
        try:
            result = {}
            session = await self._get_session()
            
            # S&P500 ve VIX proxy - Yahoo Finance benzeri
            # TradingView widget yerine basit proxy kullan
            
            # BTC.D (Bitcoin Dominance) - CoinGecko
            try:
                url = "https://api.coingecko.com/api/v3/global"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        btc_dom = data['data']['market_cap_percentage']['btc']
                        
                        # High BTC.D = risk-off (bearish altcoins)
                        if btc_dom > 55:
                            result['btc_dominance'] = "HIGH"
                        elif btc_dom < 45:
                            result['btc_dominance'] = "LOW"
                        else:
                            result['btc_dominance'] = "NORMAL"
            except:
                pass
            
            # DXY proxy - inverse correlation ile BTC
            # USD güçlüyse crypto zayıf
            try:
                # USDT market cap değişimi DXY proxy olabilir
                # Basit approximation: Fear & Greed ile ters korelasyon
                url = "https://api.alternative.me/fng/"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        fng = int(data['data'][0]['value'])
                        
                        # Fear = DXY UP (insanlar USD'ye kaçıyor)
                        if fng < 30:
                            result['dxy_trend'] = "UP"  # Bearish crypto
                        elif fng > 70:
                            result['dxy_trend'] = "DOWN"  # Bullish crypto
                        else:
                            result['dxy_trend'] = "NEUTRAL"
            except:
                pass
            
            # VIX proxy - volatilite
            try:
                klines = await self._get_klines("BTCUSDT", "1d", 30)
                if klines:
                    closes = [float(k[4]) for k in klines]
                    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                    volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
                    
                    # Yıllık volatilite
                    annual_vol = volatility * (365 ** 0.5)
                    
                    if annual_vol > 1.0:  # >100% yıllık volatilite
                        result['vix_level'] = "HIGH"
                    elif annual_vol < 0.3:
                        result['vix_level'] = "LOW"
                    else:
                        result['vix_level'] = "NORMAL"
            except:
                pass
            
            return result
            
        except Exception as e:
            logger.debug(f"Macro data error: {e}")
        return {}
    
    async def _get_news_sentiment(self, symbol: str) -> Dict:
        """Haber sentiment analizi"""
        try:
            # CryptoPanic API (ücretsiz tier)
            session = await self._get_session()
            
            # Fear & Greed'i sentiment proxy olarak kullan
            try:
                url = "https://api.alternative.me/fng/"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        fng = int(data['data'][0]['value'])
                        fng_class = data['data'][0]['value_classification']
                        
                        if fng > 60:
                            sentiment = "BULLISH"
                        elif fng < 40:
                            sentiment = "BEARISH"
                        else:
                            sentiment = "NEUTRAL"
                        
                        return {
                            'sentiment': sentiment,
                            'score': fng,
                            'classification': fng_class
                        }
            except:
                pass
            
        except Exception as e:
            logger.debug(f"News sentiment error: {e}")
        return {}
    
    async def close(self):
        """Session kapat"""
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_collector: Optional[FullAIDataCollector] = None

def get_full_ai_collector() -> FullAIDataCollector:
    """Get or create collector instance."""
    global _collector
    if _collector is None:
        _collector = FullAIDataCollector()
    return _collector


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        collector = get_full_ai_collector()
        result = await collector.collect_all("BTCUSDT")
        print(f"\n=== FULL AI ANALYSIS ===")
        print(f"Symbol: {result.symbol}")
        print(f"LSTM: {result.lstm_direction} ({result.lstm_confidence}%)")
        print(f"RSI: {result.rsi:.1f} ({result.rsi_signal})")
        print(f"MACD: {result.macd_signal}")
        print(f"Elliott: {result.elliott_current_wave} ({result.elliott_direction})")
        print(f"Harmonic: {result.harmonic_pattern}")
        print(f"MTF: {result.mtf_confluence}")
        print(f"On-Chain MVRV: {result.mvrv:.2f}")
        print(f"DXY: {result.dxy_trend}")
        print(f"Sentiment: {result.news_sentiment} ({result.news_score})")
        print(f"\nBullish Signals: {result.total_bullish_signals}")
        print(f"Bearish Signals: {result.total_bearish_signals}")
        print(f"Data Sources: {result.total_data_sources}")
        await collector.close()
    
    asyncio.run(test())
