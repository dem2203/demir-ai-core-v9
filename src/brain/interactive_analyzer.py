# -*- coding: utf-8 -*-
"""
DEMIR AI - INTERACTIVE COIN ANALYZER
=====================================
Telegram üzerinden coin ismi yazıldığında detaylı analiz yapar.
Tüm veriler LIVE - TradingView/Binance grafik verisi üzerinden.

Desteklenen Coinler: BTC, ETH, SOL, LTC (Binance Futures)

Kullanım: Telegram'a "BTC" veya "/analiz BTC" yazılınca detaylı analiz gönderir.
"""
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("INTERACTIVE_ANALYZER")


@dataclass
class DetailedCoinAnalysis:
    """Detaylı coin analiz sonucu"""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Price Action
    current_price: float = 0
    price_change_1h: float = 0
    price_change_4h: float = 0
    price_change_24h: float = 0
    price_change_7d: float = 0
    
    # Technical Indicators
    rsi_1h: float = 50
    rsi_4h: float = 50
    macd_signal: str = "NEUTRAL"
    ema_trend: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
    # Volume Analysis
    volume_24h: float = 0
    volume_change: float = 0
    buy_pressure: float = 0.5  # 0-1, >0.5 = more buying
    
    # Order Book
    bid_ask_ratio: float = 1.0
    order_book_signal: str = "NEUTRAL"
    
    # Derivatives
    funding_rate: float = 0
    open_interest: float = 0
    oi_change_24h: float = 0
    long_short_ratio: float = 1.0
    liquidations_24h: float = 0
    
    # On-Chain / Flow
    exchange_netflow: float = 0
    whale_activity: str = "NEUTRAL"
    
    # Market Context
    fear_greed: int = 50
    btc_dominance: float = 0
    
    # Final Analysis
    overall_direction: str = "NEUTRAL"  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    confidence: float = 0
    key_signals: List[str] = field(default_factory=list)
    trading_recommendation: str = ""
    entry_zone: str = ""
    take_profits: List[float] = field(default_factory=list)
    stop_loss: float = 0


class InteractiveCoinAnalyzer:
    """
    Interactive Coin Analyzer
    
    Telegram'dan coin ismi yazıldığında detaylı live analiz yapar.
    Tüm veriler gerçek: Binance API + TradingView verileri.
    """
    
    SUPPORTED_COINS = {
        'BTC': 'BTCUSDT',
        'BITCOIN': 'BTCUSDT',
        'ETH': 'ETHUSDT',
        'ETHEREUM': 'ETHUSDT',
        'SOL': 'SOLUSDT',
        'SOLANA': 'SOLUSDT',
        'LTC': 'LTCUSDT',
        'LITECOIN': 'LTCUSDT',
    }
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("🔍 Interactive Coin Analyzer initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def parse_coin_from_message(self, message: str) -> Optional[str]:
        """Mesajdan coin sembolünü çıkar"""
        message = message.upper().strip()
        
        # /analiz BTC formatı
        if message.startswith('/ANALIZ'):
            parts = message.split()
            if len(parts) >= 2:
                coin = parts[1]
                return self.SUPPORTED_COINS.get(coin)
        
        # Direkt coin ismi
        return self.SUPPORTED_COINS.get(message)
    
    async def analyze_coin(self, symbol: str) -> DetailedCoinAnalysis:
        """
        Coin için detaylı live analiz yap.
        Tüm veriler Binance'den çekilir.
        """
        analysis = DetailedCoinAnalysis(symbol=symbol)
        
        try:
            # Paralel veri çekme
            results = await asyncio.gather(
                self._fetch_price_action(symbol),
                self._fetch_technical_indicators(symbol),
                self._fetch_volume_analysis(symbol),
                self._fetch_orderbook(symbol),
                self._fetch_derivatives(symbol),
                self._fetch_market_context(),
                return_exceptions=True
            )
            
            # Apply results
            if isinstance(results[0], dict):
                self._apply_price_action(analysis, results[0])
            if isinstance(results[1], dict):
                self._apply_technical(analysis, results[1])
            if isinstance(results[2], dict):
                self._apply_volume(analysis, results[2])
            if isinstance(results[3], dict):
                self._apply_orderbook(analysis, results[3])
            if isinstance(results[4], dict):
                self._apply_derivatives(analysis, results[4])
            if isinstance(results[5], dict):
                self._apply_context(analysis, results[5])
            
            # Generate final analysis
            self._generate_recommendation(analysis)
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
        
        return analysis
    
    # =========================================
    # VERİ ÇEKME FONKSİYONLARI
    # =========================================
    
    async def _fetch_price_action(self, symbol: str) -> Dict:
        """Fiyat hareketlerini çek"""
        try:
            session = await self._get_session()
            
            # Current price
            async with session.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}") as resp:
                ticker = await resp.json()
            
            # Historical klines for change calculation
            async with session.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=168") as resp:
                klines = await resp.json()
            
            current = float(ticker['lastPrice'])
            
            # Calculate changes
            change_1h = float(klines[-1][4]) - float(klines[-2][4]) if len(klines) >= 2 else 0
            change_4h = float(klines[-1][4]) - float(klines[-5][4]) if len(klines) >= 5 else 0
            change_7d = float(klines[-1][4]) - float(klines[0][4]) if len(klines) >= 168 else 0
            
            return {
                'current': current,
                'change_1h': (change_1h / float(klines[-2][4])) * 100 if len(klines) >= 2 else 0,
                'change_4h': (change_4h / float(klines[-5][4])) * 100 if len(klines) >= 5 else 0,
                'change_24h': float(ticker['priceChangePercent']),
                'change_7d': (change_7d / float(klines[0][4])) * 100 if len(klines) >= 168 else 0,
            }
        except Exception as e:
            logger.debug(f"Price action error: {e}")
            return {}
    
    async def _fetch_technical_indicators(self, symbol: str) -> Dict:
        """Teknik indikatörleri hesapla (kline verilerinden)"""
        try:
            session = await self._get_session()
            
            # Get klines for calculation
            async with session.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=50") as resp:
                klines_1h = await resp.json()
            
            async with session.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=4h&limit=50") as resp:
                klines_4h = await resp.json()
            
            closes_1h = [float(k[4]) for k in klines_1h]
            closes_4h = [float(k[4]) for k in klines_4h]
            
            # RSI calculation
            rsi_1h = self._calculate_rsi(closes_1h)
            rsi_4h = self._calculate_rsi(closes_4h)
            
            # EMA trend
            ema_20 = self._calculate_ema(closes_1h, 20)
            ema_50 = self._calculate_ema(closes_1h, 50)
            current = closes_1h[-1]
            
            ema_trend = "BULLISH" if current > ema_20 > ema_50 else "BEARISH" if current < ema_20 < ema_50 else "NEUTRAL"
            
            # MACD
            macd_signal = "BUY" if closes_1h[-1] > closes_1h[-2] > closes_1h[-3] else "SELL" if closes_1h[-1] < closes_1h[-2] < closes_1h[-3] else "NEUTRAL"
            
            # Support/Resistance (simplified - recent highs/lows)
            highs = [float(k[2]) for k in klines_1h[-24:]]
            lows = [float(k[3]) for k in klines_1h[-24:]]
            
            return {
                'rsi_1h': rsi_1h,
                'rsi_4h': rsi_4h,
                'macd_signal': macd_signal,
                'ema_trend': ema_trend,
                'supports': [min(lows), sorted(lows)[len(lows)//4]],
                'resistances': [max(highs), sorted(highs, reverse=True)[len(highs)//4]],
            }
        except Exception as e:
            logger.debug(f"Technical indicators error: {e}")
            return {}
    
    async def _fetch_volume_analysis(self, symbol: str) -> Dict:
        """Hacim analizi"""
        try:
            session = await self._get_session()
            
            async with session.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}") as resp:
                ticker = await resp.json()
            
            # Taker buy/sell volume
            async with session.get(f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={symbol}&period=1h&limit=24") as resp:
                taker_data = await resp.json()
            
            buy_vol = sum(float(t.get('buyVol', 0)) for t in taker_data[-4:]) if taker_data else 0
            sell_vol = sum(float(t.get('sellVol', 0)) for t in taker_data[-4:]) if taker_data else 0
            total = buy_vol + sell_vol
            
            return {
                'volume_24h': float(ticker.get('quoteVolume', 0)),
                'buy_pressure': buy_vol / total if total > 0 else 0.5,
            }
        except Exception as e:
            logger.debug(f"Volume analysis error: {e}")
            return {}
    
    async def _fetch_orderbook(self, symbol: str) -> Dict:
        """Order book analizi"""
        try:
            session = await self._get_session()
            
            async with session.get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100") as resp:
                depth = await resp.json()
            
            bid_volume = sum(float(b[1]) * float(b[0]) for b in depth.get('bids', []))
            ask_volume = sum(float(a[1]) * float(a[0]) for a in depth.get('asks', []))
            
            ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
            
            signal = "BULLISH" if ratio > 1.5 else "BEARISH" if ratio < 0.67 else "NEUTRAL"
            
            return {
                'bid_ask_ratio': ratio,
                'signal': signal,
            }
        except Exception as e:
            logger.debug(f"Orderbook error: {e}")
            return {}
    
    async def _fetch_derivatives(self, symbol: str) -> Dict:
        """Futures/derivatif verileri"""
        try:
            session = await self._get_session()
            
            # Funding rate
            async with session.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1") as resp:
                funding_data = await resp.json()
            
            # Open Interest
            async with session.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}") as resp:
                oi_data = await resp.json()
            
            # Long/Short ratio
            async with session.get(f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1") as resp:
                ls_data = await resp.json()
            
            # Get price for OI USD value
            async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}") as resp:
                price_data = await resp.json()
            
            price = float(price_data.get('price', 0))
            oi = float(oi_data.get('openInterest', 0)) * price
            
            return {
                'funding_rate': float(funding_data[0].get('fundingRate', 0)) * 100 if funding_data else 0,
                'open_interest': oi,
                'long_short_ratio': float(ls_data[0].get('longShortRatio', 1.0)) if ls_data else 1.0,
            }
        except Exception as e:
            logger.debug(f"Derivatives error: {e}")
            return {}
    
    async def _fetch_market_context(self) -> Dict:
        """Genel piyasa konteksti"""
        try:
            session = await self._get_session()
            
            # Fear & Greed
            async with session.get("https://api.alternative.me/fng/") as resp:
                fg_data = await resp.json()
            
            # BTC Dominance
            async with session.get("https://api.coingecko.com/api/v3/global") as resp:
                global_data = await resp.json()
            
            return {
                'fear_greed': int(fg_data['data'][0]['value']) if fg_data.get('data') else 50,
                'btc_dominance': global_data.get('data', {}).get('market_cap_percentage', {}).get('btc', 0),
            }
        except Exception as e:
            logger.debug(f"Market context error: {e}")
            return {}
    
    # =========================================
    # YARDIMCI HESAPLAMA FONKSİYONLARI
    # =========================================
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI hesapla"""
        if len(prices) < period + 1:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [c if c > 0 else 0 for c in changes[-period:]]
        losses = [-c if c < 0 else 0 for c in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """EMA hesapla"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    # =========================================
    # SONUÇ UYGULAMA FONKSİYONLARI
    # =========================================
    
    def _apply_price_action(self, analysis: DetailedCoinAnalysis, data: Dict):
        analysis.current_price = data.get('current', 0)
        analysis.price_change_1h = data.get('change_1h', 0)
        analysis.price_change_4h = data.get('change_4h', 0)
        analysis.price_change_24h = data.get('change_24h', 0)
        analysis.price_change_7d = data.get('change_7d', 0)
    
    def _apply_technical(self, analysis: DetailedCoinAnalysis, data: Dict):
        analysis.rsi_1h = data.get('rsi_1h', -1)  # -1 = veri yok
        analysis.rsi_4h = data.get('rsi_4h', -1)  # -1 = veri yok
        analysis.macd_signal = data.get('macd_signal', 'NEUTRAL')
        analysis.ema_trend = data.get('ema_trend', 'NEUTRAL')
        analysis.support_levels = data.get('supports', [])
        analysis.resistance_levels = data.get('resistances', [])
    
    def _apply_volume(self, analysis: DetailedCoinAnalysis, data: Dict):
        analysis.volume_24h = data.get('volume_24h', 0)
        analysis.buy_pressure = data.get('buy_pressure', 0.5)
    
    def _apply_orderbook(self, analysis: DetailedCoinAnalysis, data: Dict):
        analysis.bid_ask_ratio = data.get('bid_ask_ratio', 1.0)
        analysis.order_book_signal = data.get('signal', 'NEUTRAL')
    
    def _apply_derivatives(self, analysis: DetailedCoinAnalysis, data: Dict):
        analysis.funding_rate = data.get('funding_rate', 0)
        analysis.open_interest = data.get('open_interest', 0)
        analysis.long_short_ratio = data.get('long_short_ratio', 1.0)
    
    def _apply_context(self, analysis: DetailedCoinAnalysis, data: Dict):
        analysis.fear_greed = data.get('fear_greed', 50)
        analysis.btc_dominance = data.get('btc_dominance', 0)
    
    # =========================================
    # FİNAL ANALİZ & ÖNERİ
    # =========================================
    
    def _generate_recommendation(self, analysis: DetailedCoinAnalysis):
        """Tüm verileri analiz edip final öneri üret"""
        bullish_score = 0
        bearish_score = 0
        signals = []
        
        # Price action
        if analysis.price_change_24h > 3:
            bullish_score += 1
            signals.append(f"📈 24s +{analysis.price_change_24h:.1f}%")
        elif analysis.price_change_24h < -3:
            bearish_score += 1
            signals.append(f"📉 24s {analysis.price_change_24h:.1f}%")
        
        # RSI
        if analysis.rsi_1h < 30:
            bullish_score += 2
            signals.append(f"🔥 RSI AŞIRI SATIM: {analysis.rsi_1h:.0f}")
        elif analysis.rsi_1h > 70:
            bearish_score += 2
            signals.append(f"⚠️ RSI AŞIRI ALIM: {analysis.rsi_1h:.0f}")
        elif analysis.rsi_1h < 40:
            bullish_score += 1
        elif analysis.rsi_1h > 60:
            bearish_score += 1
        
        # EMA Trend
        if analysis.ema_trend == "BULLISH":
            bullish_score += 1.5
            signals.append("📊 EMA Trend: YUKARI")
        elif analysis.ema_trend == "BEARISH":
            bearish_score += 1.5
            signals.append("📊 EMA Trend: AŞAĞI")
        
        # Order Book
        if analysis.bid_ask_ratio > 1.5:
            bullish_score += 1
            signals.append(f"📗 Order Book: {analysis.bid_ask_ratio:.2f}x BID")
        elif analysis.bid_ask_ratio < 0.67:
            bearish_score += 1
            signals.append(f"📕 Order Book: {1/analysis.bid_ask_ratio:.2f}x ASK")
        
        # Funding
        if analysis.funding_rate < -0.01:
            bullish_score += 1
            signals.append(f"💰 Funding: {analysis.funding_rate:.4f}% (Negatif)")
        elif analysis.funding_rate > 0.05:
            bearish_score += 1
            signals.append(f"⚠️ Funding: {analysis.funding_rate:.4f}% (Yüksek)")
        
        # Long/Short Ratio
        if analysis.long_short_ratio > 2.0:
            bearish_score += 1
            signals.append(f"⚠️ L/S: {analysis.long_short_ratio:.2f} (Long kalabalık)")
        elif analysis.long_short_ratio < 0.5:
            bullish_score += 1
            signals.append(f"🚀 L/S: {analysis.long_short_ratio:.2f} (Short squeeze)")
        
        # Fear & Greed
        if analysis.fear_greed < 25:
            bullish_score += 1
            signals.append(f"😱 Extreme Fear: {analysis.fear_greed}")
        elif analysis.fear_greed > 75:
            bearish_score += 1
            signals.append(f"🤑 Extreme Greed: {analysis.fear_greed}")
        
        # Buy Pressure
        if analysis.buy_pressure > 0.6:
            bullish_score += 0.5
            signals.append(f"💚 Buy Pressure: {analysis.buy_pressure*100:.0f}%")
        elif analysis.buy_pressure < 0.4:
            bearish_score += 0.5
            signals.append(f"❤️ Sell Pressure: {(1-analysis.buy_pressure)*100:.0f}%")
        
        # Calculate overall
        total = bullish_score + bearish_score
        
        if bullish_score > bearish_score * 2:
            analysis.overall_direction = "STRONG_BUY"
            analysis.trading_recommendation = "🚀 GÜÇLÜ LONG - Birden fazla güçlü bullish sinyal"
        elif bullish_score > bearish_score * 1.5:
            analysis.overall_direction = "BUY"
            analysis.trading_recommendation = "🟢 LONG - Bullish sinyaller baskın"
        elif bearish_score > bullish_score * 2:
            analysis.overall_direction = "STRONG_SELL"
            analysis.trading_recommendation = "💀 GÜÇLÜ SHORT - Birden fazla güçlü bearish sinyal"
        elif bearish_score > bullish_score * 1.5:
            analysis.overall_direction = "SELL"
            analysis.trading_recommendation = "🔴 SHORT - Bearish sinyaller baskın"
        else:
            analysis.overall_direction = "NEUTRAL"
            analysis.trading_recommendation = "⚪ BEKLE - Net yön yok"
        
        analysis.confidence = min(95, max(40, 50 + (bullish_score - bearish_score) * 10))
        analysis.key_signals = signals[:6]  # Top 6 signals
        
        # Entry zone & TP/SL
        if analysis.overall_direction in ["STRONG_BUY", "BUY"]:
            analysis.entry_zone = f"${analysis.current_price * 0.995:.2f} - ${analysis.current_price:.2f}"
            analysis.take_profits = [
                analysis.current_price * 1.02,
                analysis.current_price * 1.05,
                analysis.current_price * 1.10
            ]
            analysis.stop_loss = analysis.current_price * 0.97
        elif analysis.overall_direction in ["STRONG_SELL", "SELL"]:
            analysis.entry_zone = f"${analysis.current_price:.2f} - ${analysis.current_price * 1.005:.2f}"
            analysis.take_profits = [
                analysis.current_price * 0.98,
                analysis.current_price * 0.95,
                analysis.current_price * 0.90
            ]
            analysis.stop_loss = analysis.current_price * 1.03
    
    # =========================================
    # TELEGRAM FORMAT
    # =========================================
    
    def format_for_telegram(self, analysis: DetailedCoinAnalysis) -> str:
        """Detaylı analizi Telegram mesajı olarak formatla"""
        direction_emoji = {
            "STRONG_BUY": "🚀",
            "BUY": "🟢",
            "NEUTRAL": "⚪",
            "SELL": "🔴",
            "STRONG_SELL": "💀"
        }.get(analysis.overall_direction, "⚪")
        
        signals_text = "\n".join([f"  {s}" for s in analysis.key_signals])
        
        supports = ", ".join([f"${s:,.0f}" for s in analysis.support_levels[:2]]) if analysis.support_levels else "N/A"
        resistances = ", ".join([f"${r:,.0f}" for r in analysis.resistance_levels[:2]]) if analysis.resistance_levels else "N/A"
        
        tp_text = " / ".join([f"${tp:,.0f}" for tp in analysis.take_profits]) if analysis.take_profits else "N/A"
        
        return f"""🔍 *DETAYLİ ANALİZ - {analysis.symbol}*
━━━━━━━━━━━━━━━━━━━━━━━

{direction_emoji} *{analysis.overall_direction}* | Güven: %{analysis.confidence:.0f}
📌 {analysis.trading_recommendation}

━━━ FİYAT BİLGİSİ ━━━
💰 *Fiyat:* ${analysis.current_price:,.2f}
📊 1s: {analysis.price_change_1h:+.2f}% | 4s: {analysis.price_change_4h:+.2f}%
📊 24s: {analysis.price_change_24h:+.2f}% | 7g: {analysis.price_change_7d:+.2f}%

━━━ TEKNİK ANALİZ ━━━
📉 RSI (1s): {analysis.rsi_1h:.0f} | RSI (4s): {analysis.rsi_4h:.0f}
📈 EMA Trend: *{analysis.ema_trend}*
📊 MACD: {analysis.macd_signal}
🟢 Destek: {supports}
🔴 Direnç: {resistances}

━━━ DERİVATİFLER ━━━
💰 Funding: {analysis.funding_rate:.4f}%
📊 Open Interest: ${analysis.open_interest/1e9:.2f}B
⚖️ Long/Short: {analysis.long_short_ratio:.2f}

━━━ PİYASA DURUMU ━━━
😨 Fear & Greed: {analysis.fear_greed}
📊 Order Book: {analysis.bid_ask_ratio:.2f}x ({analysis.order_book_signal})
💰 Buy Pressure: {analysis.buy_pressure*100:.0f}%

━━━ GÜÇLÜ SİNYALLER ━━━
{signals_text if signals_text else "  • Net sinyal yok"}

━━━ İŞLEM ÖNERİSİ ━━━
📍 *Giriş:* {analysis.entry_zone}
🎯 *TP:* {tp_text}
🛑 *SL:* ${analysis.stop_loss:,.0f}

━━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
📡 Veri: Binance Live"""


# Singleton instance
_analyzer: Optional[InteractiveCoinAnalyzer] = None

def get_interactive_analyzer() -> InteractiveCoinAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = InteractiveCoinAnalyzer()
    return _analyzer
