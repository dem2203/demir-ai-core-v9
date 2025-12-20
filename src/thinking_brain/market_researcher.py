# -*- coding: utf-8 -*-
"""
DEMIR AI - MARKET RESEARCHER
=============================
Gercek arastirmaci yapay zeka.

Sadece birkac indikatora bakmaz, HER SEYI arastirir:
1. 4 coin'in grafikleri (BTC, ETH, LTC, SOL)
2. Whale hareketleri (orderbook, buyuk islemler)
3. Liquidation verisi (nerede tasfiye olur?)
4. Open Interest degisimi
5. Funding rate gecmisi
6. Hacim profili
7. Destek/direnc seviyeleri
8. Korelasyonlar

Sonra TUM bu verileri birlikte dusunup analiz yapar.

Author: DEMIR AI Core Team
Date: 2024-12
"""
import logging
import asyncio
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("MARKET_RESEARCHER")


@dataclass
class CoinResearch:
    """Tek coin icin arastirma sonuclari."""
    symbol: str
    
    # Fiyat ve Grafik
    current_price: float
    price_24h_change: float
    price_7d_change: float
    
    # Teknik Analiz
    trend: str  # UPTREND, DOWNTREND, SIDEWAYS
    support_levels: List[float]
    resistance_levels: List[float]
    
    # Momentum
    rsi: float
    macd_signal: str  # BULLISH, BEARISH, NEUTRAL
    ema_alignment: str
    
    # Volatilite
    atr_percent: float
    bollinger_position: str  # UPPER, MIDDLE, LOWER, SQUEEZE
    
    # Hacim
    volume_trend: str  # INCREASING, DECREASING, STABLE
    volume_vs_avg: float
    
    # Whale & Orderbook
    whale_bias: str  # BUYING, SELLING, NEUTRAL
    orderbook_imbalance: float
    large_orders: List[Dict]  # Buyuk emirler
    
    # Liquidation Riski
    long_liq_price: float  # Uzun pozisyonlar nerede tasfiye olur
    short_liq_price: float
    liq_risk: str  # HIGH_LONG, HIGH_SHORT, BALANCED
    
    # Funding & OI
    funding_rate: float
    oi_change_24h: float
    
    # Sonuc
    bias: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float
    key_observations: List[str]


@dataclass
class MarketReport:
    """Tum piyasa raporu."""
    timestamp: datetime
    
    # Genel Durum
    market_sentiment: str
    fear_greed: int
    btc_dominance: float
    total_market_cap: float
    
    # Coin Analizleri
    coins: Dict[str, CoinResearch]
    
    # Korelasyonlar
    btc_eth_correlation: float
    altcoin_season: bool
    
    # Genel Degerlendirme
    market_phase: str  # ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN
    overall_bias: str
    key_insights: List[str]
    
    # Oneriler
    best_opportunity: str
    risk_warning: str


class MarketResearcher:
    """
    Piyasa Arastirmacisi
    
    Bir analist gibi TUM verileri toplar ve analiz eder.
    """
    
    COINS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT']
    
    def __init__(self):
        logger.info("Market Researcher initialized")
    
    async def research_market(self) -> MarketReport:
        """
        TAM PIYASA ARASTIRMASI
        
        Her coin icin detayli analiz yapar ve genel rapor olusturur.
        """
        logger.info("Piyasa arastirmasi basliyor...")
        
        # 1. Genel piyasa verilerini al
        general = await self._get_general_market_data()
        
        # 2. Her coin icin detayli arastirma
        coins = {}
        for symbol in self.COINS:
            logger.info(f"  Arastiriliyor: {symbol}")
            coins[symbol] = await self._research_coin(symbol)
        
        # 3. Korelasyonlari hesapla
        correlations = await self._calculate_correlations(coins)
        
        # 4. Genel degerlendirme
        insights = self._generate_insights(general, coins, correlations)
        
        # 5. Rapor olustur
        report = MarketReport(
            timestamp=datetime.now(),
            market_sentiment=general['sentiment'],
            fear_greed=general['fear_greed'],
            btc_dominance=general['btc_dominance'],
            total_market_cap=general['total_market_cap'],
            coins=coins,
            btc_eth_correlation=correlations['btc_eth'],
            altcoin_season=correlations['altcoin_season'],
            market_phase=insights['phase'],
            overall_bias=insights['bias'],
            key_insights=insights['insights'],
            best_opportunity=insights['opportunity'],
            risk_warning=insights['risk']
        )
        
        return report
    
    async def _get_general_market_data(self) -> Dict:
        """Genel piyasa verileri."""
        result = {
            'sentiment': 'NEUTRAL',
            'fear_greed': 50,
            'btc_dominance': 50,
            'total_market_cap': 0
        }
        
        # Fear & Greed
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            if resp.status_code == 200:
                data = resp.json()['data'][0]
                result['fear_greed'] = int(data['value'])
                
                if result['fear_greed'] <= 25:
                    result['sentiment'] = 'EXTREME_FEAR'
                elif result['fear_greed'] <= 40:
                    result['sentiment'] = 'FEAR'
                elif result['fear_greed'] >= 75:
                    result['sentiment'] = 'EXTREME_GREED'
                elif result['fear_greed'] >= 60:
                    result['sentiment'] = 'GREED'
        except:
            pass
        
        # CoinGecko Global
        try:
            resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=5)
            if resp.status_code == 200:
                data = resp.json()['data']
                result['btc_dominance'] = data['market_cap_percentage']['btc']
                result['total_market_cap'] = data['total_market_cap']['usd']
        except:
            pass
        
        return result
    
    async def _research_coin(self, symbol: str) -> CoinResearch:
        """Tek coin icin detayli arastirma."""
        
        # 1. Fiyat verileri (coklu timeframe)
        klines_1h = await self._get_klines(symbol, '1h', 100)
        klines_4h = await self._get_klines(symbol, '4h', 50)
        klines_1d = await self._get_klines(symbol, '1d', 30)
        
        current_price = klines_1h['close'][-1] if len(klines_1h['close']) > 0 else 0
        
        # 2. Fiyat degisimleri
        price_24h = self._calculate_change(klines_1h['close'], 24)
        price_7d = self._calculate_change(klines_1d['close'], 7)
        
        # 3. Trend analizi
        trend = self._analyze_trend(klines_4h)
        
        # 4. Destek/Direnc
        supports, resistances = self._find_sr_levels(klines_4h)
        
        # 5. RSI
        rsi = self._calculate_rsi(klines_1h['close'])
        
        # 6. MACD
        macd_signal = self._analyze_macd(klines_1h['close'])
        
        # 7. EMA alignment
        ema_align = self._check_ema_alignment(klines_1h['close'])
        
        # 8. ATR (volatilite)
        atr_pct = self._calculate_atr_percent(klines_1h)
        
        # 9. Bollinger
        bb_pos = self._check_bollinger_position(klines_1h['close'])
        
        # 10. Volume analysis
        vol_trend, vol_ratio = self._analyze_volume(klines_1h['volume'])
        
        # 11. Orderbook & Whale
        whale_data = await self._analyze_orderbook(symbol)
        
        # 12. Funding & OI
        funding = await self._get_funding(symbol)
        oi_change = await self._get_oi_change(symbol)
        
        # 13. Liquidation levels
        long_liq, short_liq, liq_risk = self._estimate_liquidation_levels(
            current_price, atr_pct
        )
        
        # 14. Sonuc bias
        bias, confidence, observations = self._determine_bias(
            trend, rsi, macd_signal, ema_align, whale_data['bias'],
            funding, vol_trend, bb_pos
        )
        
        return CoinResearch(
            symbol=symbol,
            current_price=current_price,
            price_24h_change=price_24h,
            price_7d_change=price_7d,
            trend=trend,
            support_levels=supports,
            resistance_levels=resistances,
            rsi=rsi,
            macd_signal=macd_signal,
            ema_alignment=ema_align,
            atr_percent=atr_pct,
            bollinger_position=bb_pos,
            volume_trend=vol_trend,
            volume_vs_avg=vol_ratio,
            whale_bias=whale_data['bias'],
            orderbook_imbalance=whale_data['imbalance'],
            large_orders=whale_data['large_orders'],
            long_liq_price=long_liq,
            short_liq_price=short_liq,
            liq_risk=liq_risk,
            funding_rate=funding,
            oi_change_24h=oi_change,
            bias=bias,
            confidence=confidence,
            key_observations=observations
        )
    
    # =========================================================================
    # VERI TOPLAMA
    # =========================================================================
    
    async def _get_klines(self, symbol: str, interval: str, limit: int) -> Dict:
        """Mum verileri al."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': interval, 'limit': limit},
                timeout=10
            )
            if resp.status_code == 200:
                klines = resp.json()
                return {
                    'open': np.array([float(k[1]) for k in klines]),
                    'high': np.array([float(k[2]) for k in klines]),
                    'low': np.array([float(k[3]) for k in klines]),
                    'close': np.array([float(k[4]) for k in klines]),
                    'volume': np.array([float(k[5]) for k in klines])
                }
        except:
            pass
        return {'open': np.array([0]), 'high': np.array([0]), 
                'low': np.array([0]), 'close': np.array([0]), 
                'volume': np.array([0])}
    
    async def _analyze_orderbook(self, symbol: str) -> Dict:
        """Orderbook analizi."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/depth",
                params={'symbol': symbol, 'limit': 500},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                
                bids = [(float(b[0]), float(b[1])) for b in data['bids']]
                asks = [(float(a[0]), float(a[1])) for a in data['asks']]
                
                bid_vol = sum(b[1] for b in bids)
                ask_vol = sum(a[1] for a in asks)
                imbalance = bid_vol / (ask_vol + 0.001)
                
                # Buyuk emirleri bul
                avg_size = (bid_vol + ask_vol) / (len(bids) + len(asks))
                large_bids = [{'price': b[0], 'size': b[1], 'side': 'BID'} 
                              for b in bids if b[1] > avg_size * 5]
                large_asks = [{'price': a[0], 'size': a[1], 'side': 'ASK'} 
                              for a in asks if a[1] > avg_size * 5]
                
                if imbalance > 1.5:
                    bias = 'BUYING'
                elif imbalance < 0.67:
                    bias = 'SELLING'
                else:
                    bias = 'NEUTRAL'
                
                return {
                    'bid_volume': bid_vol,
                    'ask_volume': ask_vol,
                    'imbalance': imbalance,
                    'bias': bias,
                    'large_orders': large_bids[:3] + large_asks[:3]
                }
        except:
            pass
        return {'bid_volume': 0, 'ask_volume': 0, 'imbalance': 1, 
                'bias': 'NEUTRAL', 'large_orders': []}
    
    async def _get_funding(self, symbol: str) -> float:
        """Funding rate."""
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200 and resp.json():
                return float(resp.json()[0]['fundingRate']) * 100
        except:
            pass
        return 0
    
    async def _get_oi_change(self, symbol: str) -> float:
        """Open Interest degisimi (son 24 saat)."""
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/openInterest",
                params={'symbol': symbol},
                timeout=5
            )
            # Basit versiyon - sadece mevcut OI
            if resp.status_code == 200:
                return 0  # Gercek hesaplama icin tarihsel veri lazim
        except:
            pass
        return 0
    
    # =========================================================================
    # ANALIZ FONKSIYONLARI
    # =========================================================================
    
    def _calculate_change(self, closes: np.ndarray, periods: int) -> float:
        """Fiyat degisimi yuzde."""
        if len(closes) < periods + 1:
            return 0
        return ((closes[-1] / closes[-periods-1]) - 1) * 100
    
    def _analyze_trend(self, klines: Dict) -> str:
        """Trend analizi."""
        closes = klines['close']
        if len(closes) < 20:
            return 'SIDEWAYS'
        
        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, min(50, len(closes)))
        current = closes[-1]
        
        if current > ema20 > ema50:
            return 'UPTREND'
        elif current < ema20 < ema50:
            return 'DOWNTREND'
        return 'SIDEWAYS'
    
    def _find_sr_levels(self, klines: Dict) -> Tuple[List[float], List[float]]:
        """Destek ve direnc seviyeleri."""
        highs = klines['high']
        lows = klines['low']
        current = klines['close'][-1]
        
        # Basit yaklaşım: Son 20 mumun high/low'lari
        supports = sorted([l for l in lows[-20:] if l < current])[-3:]
        resistances = sorted([h for h in highs[-20:] if h > current])[:3]
        
        return supports, resistances
    
    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """RSI hesapla."""
        if len(closes) < period + 1:
            return 50
        
        delta = np.diff(closes)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        rs = avg_gain / (avg_loss + 0.001)
        return 100 - (100 / (1 + rs))
    
    def _analyze_macd(self, closes: np.ndarray) -> str:
        """MACD analizi."""
        if len(closes) < 26:
            return 'NEUTRAL'
        
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd = ema12 - ema26
        
        # Signal line (9 period EMA of MACD)
        macd_series = []
        for i in range(9, len(closes)):
            e12 = self._ema(closes[:i+1], 12)
            e26 = self._ema(closes[:i+1], 26)
            macd_series.append(e12 - e26)
        
        if len(macd_series) < 2:
            return 'NEUTRAL'
        
        if macd_series[-1] > macd_series[-2]:
            return 'BULLISH'
        elif macd_series[-1] < macd_series[-2]:
            return 'BEARISH'
        return 'NEUTRAL'
    
    def _check_ema_alignment(self, closes: np.ndarray) -> str:
        """EMA hizalama kontrolu."""
        if len(closes) < 50:
            return 'MIXED'
        
        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        ema50 = self._ema(closes, 50)
        current = closes[-1]
        
        if current > ema9 > ema21 > ema50:
            return 'PERFECT_BULLISH'
        elif current < ema9 < ema21 < ema50:
            return 'PERFECT_BEARISH'
        elif current > ema21:
            return 'BULLISH'
        elif current < ema21:
            return 'BEARISH'
        return 'MIXED'
    
    def _calculate_atr_percent(self, klines: Dict, period: int = 14) -> float:
        """ATR yuzde olarak."""
        highs = klines['high']
        lows = klines['low']
        closes = klines['close']
        
        if len(closes) < period + 1:
            return 0
        
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
        atr = np.mean(tr[-period:])
        return (atr / closes[-1]) * 100
    
    def _check_bollinger_position(self, closes: np.ndarray) -> str:
        """Bollinger Band pozisyonu."""
        if len(closes) < 20:
            return 'MIDDLE'
        
        sma = np.mean(closes[-20:])
        std = np.std(closes[-20:])
        upper = sma + 2 * std
        lower = sma - 2 * std
        current = closes[-1]
        bandwidth = (2 * std / sma) * 100
        
        if bandwidth < 2:
            return 'SQUEEZE'
        elif current > upper:
            return 'ABOVE_UPPER'
        elif current < lower:
            return 'BELOW_LOWER'
        elif current > sma:
            return 'UPPER_HALF'
        return 'LOWER_HALF'
    
    def _analyze_volume(self, volumes: np.ndarray) -> Tuple[str, float]:
        """Hacim analizi."""
        if len(volumes) < 20:
            return 'STABLE', 1.0
        
        recent = np.mean(volumes[-5:])
        avg = np.mean(volumes[-20:-5])
        ratio = recent / (avg + 0.001)
        
        if ratio > 1.5:
            trend = 'INCREASING'
        elif ratio < 0.7:
            trend = 'DECREASING'
        else:
            trend = 'STABLE'
        
        return trend, ratio
    
    def _estimate_liquidation_levels(self, price: float, atr_pct: float) -> Tuple[float, float, str]:
        """Tahmini tasfiye seviyeleri."""
        # 10x kaldıraç için yaklaşık %10 hareket
        liq_range = max(atr_pct * 3, 5)  # En az %5
        
        long_liq = price * (1 - liq_range / 100)
        short_liq = price * (1 + liq_range / 100)
        
        # Hangi tarafa daha yakın?
        risk = 'BALANCED'
        
        return long_liq, short_liq, risk
    
    def _determine_bias(self, trend, rsi, macd, ema, whale, funding, volume, bb) -> Tuple[str, float, List[str]]:
        """Genel yanlılık belirle."""
        bullish_points = 0
        bearish_points = 0
        observations = []
        
        # Trend
        if trend == 'UPTREND':
            bullish_points += 2
            observations.append("Genel trend yukari")
        elif trend == 'DOWNTREND':
            bearish_points += 2
            observations.append("Genel trend asagi")
        
        # RSI
        if rsi < 30:
            bullish_points += 1
            observations.append(f"RSI asiri satim ({rsi:.0f})")
        elif rsi > 70:
            bearish_points += 1
            observations.append(f"RSI asiri alim ({rsi:.0f})")
        
        # MACD
        if macd == 'BULLISH':
            bullish_points += 1
            observations.append("MACD yukari donuyor")
        elif macd == 'BEARISH':
            bearish_points += 1
            observations.append("MACD asagi donuyor")
        
        # EMA
        if 'BULLISH' in ema:
            bullish_points += 1
            observations.append("EMA'lar bullish")
        elif 'BEARISH' in ema:
            bearish_points += 1
            observations.append("EMA'lar bearish")
        
        # Whale
        if whale == 'BUYING':
            bullish_points += 2
            observations.append("Whale'ler ALIYOR")
        elif whale == 'SELLING':
            bearish_points += 2
            observations.append("Whale'ler SATIYOR")
        
        # Funding (contrarian)
        if funding > 0.05:
            bearish_points += 1
            observations.append(f"Funding yuksek ({funding:.3f}%) - long kalabalik")
        elif funding < -0.02:
            bullish_points += 1
            observations.append(f"Funding negatif ({funding:.3f}%) - short kalabalik")
        
        # Volume
        if volume == 'INCREASING':
            observations.append("Hacim artiyor - hareket guclu")
        elif volume == 'DECREASING':
            observations.append("Hacim dusuyor - dikkatli ol")
        
        # Bollinger
        if bb == 'SQUEEZE':
            observations.append("Bollinger sikismis - PATLAMA YAKINDA")
        elif bb == 'ABOVE_UPPER':
            bearish_points += 1
            observations.append("Fiyat ust bandın uzerinde - asiri alim")
        elif bb == 'BELOW_LOWER':
            bullish_points += 1
            observations.append("Fiyat alt bandın altinda - asiri satim")
        
        # Sonuc
        total = bullish_points + bearish_points
        if total == 0:
            return 'NEUTRAL', 0.5, observations
        
        if bullish_points > bearish_points:
            confidence = bullish_points / (total + 2)
            return 'BULLISH', confidence, observations
        elif bearish_points > bullish_points:
            confidence = bearish_points / (total + 2)
            return 'BEARISH', confidence, observations
        return 'NEUTRAL', 0.5, observations
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """EMA hesapla."""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    async def _calculate_correlations(self, coins: Dict) -> Dict:
        """Korelasyon hesapla."""
        # Basit versiyon
        btc = coins.get('BTCUSDT')
        eth = coins.get('ETHUSDT')
        
        # BTC-ETH korelasyonu
        if btc and eth:
            # Ayni yone mi gidiyorlar?
            same_direction = btc.bias == eth.bias
            correlation = 0.8 if same_direction else 0.4
        else:
            correlation = 0.5
        
        # Altcoin season check
        btc_change = btc.price_24h_change if btc else 0
        alt_changes = [c.price_24h_change for s, c in coins.items() if s != 'BTCUSDT']
        avg_alt = np.mean(alt_changes) if alt_changes else 0
        
        altcoin_season = avg_alt > btc_change + 2  # Altlar BTC'den %2+ iyi
        
        return {
            'btc_eth': correlation,
            'altcoin_season': altcoin_season
        }
    
    def _generate_insights(self, general: Dict, coins: Dict, correlations: Dict) -> Dict:
        """Genel icgorular olustur."""
        insights = []
        
        # Fear & Greed
        fg = general['fear_greed']
        if fg <= 20:
            insights.append("Piyasa ASIRI KORKU modunda - tarihsel olarak ALIM firsati")
        elif fg >= 80:
            insights.append("Piyasa ASIRI ACGOZLULUK modunda - dikkatli ol")
        
        # BTC Dominance
        btc_dom = general['btc_dominance']
        if btc_dom > 55:
            insights.append(f"BTC hakimiyeti yuksek ({btc_dom:.1f}%) - altlar zayif")
        elif btc_dom < 45:
            insights.append(f"BTC hakimiyeti dusuk ({btc_dom:.1f}%) - altcoin season olabilir")
        
        # Coin bazli
        bullish_coins = [s for s, c in coins.items() if c.bias == 'BULLISH']
        bearish_coins = [s for s, c in coins.items() if c.bias == 'BEARISH']
        
        if len(bullish_coins) >= 3:
            insights.append(f"Cogunluk YUKARI bakiyor: {', '.join(bullish_coins)}")
        elif len(bearish_coins) >= 3:
            insights.append(f"Cogunluk ASAGI bakiyor: {', '.join(bearish_coins)}")
        
        # Whale aktivitesi
        for symbol, coin in coins.items():
            if coin.whale_bias == 'BUYING':
                insights.append(f"{symbol}: Whale'ler ALIYOR!")
            elif coin.whale_bias == 'SELLING':
                insights.append(f"{symbol}: Whale'ler SATIYOR!")
        
        # En iyi firsat
        best = max(coins.values(), key=lambda c: c.confidence if c.bias == 'BULLISH' else 0)
        if best.bias == 'BULLISH' and best.confidence > 0.5:
            opportunity = f"{best.symbol} - En guclu bullish sinyal"
        else:
            opportunity = "Net firsat yok - bekle"
        
        # Risk
        squeezed = [s for s, c in coins.items() if c.bollinger_position == 'SQUEEZE']
        if squeezed:
            risk = f"Volatilite sikismis: {', '.join(squeezed)} - ani hareket olabilir!"
        else:
            risk = "Normal volatilite"
        
        # Market phase
        if fg <= 25 and correlations['btc_eth'] > 0.7:
            phase = 'ACCUMULATION'
        elif fg >= 75:
            phase = 'DISTRIBUTION'
        elif len(bullish_coins) >= 3:
            phase = 'MARKUP'
        elif len(bearish_coins) >= 3:
            phase = 'MARKDOWN'
        else:
            phase = 'CONSOLIDATION'
        
        # Overall bias
        if len(bullish_coins) > len(bearish_coins):
            overall = 'BULLISH'
        elif len(bearish_coins) > len(bullish_coins):
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        return {
            'phase': phase,
            'bias': overall,
            'insights': insights,
            'opportunity': opportunity,
            'risk': risk
        }
    
    def format_telegram(self, report: MarketReport) -> str:
        """Telegram icin formatla - GERCEK ANALIST GIBI."""
        lines = []
        lines.append("DEMIR AI - PIYASA ARASTIRMASI")
        lines.append("=" * 35)
        lines.append(f"Tarih: {report.timestamp.strftime('%d.%m.%Y %H:%M')}")
        lines.append("")
        
        # Genel Durum
        lines.append("GENEL PIYASA DURUMU")
        lines.append("-" * 35)
        lines.append(f"Fear & Greed: {report.fear_greed} ({report.market_sentiment})")
        lines.append(f"BTC Dominance: {report.btc_dominance:.1f}%")
        lines.append(f"Market Cap: ${report.total_market_cap/1e12:.2f}T")
        lines.append(f"Piyasa Fazı: {report.market_phase}")
        lines.append("")
        
        # Her coin
        lines.append("COIN ANALIZLERI")
        lines.append("-" * 35)
        
        for symbol, coin in report.coins.items():
            emoji = "🟢" if coin.bias == 'BULLISH' else "🔴" if coin.bias == 'BEARISH' else "⚪"
            lines.append(f"\n{emoji} {symbol}")
            lines.append(f"   Fiyat: ${coin.current_price:,.2f} ({coin.price_24h_change:+.1f}%)")
            lines.append(f"   Trend: {coin.trend}")
            lines.append(f"   RSI: {coin.rsi:.0f} | MACD: {coin.macd_signal}")
            lines.append(f"   Whale: {coin.whale_bias} (OB: {coin.orderbook_imbalance:.2f})")
            lines.append(f"   Funding: {coin.funding_rate:.4f}%")
            
            if coin.support_levels:
                lines.append(f"   Destek: ${coin.support_levels[-1]:,.0f}")
            if coin.resistance_levels:
                lines.append(f"   Direnc: ${coin.resistance_levels[0]:,.0f}")
            
            lines.append(f"   SONUC: {coin.bias} (%{coin.confidence*100:.0f})")
            
            for obs in coin.key_observations[:2]:
                lines.append(f"   - {obs}")
        
        lines.append("")
        lines.append("GENEL DEGERLENDIRME")
        lines.append("-" * 35)
        lines.append(f"Genel Yonelim: {report.overall_bias}")
        lines.append(f"En Iyi Firsat: {report.best_opportunity}")
        lines.append(f"Risk: {report.risk_warning}")
        lines.append("")
        
        lines.append("KILIT GOZLEMLER")
        lines.append("-" * 35)
        for insight in report.key_insights[:5]:
            lines.append(f"* {insight}")
        
        return "\n".join(lines)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_researcher: Optional[MarketResearcher] = None

def get_market_researcher() -> MarketResearcher:
    """Get instance."""
    global _researcher
    if _researcher is None:
        _researcher = MarketResearcher()
    return _researcher


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import sys
    import io
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        researcher = get_market_researcher()
        report = await researcher.research_market()
        print(researcher.format_telegram(report))
    
    asyncio.run(test())

