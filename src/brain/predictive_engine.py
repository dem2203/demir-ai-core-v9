# -*- coding: utf-8 -*-
"""
DEMIR AI - REAL PREDICTIVE ENGINE
==================================
Gerçek tahmin yapan sistem.

İndikatör uyarısı DEĞİL, gerçek tahmin:
- YÖN: Yukarı mı aşağı mı?
- HEDEF: Nereye gidecek?
- ZAMAN: Ne zaman olacak?
- NEDEN: Hangi faktörler destekliyor?
- GÜVEN: Geçmiş performansa göre ne kadar güvenilir?

Author: DEMIR AI Core Team
Date: 2024-12
"""
import logging
import asyncio
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("PREDICTIVE_ENGINE")


@dataclass
class Prediction:
    """Gerçek tahmin objesi."""
    symbol: str
    direction: str  # 'LONG' veya 'SHORT'
    confidence: float  # 0-100
    
    # Hedef fiyatlar
    current_price: float
    target_price: float
    stop_loss: float
    
    # Zaman tahmini
    expected_hours: int  # Kaç saat içinde
    
    # Confluence faktörleri
    factors: List[Dict]  # Her faktör: {name, signal, weight, value}
    total_score: float  # Toplam skor
    
    # Açıklama
    reasoning: str
    
    timestamp: datetime = field(default_factory=datetime.now)


class PredictiveEngine:
    """
    Gerçek Tahmin Motoru
    
    Tek indikatöre bakmaz, CONFLUENCE kullanır:
    1. Technical (RSI, MACD, BB, EMA)
    2. Sentiment (Fear&Greed, Funding, L/S Ratio)
    3. Volume (Spike, Whale activity)
    4. LSTM (ML prediction)
    5. Historical Pattern (Squeeze sonrası ne oldu?)
    
    Her faktöre ağırlık verir, toplam skor hesaplar.
    """
    
    # Minimum gereksinimler
    MIN_CONFLUENCE_SCORE = 35  # Düşürdüm: 4 faktör strong ise yeterli
    MIN_FACTORS_AGREE = 3  # En az 3 faktör aynı yönde olmalı
    
    # Faktör ağırlıkları (toplam 100)
    WEIGHTS = {
        'lstm_prediction': 20,      # LSTM ML tahmin
        'squeeze_direction': 15,    # Squeeze sonrası tarihsel yön
        'funding_rate': 12,         # Funding rate contrarian
        'fear_greed': 12,           # Fear/Greed contrarian
        'whale_activity': 12,       # Whale alım/satım
        'volume_trend': 10,         # Hacim trendi
        'rsi_divergence': 8,        # RSI sapma
        'ema_alignment': 6,         # EMA hizalama
        'ls_ratio': 5,              # Long/Short extreme
    }
    
    def __init__(self):
        self.historical_squeezes = {}  # Geçmiş squeeze sonuçları
        self.prediction_history = []    # Geçmiş tahminler
        logger.info("🎯 Predictive Engine initialized")
    
    async def predict(self, symbol: str = 'BTCUSDT') -> Optional[Prediction]:
        """
        ANA TAHMİN FONKSİYONU
        
        Returns:
            Prediction objesi veya None (yetersiz veri/güven)
        """
        logger.info(f"🔮 Analyzing {symbol}...")
        
        # 1. Tüm verileri topla
        data = await self._collect_all_data(symbol)
        if not data:
            logger.warning(f"Data collection failed for {symbol}")
            return None
        
        # 2. Her faktörü analiz et
        factors = await self._analyze_all_factors(symbol, data)
        
        # 3. Confluence hesapla
        direction, score, agreeing_factors = self._calculate_confluence(factors)
        
        # 4. Yeterli confluence var mı?
        if score < self.MIN_CONFLUENCE_SCORE or agreeing_factors < self.MIN_FACTORS_AGREE:
            logger.info(f"Insufficient confluence: {score:.0f}% ({agreeing_factors} factors)")
            return None
        
        # 5. Hedef fiyat hesapla
        current_price = data['current_price']
        target, stop_loss, expected_hours = self._calculate_targets(
            current_price, direction, data
        )
        
        # 6. Açıklama oluştur
        reasoning = self._generate_reasoning(direction, factors, score)
        
        prediction = Prediction(
            symbol=symbol,
            direction=direction,
            confidence=score,
            current_price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            expected_hours=expected_hours,
            factors=[f for f in factors if f['signal'] != 'NEUTRAL'],
            total_score=score,
            reasoning=reasoning
        )
        
        self.prediction_history.append(prediction)
        logger.info(f"🎯 Prediction: {symbol} {direction} → ${target:,.0f} ({score:.0f}%)")
        
        return prediction
    
    async def _collect_all_data(self, symbol: str) -> Optional[Dict]:
        """Tüm gerekli verileri topla."""
        try:
            result = {}
            
            # 1. Fiyat verileri (klines)
            kline_resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1h', 'limit': 100},
                timeout=15
            )
            if kline_resp.status_code != 200:
                return None
            
            klines = kline_resp.json()
            closes = np.array([float(k[4]) for k in klines])
            volumes = np.array([float(k[5]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            
            result['closes'] = closes
            result['volumes'] = volumes
            result['highs'] = highs
            result['lows'] = lows
            result['current_price'] = closes[-1]
            
            # 2. Funding rate
            try:
                fr_resp = requests.get(
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                    params={'symbol': symbol, 'limit': 1},
                    timeout=5
                )
                if fr_resp.status_code == 200 and fr_resp.json():
                    result['funding_rate'] = float(fr_resp.json()[0]['fundingRate']) * 100
                else:
                    result['funding_rate'] = 0
            except:
                result['funding_rate'] = 0
            
            # 3. Long/Short ratio
            try:
                ls_resp = requests.get(
                    "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                    params={'symbol': symbol, 'period': '1h', 'limit': 1},
                    timeout=5
                )
                if ls_resp.status_code == 200 and ls_resp.json():
                    result['ls_ratio'] = float(ls_resp.json()[0]['longShortRatio'])
                else:
                    result['ls_ratio'] = 1.0
            except:
                result['ls_ratio'] = 1.0
            
            # 4. Fear & Greed
            try:
                fg_resp = requests.get(
                    "https://api.alternative.me/fng/?limit=1",
                    timeout=5
                )
                if fg_resp.status_code == 200:
                    result['fear_greed'] = int(fg_resp.json()['data'][0]['value'])
                else:
                    result['fear_greed'] = 50
            except:
                result['fear_greed'] = 50
            
            # 5. Order book (whale detection)
            try:
                depth_resp = requests.get(
                    "https://api.binance.com/api/v3/depth",
                    params={'symbol': symbol, 'limit': 100},
                    timeout=5
                )
                if depth_resp.status_code == 200:
                    depth = depth_resp.json()
                    bid_vol = sum(float(b[1]) for b in depth['bids'])
                    ask_vol = sum(float(a[1]) for a in depth['asks'])
                    result['bid_volume'] = bid_vol
                    result['ask_volume'] = ask_vol
                    result['orderbook_imbalance'] = bid_vol / (ask_vol + 0.001)
                else:
                    result['orderbook_imbalance'] = 1.0
            except:
                result['orderbook_imbalance'] = 1.0
            
            return result
            
        except Exception as e:
            logger.error(f"Data collection error: {e}")
            return None
    
    async def _analyze_all_factors(self, symbol: str, data: Dict) -> List[Dict]:
        """Her faktörü analiz et ve sinyal üret."""
        factors = []
        closes = data['closes']
        volumes = data['volumes']
        
        # 1. LSTM Prediction
        lstm_factor = await self._analyze_lstm(symbol, data)
        factors.append(lstm_factor)
        
        # 2. Squeeze Direction (Historical pattern)
        squeeze_factor = self._analyze_squeeze_history(symbol, data)
        factors.append(squeeze_factor)
        
        # 3. Funding Rate (Contrarian)
        funding_factor = self._analyze_funding(data['funding_rate'])
        factors.append(funding_factor)
        
        # 4. Fear & Greed (Contrarian)
        fg_factor = self._analyze_fear_greed(data['fear_greed'])
        factors.append(fg_factor)
        
        # 5. Whale Activity (Orderbook)
        whale_factor = self._analyze_whale_activity(data)
        factors.append(whale_factor)
        
        # 6. Volume Trend
        volume_factor = self._analyze_volume_trend(volumes)
        factors.append(volume_factor)
        
        # 7. RSI Divergence
        rsi_factor = self._analyze_rsi(closes)
        factors.append(rsi_factor)
        
        # 8. EMA Alignment
        ema_factor = self._analyze_ema_alignment(closes)
        factors.append(ema_factor)
        
        # 9. L/S Ratio (Contrarian)
        ls_factor = self._analyze_ls_ratio(data['ls_ratio'])
        factors.append(ls_factor)
        
        return factors
    
    async def _analyze_lstm(self, symbol: str, data: Dict) -> Dict:
        """LSTM model tahmini."""
        try:
            from src.brain.models.lstm_trend import LSTMTrendPredictor
            
            model = LSTMTrendPredictor(symbol=symbol)
            
            if not model.trained:
                return {
                    'name': 'lstm_prediction',
                    'signal': 'NEUTRAL',
                    'weight': self.WEIGHTS['lstm_prediction'],
                    'value': 0,
                    'description': 'Model not trained'
                }
            
            # Prepare DataFrame
            df = pd.DataFrame({
                'close': data['closes'],
                'volume': data['volumes'],
                'high': data['highs'],
                'low': data['lows']
            })
            
            pred = model.predict(df)
            direction = pred.get('direction', 'NEUTRAL')
            confidence = pred.get('confidence', 0)
            
            signal = 'LONG' if direction == 'UP' else 'SHORT' if direction == 'DOWN' else 'NEUTRAL'
            
            return {
                'name': 'lstm_prediction',
                'signal': signal,
                'weight': self.WEIGHTS['lstm_prediction'],
                'value': confidence,
                'description': f'LSTM: {direction} %{confidence:.0f}'
            }
            
        except Exception as e:
            logger.debug(f"LSTM analysis failed: {e}")
            return {
                'name': 'lstm_prediction',
                'signal': 'NEUTRAL',
                'weight': self.WEIGHTS['lstm_prediction'],
                'value': 0,
                'description': f'LSTM error: {e}'
            }
    
    def _analyze_squeeze_history(self, symbol: str, data: Dict) -> Dict:
        """
        Squeeze sonrası tarihsel olarak hangi yöne gitti?
        Son 50 squeeze'in %60'ından fazlası bir yöne gittiyse o yön
        """
        closes = data['closes']
        
        # Bollinger hesapla
        sma = np.mean(closes[-20:])
        std = np.std(closes[-20:])
        bandwidth = (2 * std / sma) * 100 if sma > 0 else 5
        
        # Squeeze var mı?
        if bandwidth > 2.0:
            return {
                'name': 'squeeze_direction',
                'signal': 'NEUTRAL',
                'weight': self.WEIGHTS['squeeze_direction'],
                'value': bandwidth,
                'description': f'No squeeze (BW: {bandwidth:.1f}%)'
            }
        
        # Squeeze var - tarihsel pattern'e bak
        # Son 24 saatlik momentum
        momentum_24h = (closes[-1] / closes[-24] - 1) * 100 if len(closes) >= 24 else 0
        
        # Son 4 saatlik momentum (daha kısa vadeli)
        momentum_4h = (closes[-1] / closes[-4] - 1) * 100 if len(closes) >= 4 else 0
        
        # Hacim trendi (artan hacim = patlama yakın)
        vol_recent = np.mean(data['volumes'][-4:])
        vol_avg = np.mean(data['volumes'][-24:-4])
        vol_ratio = vol_recent / (vol_avg + 0.001)
        
        # Karar
        bullish_signs = 0
        bearish_signs = 0
        
        if momentum_4h > 0.2:
            bullish_signs += 1
        elif momentum_4h < -0.2:
            bearish_signs += 1
        
        if momentum_24h > 0:
            bullish_signs += 1
        elif momentum_24h < 0:
            bearish_signs += 1
        
        # Fiyat pozisyonu (SMA'nın üstünde/altında)
        if closes[-1] > sma:
            bullish_signs += 1
        else:
            bearish_signs += 1
        
        if bullish_signs > bearish_signs:
            signal = 'LONG'
            confidence = 60 + (bullish_signs * 10)
        elif bearish_signs > bullish_signs:
            signal = 'SHORT'
            confidence = 60 + (bearish_signs * 10)
        else:
            signal = 'NEUTRAL'
            confidence = 50
        
        return {
            'name': 'squeeze_direction',
            'signal': signal,
            'weight': self.WEIGHTS['squeeze_direction'],
            'value': confidence,
            'description': f'Squeeze->{signal} (BW:{bandwidth:.1f}%, Mom:{momentum_4h:+.1f}%)'
        }
    
    def _analyze_funding(self, funding_rate: float) -> Dict:
        """Funding rate analizi (contrarian)."""
        # Yüksek pozitif funding = çok long = SHORT sinyali
        # Yüksek negatif funding = çok short = LONG sinyali
        
        if funding_rate > 0.05:
            signal = 'SHORT'
            strength = min(90, 60 + abs(funding_rate) * 300)
            desc = f'High funding {funding_rate:.3f}% → Longs overleveraged'
        elif funding_rate < -0.03:
            signal = 'LONG'
            strength = min(90, 60 + abs(funding_rate) * 400)
            desc = f'Negative funding {funding_rate:.3f}% → Shorts paying'
        else:
            signal = 'NEUTRAL'
            strength = 50
            desc = f'Neutral funding {funding_rate:.3f}%'
        
        return {
            'name': 'funding_rate',
            'signal': signal,
            'weight': self.WEIGHTS['funding_rate'],
            'value': strength,
            'description': desc
        }
    
    def _analyze_fear_greed(self, fg_index: int) -> Dict:
        """Fear & Greed analizi (contrarian)."""
        # Extreme Fear = LONG (contrarian)
        # Extreme Greed = SHORT (contrarian)
        
        if fg_index <= 20:
            signal = 'LONG'
            strength = 85
            desc = f'Extreme Fear ({fg_index}) → Contrarian LONG'
        elif fg_index <= 35:
            signal = 'LONG'
            strength = 70
            desc = f'Fear ({fg_index}) → Mild LONG bias'
        elif fg_index >= 80:
            signal = 'SHORT'
            strength = 85
            desc = f'Extreme Greed ({fg_index}) → Contrarian SHORT'
        elif fg_index >= 65:
            signal = 'SHORT'
            strength = 70
            desc = f'Greed ({fg_index}) → Mild SHORT bias'
        else:
            signal = 'NEUTRAL'
            strength = 50
            desc = f'Neutral sentiment ({fg_index})'
        
        return {
            'name': 'fear_greed',
            'signal': signal,
            'weight': self.WEIGHTS['fear_greed'],
            'value': strength,
            'description': desc
        }
    
    def _analyze_whale_activity(self, data: Dict) -> Dict:
        """Whale aktivitesi (orderbook imbalance)."""
        imbalance = data.get('orderbook_imbalance', 1.0)
        
        if imbalance > 1.5:
            signal = 'LONG'
            strength = min(85, 60 + (imbalance - 1) * 30)
            desc = f'Whale BUY walls ({imbalance:.1f}x bid)'
        elif imbalance < 0.67:
            signal = 'SHORT'
            strength = min(85, 60 + (1/imbalance - 1) * 30)
            desc = f'Whale SELL walls ({1/imbalance:.1f}x ask)'
        else:
            signal = 'NEUTRAL'
            strength = 50
            desc = f'Balanced orderbook ({imbalance:.2f})'
        
        return {
            'name': 'whale_activity',
            'signal': signal,
            'weight': self.WEIGHTS['whale_activity'],
            'value': strength,
            'description': desc
        }
    
    def _analyze_volume_trend(self, volumes: np.ndarray) -> Dict:
        """Hacim trendi analizi."""
        recent_vol = np.mean(volumes[-6:])
        avg_vol = np.mean(volumes[-24:-6])
        
        if avg_vol == 0:
            return {
                'name': 'volume_trend',
                'signal': 'NEUTRAL',
                'weight': self.WEIGHTS['volume_trend'],
                'value': 50,
                'description': 'No volume data'
            }
        
        vol_ratio = recent_vol / avg_vol
        
        if vol_ratio > 1.5:
            # Artan hacim - yönü fiyat hareketinden belirle
            signal = 'NEUTRAL'  # Hacim tek başına yön söylemez
            strength = 70
            desc = f'Volume surge ({vol_ratio:.1f}x) → Breakout imminent'
        elif vol_ratio < 0.5:
            signal = 'NEUTRAL'
            strength = 40
            desc = f'Low volume ({vol_ratio:.1f}x) → Weak move'
        else:
            signal = 'NEUTRAL'
            strength = 50
            desc = f'Normal volume ({vol_ratio:.1f}x)'
        
        return {
            'name': 'volume_trend',
            'signal': signal,
            'weight': self.WEIGHTS['volume_trend'],
            'value': strength,
            'description': desc
        }
    
    def _analyze_rsi(self, closes: np.ndarray) -> Dict:
        """RSI analizi (oversold/overbought)."""
        delta = np.diff(closes)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        rs = avg_gain / (avg_loss + 0.001)
        rsi = 100 - (100 / (1 + rs))
        
        if rsi < 30:
            signal = 'LONG'
            strength = 75
            desc = f'RSI oversold ({rsi:.0f})'
        elif rsi > 70:
            signal = 'SHORT'
            strength = 75
            desc = f'RSI overbought ({rsi:.0f})'
        else:
            signal = 'NEUTRAL'
            strength = 50
            desc = f'RSI neutral ({rsi:.0f})'
        
        return {
            'name': 'rsi_divergence',
            'signal': signal,
            'weight': self.WEIGHTS['rsi_divergence'],
            'value': strength,
            'description': desc
        }
    
    def _analyze_ema_alignment(self, closes: np.ndarray) -> Dict:
        """EMA hizalama analizi."""
        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        ema50 = self._ema(closes, 50)
        
        current = closes[-1]
        
        # Tam bullish alignment: price > EMA9 > EMA21 > EMA50
        if current > ema9 > ema21 > ema50:
            signal = 'LONG'
            strength = 80
            desc = 'Perfect bullish EMA alignment'
        # Tam bearish alignment
        elif current < ema9 < ema21 < ema50:
            signal = 'SHORT'
            strength = 80
            desc = 'Perfect bearish EMA alignment'
        # Kısmi alignment
        elif current > ema21 and ema9 > ema21:
            signal = 'LONG'
            strength = 65
            desc = 'Partial bullish EMA'
        elif current < ema21 and ema9 < ema21:
            signal = 'SHORT'
            strength = 65
            desc = 'Partial bearish EMA'
        else:
            signal = 'NEUTRAL'
            strength = 50
            desc = 'Mixed EMA signals'
        
        return {
            'name': 'ema_alignment',
            'signal': signal,
            'weight': self.WEIGHTS['ema_alignment'],
            'value': strength,
            'description': desc
        }
    
    def _analyze_ls_ratio(self, ls_ratio: float) -> Dict:
        """Long/Short ratio (contrarian)."""
        if ls_ratio > 1.8:
            signal = 'SHORT'
            strength = 75
            desc = f'Too many longs ({ls_ratio:.2f}) → Squeeze risk'
        elif ls_ratio < 0.6:
            signal = 'LONG'
            strength = 80
            desc = f'Too many shorts ({ls_ratio:.2f}) → Short squeeze!'
        else:
            signal = 'NEUTRAL'
            strength = 50
            desc = f'Balanced L/S ({ls_ratio:.2f})'
        
        return {
            'name': 'ls_ratio',
            'signal': signal,
            'weight': self.WEIGHTS['ls_ratio'],
            'value': strength,
            'description': desc
        }
    
    def _calculate_confluence(self, factors: List[Dict]) -> Tuple[str, float, int]:
        """Confluence hesapla."""
        long_score = 0
        short_score = 0
        long_count = 0
        short_count = 0
        
        for f in factors:
            weighted_value = f['value'] * (f['weight'] / 100)
            
            if f['signal'] == 'LONG':
                long_score += weighted_value
                long_count += 1
            elif f['signal'] == 'SHORT':
                short_score += weighted_value
                short_count += 1
        
        if long_score > short_score:
            direction = 'LONG'
            score = long_score
            agreeing = long_count
        elif short_score > long_score:
            direction = 'SHORT'
            score = short_score
            agreeing = short_count
        else:
            direction = 'NEUTRAL'
            score = 50
            agreeing = 0
        
        return direction, score, agreeing
    
    def _calculate_targets(
        self, 
        current_price: float, 
        direction: str, 
        data: Dict
    ) -> Tuple[float, float, int]:
        """Hedef fiyat, stop loss ve zaman hesapla."""
        
        # ATR hesapla (volatilite bazlı)
        highs = data['highs']
        lows = data['lows']
        closes = data['closes']
        
        tr = np.maximum(highs[1:] - lows[1:], 
                       np.abs(highs[1:] - closes[:-1]),
                       np.abs(lows[1:] - closes[:-1]))
        atr = np.mean(tr[-14:])
        
        # ATR yüzdesi
        atr_pct = (atr / current_price) * 100
        
        # Hedefler: 2x ATR (R:R = 2)
        if direction == 'LONG':
            target = current_price * (1 + atr_pct * 2 / 100)
            stop_loss = current_price * (1 - atr_pct / 100)
        else:
            target = current_price * (1 - atr_pct * 2 / 100)
            stop_loss = current_price * (1 + atr_pct / 100)
        
        # Zaman tahmini: Düşük volatilite = daha uzun süre
        if atr_pct < 1.5:
            expected_hours = 12  # Düşük vol, yavaş hareket
        elif atr_pct < 3:
            expected_hours = 6   # Normal
        else:
            expected_hours = 2   # Yüksek vol, hızlı hareket
        
        return round(target, 2), round(stop_loss, 2), expected_hours
    
    def _generate_reasoning(self, direction: str, factors: List[Dict], score: float) -> str:
        """Türkçe açıklama oluştur."""
        dir_tr = 'YUKARI' if direction == 'LONG' else 'ASAGI'
        
        # En güçlü faktörler
        supporting = [f for f in factors if f['signal'] == direction]
        supporting.sort(key=lambda x: x['value'], reverse=True)
        
        lines = [f"📊 {len(supporting)} faktör {dir_tr} yönünü destekliyor:"]
        
        for f in supporting[:4]:
            emoji = '✅' if f['value'] > 70 else '◦'
            lines.append(f"  {emoji} {f['description']}")
        
        return '\n'.join(lines)
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """EMA hesapla."""
        if len(data) < period:
            return data[-1]
        
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def format_telegram(self, pred: Prediction) -> str:
        """Telegram formatında tahmin."""
        emoji = '🟢' if pred.direction == 'LONG' else '🔴'
        dir_tr = 'YUKARI' if pred.direction == 'LONG' else 'ASAGI'
        
        # Faktör listesi
        factor_lines = []
        for f in pred.factors[:5]:
            check = '✅' if f['value'] > 70 else '◦'
            factor_lines.append(f"  {check} {f['description']}")
        
        return f"""
🎯 DEMIR AI TAHMİN
━━━━━━━━━━━━━━━━━━━━━━━━
{emoji} {pred.symbol}: {dir_tr}

📍 GÜNCEL: ${pred.current_price:,.2f}
🎯 HEDEF: ${pred.target_price:,.2f}
🛑 STOP: ${pred.stop_loss:,.2f}
⏰ SÜRE: ~{pred.expected_hours} saat

📊 GÜVEN: %{pred.confidence:.0f}
({len(pred.factors)} faktör aynı yönde)
━━━━━━━━━━━━━━━━━━━━━━━━

{chr(10).join(factor_lines)}
━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {pred.timestamp.strftime('%H:%M')}
""".strip()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_engine: Optional[PredictiveEngine] = None

def get_predictive_engine() -> PredictiveEngine:
    """Get or create engine instance."""
    global _engine
    if _engine is None:
        _engine = PredictiveEngine()
    return _engine


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        engine = get_predictive_engine()
        pred = await engine.predict('BTCUSDT')
        
        if pred:
            print(engine.format_telegram(pred))
        else:
            print("No prediction (insufficient confluence)")
    
    asyncio.run(test())
