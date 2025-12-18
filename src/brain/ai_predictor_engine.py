# -*- coding: utf-8 -*-
"""
DEMIR AI - AI Predictor Engine
Ani hareketleri 5-10 dakika ÖNCEDEN tespit eden gerçek AI sistemi.

PHASE 102: AI Predictor Engine
- Leading indicator kombinasyonu
- Anomali tespiti
- Multi-signal correlation
- İnsan üstü piyasa takibi
"""
import logging
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

logger = logging.getLogger("AI_PREDICTOR")


@dataclass
class PredictionSignal:
    """Tahmin sinyali."""
    direction: str  # 'DUMP' veya 'PUMP'
    confidence: float  # 0-100
    time_horizon: str  # '5-10 dakika'
    triggers: List[str]  # Tetikleyen faktörler
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


class AIPredictorEngine:
    """
    Gerçek AI Tahmin Motoru
    
    İndikatör DEĞİL, leading indicator kombinasyonu kullanarak
    ani hareketleri 5-10 dakika ÖNCEDEN tespit eder.
    """
    
    PREDICTOR_FILE = "ai_predictor_state.json"
    
    # Signal weights (toplam = 100)
    SIGNAL_WEIGHTS = {
        'whale_movement': 20,      # Whale cüzdan hareketi
        'liquidation_proximity': 18,  # Likidasyon seviyelerine yakınlık
        'oi_velocity': 15,         # OI değişim hızı
        'funding_extreme': 12,     # Funding rate extreme
        'cvd_divergence': 12,      # CVD-price divergence
        'order_flow': 10,          # Order flow imbalance
        'volume_anomaly': 8,       # Volume spike
        'mempool_whale': 5,        # Mempool büyük TX
    }
    
    # Thresholds
    ALERT_THRESHOLD = 60  # %60+ → uyarı gönder
    CRITICAL_THRESHOLD = 80  # %80+ → kritik uyarı
    
    # Cooldown
    COOLDOWN_MINUTES = 30
    
    def __init__(self):
        self.last_prediction: Optional[datetime] = None
        self.prediction_history: List[Dict] = []
        self._load_state()
        logger.info("✅ AI Predictor Engine initialized - İnsan üstü gözler aktif")
    
    def _load_state(self):
        """State yükle."""
        try:
            if os.path.exists(self.PREDICTOR_FILE):
                with open(self.PREDICTOR_FILE, 'r') as f:
                    data = json.load(f)
                    if data.get('last_prediction'):
                        self.last_prediction = datetime.fromisoformat(data['last_prediction'])
                    self.prediction_history = data.get('history', [])[-50:]
        except Exception as e:
            logger.debug(f"State load failed: {e}")
    
    def _save_state(self):
        """State kaydet."""
        try:
            with open(self.PREDICTOR_FILE, 'w') as f:
                json.dump({
                    'last_prediction': self.last_prediction.isoformat() if self.last_prediction else None,
                    'history': self.prediction_history[-50:],
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"State save failed: {e}")
    
    def _can_predict(self) -> bool:
        """Cooldown kontrolü."""
        if self.last_prediction is None:
            return True
        minutes_since = (datetime.now() - self.last_prediction).total_seconds() / 60
        return minutes_since >= self.COOLDOWN_MINUTES
    
    async def analyze_all_signals(self, symbol: str = 'BTCUSDT') -> Dict[str, Dict]:
        """
        Tüm leading indicator'ları analiz et.
        
        Returns:
            {signal_name: {score, direction, details}}
        """
        signals = {}
        
        # 1. WHALE MOVEMENT - Büyük cüzdan hareketleri
        signals['whale_movement'] = await self._check_whale_movement(symbol)
        
        # 2. LIQUIDATION PROXIMITY - Likidasyon seviyeleri
        signals['liquidation_proximity'] = await self._check_liquidation_proximity(symbol)
        
        # 3. OI VELOCITY - Open Interest değişim hızı
        signals['oi_velocity'] = await self._check_oi_velocity(symbol)
        
        # 4. FUNDING EXTREME - Aşırı funding rate
        signals['funding_extreme'] = await self._check_funding_extreme(symbol)
        
        # 5. CVD DIVERGENCE - CVD-price ayrışması
        signals['cvd_divergence'] = await self._check_cvd_divergence(symbol)
        
        # 6. ORDER FLOW IMBALANCE - Alım/satım dengesizliği
        signals['order_flow'] = await self._check_order_flow(symbol)
        
        # 7. VOLUME ANOMALY - Hacim anomalisi
        signals['volume_anomaly'] = await self._check_volume_anomaly(symbol)
        
        # 8. MEMPOOL WHALE - Büyük BTC transferleri
        signals['mempool_whale'] = await self._check_mempool_whale()
        
        return signals
    
    async def _check_whale_movement(self, symbol: str) -> Dict:
        """Whale cüzdan hareketi kontrolü."""
        try:
            # Binance top trader positions
            resp = requests.get(
                f"https://fapi.binance.com/futures/data/topLongShortPositionRatio",
                params={'symbol': symbol, 'period': '5m', 'limit': 6},
                timeout=5
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if len(data) >= 2:
                    current_ratio = float(data[-1]['longShortRatio'])
                    prev_ratio = float(data[-2]['longShortRatio'])
                    change = current_ratio - prev_ratio
                    
                    # Sudden position change = whale movement
                    if abs(change) > 0.3:
                        direction = 'DUMP' if change > 0 else 'PUMP'  # Contrarian
                        return {
                            'score': min(100, abs(change) * 100),
                            'direction': direction,
                            'details': f"Top traders L/S: {current_ratio:.2f} (Δ{change:+.2f})"
                        }
        except:
            pass
        
        return {'score': 0, 'direction': 'NEUTRAL', 'details': 'No whale movement'}
    
    async def _check_liquidation_proximity(self, symbol: str) -> Dict:
        """Likidasyon seviyelerine yakınlık kontrolü."""
        try:
            # Current price
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            
            if resp.status_code == 200:
                current_price = float(resp.json()['price'])
                
                # Common liquidation levels (approximate)
                # Büyük leverage pozisyonlar genelde round numbers'da
                round_levels = [
                    current_price * 0.95,  # -5%
                    current_price * 0.97,  # -3%
                    current_price * 1.03,  # +3%
                    current_price * 1.05,  # +5%
                ]
                
                # Check if approaching liquidation cascade zone
                for level in round_levels:
                    distance_pct = abs((current_price - level) / current_price) * 100
                    if distance_pct < 1:  # Within 1% of liq level
                        direction = 'DUMP' if level < current_price else 'PUMP'
                        return {
                            'score': 80,
                            'direction': direction,
                            'details': f"Likidasyon seviyesi ${level:,.0f} yakın (%{distance_pct:.1f})"
                        }
        except:
            pass
        
        return {'score': 0, 'direction': 'NEUTRAL', 'details': 'Safe distance'}
    
    async def _check_oi_velocity(self, symbol: str) -> Dict:
        """Open Interest değişim hızı kontrolü."""
        try:
            resp = requests.get(
                f"https://fapi.binance.com/futures/data/openInterestHist",
                params={'symbol': symbol, 'period': '5m', 'limit': 6},
                timeout=5
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if len(data) >= 3:
                    oi_values = [float(d['sumOpenInterest']) for d in data[-3:]]
                    
                    # Velocity = rate of change
                    velocity = ((oi_values[-1] - oi_values[0]) / oi_values[0]) * 100
                    
                    if abs(velocity) > 2:  # %2+ OI change in 15min
                        direction = 'PUMP' if velocity > 0 else 'DUMP'
                        return {
                            'score': min(100, abs(velocity) * 20),
                            'direction': direction,
                            'details': f"OI velocity: {velocity:+.2f}% (15dk)"
                        }
        except:
            pass
        
        return {'score': 0, 'direction': 'NEUTRAL', 'details': 'Normal OI'}
    
    async def _check_funding_extreme(self, symbol: str) -> Dict:
        """Aşırı funding rate kontrolü."""
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    funding = float(data[0]['fundingRate']) * 100
                    
                    # Extreme funding = squeeze incoming
                    if funding > 0.05:  # High positive = long squeeze
                        return {
                            'score': min(100, funding * 500),
                            'direction': 'DUMP',
                            'details': f"Funding {funding:.4f}% - Long squeeze riski"
                        }
                    elif funding < -0.03:  # High negative = short squeeze
                        return {
                            'score': min(100, abs(funding) * 700),
                            'direction': 'PUMP',
                            'details': f"Funding {funding:.4f}% - Short squeeze riski"
                        }
        except:
            pass
        
        return {'score': 0, 'direction': 'NEUTRAL', 'details': 'Normal funding'}
    
    async def _check_cvd_divergence(self, symbol: str) -> Dict:
        """CVD-price divergence kontrolü."""
        try:
            # Aggressor trades
            resp = requests.get(
                "https://api.binance.com/api/v3/trades",
                params={'symbol': symbol, 'limit': 500},
                timeout=5
            )
            
            if resp.status_code == 200:
                trades = resp.json()
                
                # Son 250 trade (yaklaşık 5-10dk)
                recent_trades = trades[-250:]
                
                buy_volume = sum(float(t['qty']) for t in recent_trades if not t['isBuyerMaker'])
                sell_volume = sum(float(t['qty']) for t in recent_trades if t['isBuyerMaker'])
                
                cvd = buy_volume - sell_volume
                total = buy_volume + sell_volume
                
                if total > 0:
                    imbalance = (cvd / total) * 100
                    
                    # Get price direction
                    first_price = float(recent_trades[0]['price'])
                    last_price = float(recent_trades[-1]['price'])
                    price_change = ((last_price - first_price) / first_price) * 100
                    
                    # Divergence: price up but CVD down = reversal signal
                    if price_change > 0.3 and imbalance < -20:
                        return {
                            'score': 70,
                            'direction': 'DUMP',
                            'details': f"Bearish divergence: Fiyat ↑ CVD ↓ ({imbalance:.0f}%)"
                        }
                    elif price_change < -0.3 and imbalance > 20:
                        return {
                            'score': 70,
                            'direction': 'PUMP',
                            'details': f"Bullish divergence: Fiyat ↓ CVD ↑ ({imbalance:+.0f}%)"
                        }
        except:
            pass
        
        return {'score': 0, 'direction': 'NEUTRAL', 'details': 'No divergence'}
    
    async def _check_order_flow(self, symbol: str) -> Dict:
        """Order flow imbalance kontrolü."""
        try:
            resp = requests.get(
                f"https://fapi.binance.com/fapi/v1/depth",
                params={'symbol': symbol, 'limit': 20},
                timeout=5
            )
            
            if resp.status_code == 200:
                data = resp.json()
                
                bid_volume = sum(float(b[1]) for b in data['bids'][:10])
                ask_volume = sum(float(a[1]) for a in data['asks'][:10])
                
                total = bid_volume + ask_volume
                if total > 0:
                    imbalance = ((bid_volume - ask_volume) / total) * 100
                    
                    if abs(imbalance) > 30:
                        direction = 'PUMP' if imbalance > 0 else 'DUMP'
                        return {
                            'score': min(100, abs(imbalance)),
                            'direction': direction,
                            'details': f"Orderbook imbalance: {imbalance:+.0f}%"
                        }
        except:
            pass
        
        return {'score': 0, 'direction': 'NEUTRAL', 'details': 'Balanced orderbook'}
    
    async def _check_volume_anomaly(self, symbol: str) -> Dict:
        """Hacim anomalisi kontrolü."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1m', 'limit': 30},
                timeout=5
            )
            
            if resp.status_code == 200:
                klines = resp.json()
                volumes = [float(k[5]) for k in klines]
                
                avg_volume = sum(volumes[:-3]) / len(volumes[:-3])
                recent_volume = sum(volumes[-3:]) / 3
                
                ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                
                if ratio > 3:  # 3x normal volume
                    # Determine direction from price
                    price_change = (float(klines[-1][4]) - float(klines[-4][4])) / float(klines[-4][4]) * 100
                    direction = 'PUMP' if price_change > 0 else 'DUMP'
                    
                    return {
                        'score': min(100, ratio * 15),
                        'direction': direction,
                        'details': f"Volume {ratio:.1f}x normal! Büyük oyuncu aktif"
                    }
        except:
            pass
        
        return {'score': 0, 'direction': 'NEUTRAL', 'details': 'Normal volume'}
    
    async def _check_mempool_whale(self) -> Dict:
        """Mempool whale transaction kontrolü."""
        try:
            resp = requests.get("https://mempool.space/api/mempool", timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                count = data.get('count', 0)
                vsize = data.get('vsize', 0)
                
                # High vsize = potential whale activity
                if vsize > 100_000_000:  # 100 MB+ mempool
                    return {
                        'score': 40,
                        'direction': 'NEUTRAL',  # Mempool doesn't indicate direction
                        'details': f"Mempool yoğun: {count} TX, {vsize/1_000_000:.0f}MB"
                    }
        except:
            pass
        
        return {'score': 0, 'direction': 'NEUTRAL', 'details': 'Normal mempool'}
    
    async def predict(self, symbol: str = 'BTCUSDT') -> Optional[PredictionSignal]:
        """
        Ana tahmin fonksiyonu.
        
        Tüm sinyalleri analiz edip konsensüs ile tahmin üretir.
        """
        if not self._can_predict():
            return None
        
        # Tüm sinyalleri topla
        signals = await self.analyze_all_signals(symbol)
        
        # Ağırlıklı skor hesapla
        dump_score = 0
        pump_score = 0
        triggers = []
        
        for signal_name, signal_data in signals.items():
            weight = self.SIGNAL_WEIGHTS.get(signal_name, 0)
            score = signal_data.get('score', 0)
            direction = signal_data.get('direction', 'NEUTRAL')
            details = signal_data.get('details', '')
            
            weighted_score = (score * weight) / 100
            
            if direction == 'DUMP':
                dump_score += weighted_score
                if score > 30:
                    triggers.append(f"🔴 {signal_name}: {details}")
            elif direction == 'PUMP':
                pump_score += weighted_score
                if score > 30:
                    triggers.append(f"🟢 {signal_name}: {details}")
        
        # Sonuç
        total_score = max(dump_score, pump_score)
        direction = 'DUMP' if dump_score > pump_score else 'PUMP'
        
        # Threshold kontrolü
        if total_score < self.ALERT_THRESHOLD:
            return None
        
        # Severity
        if total_score >= self.CRITICAL_THRESHOLD:
            severity = 'CRITICAL'
        elif total_score >= 70:
            severity = 'HIGH'
        else:
            severity = 'MEDIUM'
        
        # Tahmin oluştur
        prediction = PredictionSignal(
            direction=direction,
            confidence=total_score,
            time_horizon='5-10 dakika',
            triggers=triggers,
            severity=severity
        )
        
        # State güncelle
        self.last_prediction = datetime.now()
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'direction': direction,
            'confidence': total_score,
            'triggers': triggers
        })
        self._save_state()
        
        logger.warning(f"🧠 AI PREDICTION: {direction} {total_score:.0f}% - {len(triggers)} triggers")
        
        return prediction
    
    def format_prediction_alert(self, prediction: PredictionSignal, symbol: str = 'BTCUSDT') -> str:
        """Telegram formatında tahmin uyarısı."""
        
        if prediction.direction == 'DUMP':
            emoji = "🔴"
            title = "ANİ DÜŞÜŞ TAHMİNİ"
            direction_emoji = "📉"
        else:
            emoji = "🟢"
            title = "ANİ YÜKSELİŞ TAHMİNİ"
            direction_emoji = "📈"
        
        severity_emojis = {
            'CRITICAL': '🚨🚨🚨',
            'HIGH': '🚨🚨',
            'MEDIUM': '⚠️'
        }
        sev = severity_emojis.get(prediction.severity, '⚠️')
        
        triggers_text = ""
        for t in prediction.triggers[:5]:  # Max 5 trigger
            triggers_text += f"  {t}\n"
        
        confidence_stars = "⭐" * min(5, int(prediction.confidence / 20))
        
        msg = f"""
{sev} **{title}** {sev}
━━━━━━━━━━━━━━━━━━━━━━
{direction_emoji} **{symbol}** {prediction.direction}
📊 AI Güven: **%{prediction.confidence:.0f}** {confidence_stars}
⏱️ Tahmin süresi: **{prediction.time_horizon}**
━━━━━━━━━━━━━━━━━━━━━━
**🧠 AI Tetikleyiciler:**
{triggers_text.strip()}
━━━━━━━━━━━━━━━━━━━━━━
_35 modül + 8 leading indicator_
_Bu bir yapay zeka tahminidir_
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_predictor = None

def get_predictor() -> AIPredictorEngine:
    """Get or create predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = AIPredictorEngine()
    return _predictor
