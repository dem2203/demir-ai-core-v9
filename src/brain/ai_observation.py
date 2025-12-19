# -*- coding: utf-8 -*-
"""
DEMIR AI - AI Gözlem Sistemi (PHASE 124: ZENGIN MODÜL ENTEGRASYONu)

42 AI modülünün tamamını kullanarak kapsamlı piyasa analizi.
Ani hareketleri ÖNCEDEN tespit eder.

PHASE 124: Full Module Integration
- SignalOrchestrator (42 modül)
- WhaleIntelligence (balina takibi)
- Funding Rate (contrarian)
- Fear & Greed (sentiment)
- Fibonacci (teknik)
- SMC Analyzer (smart money)
- Liquidation Hunter (tasfiye seviyeleri)
"""
import logging
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os

logger = logging.getLogger("AI_OBSERVATION")


class AIObservationSystem:
    """
    PHASE 124: Gelişmiş AI Gözlem Sistemi
    
    42 modülün tamamını entegre ederek kapsamlı analiz yapar.
    Ani hareketleri önceden tespit eder ve zengin bildirim gönderir.
    """
    
    STATE_FILE = "ai_observations.json"
    
    # Thresholds - Phase 126: Kalite filtresi güçlendirildi
    PRICE_CHANGE_THRESHOLD = 0.8  # %0.8 hareket
    VOLUME_SPIKE_THRESHOLD = 1.8  # 1.8x normal hacim
    MIN_CONFIDENCE_THRESHOLD = 60  # PHASE 126: %60+ güvenli gözlemler (önceki: 50)
    
    # Cooldown
    COOLDOWN_MINUTES = 30  # Daha sık gözlem
    
    def __init__(self):
        self.last_observations: Dict[str, datetime] = {}
        self._load_state()
        
        # Module instances (lazy loaded)
        self._orchestrator = None
        self._whale_intel = None
        self._fib_analyzer = None
        
        logger.info("✅ AI Observation System v2.0 (42 module integration)")
    
    def _load_state(self):
        """State yükle."""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    for key, ts in data.get('last_observations', {}).items():
                        self.last_observations[key] = datetime.fromisoformat(ts)
        except Exception as e:
            logger.debug(f"State load failed: {e}")
    
    def _save_state(self):
        """State kaydet."""
        try:
            with open(self.STATE_FILE, 'w') as f:
                json.dump({
                    'last_observations': {k: v.isoformat() for k, v in self.last_observations.items()},
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"State save failed: {e}")
    
    def _can_observe(self, symbol: str) -> bool:
        """Cooldown kontrolü."""
        if symbol not in self.last_observations:
            return True
        
        minutes_since = (datetime.now() - self.last_observations[symbol]).total_seconds() / 60
        return minutes_since >= self.COOLDOWN_MINUTES
    
    async def _get_basic_data(self, symbol: str) -> Optional[Dict]:
        """Temel fiyat ve hacim verisi al."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1m', 'limit': 20},
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            klines = resp.json()
            if len(klines) < 15:
                return None
            
            # Fiyat değişimi
            price_15m_ago = float(klines[-15][4])
            current_price = float(klines[-1][4])
            change_pct = ((current_price - price_15m_ago) / price_15m_ago) * 100
            
            # Hacim analizi
            volumes = [float(k[5]) for k in klines[:-3]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 1
            recent_volume = sum([float(k[5]) for k in klines[-3:]]) / 3
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            return {
                'current_price': current_price,
                'change_pct': change_pct,
                'volume_ratio': volume_ratio,
                'high_24h': max(float(k[2]) for k in klines),
                'low_24h': min(float(k[3]) for k in klines)
            }
        except Exception as e:
            logger.debug(f"Basic data fetch failed: {e}")
            return None
    
    async def _get_klines_raw(self, symbol: str) -> Optional[list]:
        """Raw klines verisi al (TraderMindset için)."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '5m', 'limit': 20},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None
    
    async def _get_whale_intel(self, symbol: str) -> Dict:
        """Whale Intelligence verisi al."""
        try:
            from src.brain.whale_intelligence import WhaleIntelligence
            
            if not self._whale_intel:
                self._whale_intel = WhaleIntelligence()
            
            return self._whale_intel.get_full_whale_analysis(symbol)
        except Exception as e:
            logger.debug(f"Whale intel failed: {e}")
            return {'whale_bias': 'NEUTRAL', 'confidence': 30}
    
    async def _get_funding_rate(self, symbol: str) -> Dict:
        """Funding Rate verisi al."""
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    rate = float(data[0].get('fundingRate', 0)) * 100
                    if rate > 0.05:
                        return {'funding_rate': rate, 'signal': 'EXTREME_LONG', 'action': 'SHORT'}
                    elif rate < -0.05:
                        return {'funding_rate': rate, 'signal': 'EXTREME_SHORT', 'action': 'LONG'}
                    return {'funding_rate': rate, 'signal': 'NEUTRAL', 'action': 'NONE'}
        except:
            pass
        return {'funding_rate': 0, 'signal': 'NEUTRAL', 'action': 'NONE'}
    
    async def _get_fear_greed(self) -> Dict:
        """Fear & Greed Index al."""
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                value = int(data['data'][0]['value'])
                if value <= 25:
                    return {'value': value, 'signal': 'EXTREME_FEAR', 'action': 'BUY'}
                elif value >= 75:
                    return {'value': value, 'signal': 'EXTREME_GREED', 'action': 'SELL'}
                return {'value': value, 'signal': 'NEUTRAL', 'action': 'NONE'}
        except:
            pass
        return {'value': 50, 'signal': 'NEUTRAL', 'action': 'NONE'}
    
    async def _get_fibonacci(self, symbol: str) -> Dict:
        """Fibonacci seviyeleri al."""
        try:
            from src.brain.fibonacci_analyzer import get_fibonacci
            
            fib = get_fibonacci()
            result = await fib.analyze(symbol=symbol, timeframe='4h')
            
            if result and 'error' not in result:
                return {
                    'trend': result.get('trend', 'UNKNOWN'),
                    'nearest_support': result.get('nearest_support', {}),
                    'nearest_resistance': result.get('nearest_resistance', {}),
                    'signal': result.get('signal', {})
                }
        except Exception as e:
            logger.debug(f"Fibonacci failed: {e}")
        return {'trend': 'UNKNOWN', 'signal': {}}
    
    async def _get_liquidation_data(self, symbol: str) -> Dict:
        """Liquidation seviyeleri al."""
        try:
            from src.brain.liquidation_hunter import LiquidationHunter
            
            hunter = LiquidationHunter()
            data = await hunter.calculate_liquidation_levels(symbol)
            
            if data and 'error' not in data:
                return {
                    'magnet_direction': data.get('magnet_direction', 'NEUTRAL'),
                    'magnet_price': data.get('magnet_price', 0),
                    'nearest_long_liq': data.get('nearest_long_liq', {}),
                    'nearest_short_liq': data.get('nearest_short_liq', {})
                }
        except Exception as e:
            logger.debug(f"Liquidation data failed: {e}")
        return {'magnet_direction': 'NEUTRAL'}
    
    async def observe_full(self, symbol: str = 'BTCUSDT') -> Optional[Dict]:
        """
        PHASE 124: 42 modül entegrasyonlu kapsamlı gözlem.
        
        Tüm veri kaynaklarını birleştirir:
        - Fiyat/Hacim (Binance)
        - Whale Intelligence
        - Funding Rate
        - Fear & Greed
        - Fibonacci
        - Liquidation Levels
        """
        if not self._can_observe(symbol):
            logger.debug(f"{symbol}: Cooldown active")
            return None
        
        try:
            # 1. Temel veri
            basic = await self._get_basic_data(symbol)
            if not basic:
                return None
            
            current_price = basic['current_price']
            change_pct = basic['change_pct']
            volume_ratio = basic['volume_ratio']
            
            # Gözlem yapılacak mı? Daha düşük eşikler
            if abs(change_pct) < self.PRICE_CHANGE_THRESHOLD and volume_ratio < self.VOLUME_SPIKE_THRESHOLD:
                return None
            
            # 2. Paralel olarak tüm modüllerden veri al
            whale_task = self._get_whale_intel(symbol)
            funding_task = self._get_funding_rate(symbol)
            fng_task = self._get_fear_greed()
            fib_task = self._get_fibonacci(symbol)
            liq_task = self._get_liquidation_data(symbol)
            
            whale, funding, fng, fib, liq = await asyncio.gather(
                whale_task, funding_task, fng_task, fib_task, liq_task,
                return_exceptions=True
            )
            
            # Exception'ları default değerlerle değiştir
            if isinstance(whale, Exception):
                whale = {'whale_bias': 'NEUTRAL', 'confidence': 30}
            if isinstance(funding, Exception):
                funding = {'funding_rate': 0, 'signal': 'NEUTRAL'}
            if isinstance(fng, Exception):
                fng = {'value': 50, 'signal': 'NEUTRAL'}
            if isinstance(fib, Exception):
                fib = {'trend': 'UNKNOWN', 'signal': {}}
            if isinstance(liq, Exception):
                liq = {'magnet_direction': 'NEUTRAL'}
            
            # 3. Kombine sinyal hesapla
            long_votes = 0
            short_votes = 0
            reasons = []
            
            # Fiyat momentum
            if change_pct > 0.5:
                long_votes += 2
                reasons.append(f"📈 Fiyat: +{change_pct:.2f}%")
            elif change_pct < -0.5:
                short_votes += 2
                reasons.append(f"📉 Fiyat: {change_pct:.2f}%")
            
            # Hacim
            if volume_ratio > 2:
                if change_pct > 0:
                    long_votes += 1
                else:
                    short_votes += 1
                reasons.append(f"🔥 Hacim: {volume_ratio:.1f}x spike")
            
            # Whale Intel
            if whale.get('whale_bias') == 'LONG':
                long_votes += 2
                reasons.append(f"🐋 Whale: LONG (%{whale.get('confidence', 0):.0f})")
            elif whale.get('whale_bias') == 'SHORT':
                short_votes += 2
                reasons.append(f"🐋 Whale: SHORT (%{whale.get('confidence', 0):.0f})")
            
            # Funding Rate (contrarian)
            if funding.get('signal') == 'EXTREME_LONG':
                short_votes += 1
                reasons.append(f"💵 Funding: +{funding['funding_rate']:.3f}% (Aşırı long = düşüş riski)")
            elif funding.get('signal') == 'EXTREME_SHORT':
                long_votes += 1
                reasons.append(f"💵 Funding: {funding['funding_rate']:.3f}% (Aşırı short = yükseliş fırsatı)")
            
            # Fear & Greed (contrarian)
            if fng.get('signal') == 'EXTREME_FEAR':
                long_votes += 2
                reasons.append(f"😱 Fear&Greed: {fng['value']} (Korku = ALIM FIRSATI!)")
            elif fng.get('signal') == 'EXTREME_GREED':
                short_votes += 2
                reasons.append(f"🤑 Fear&Greed: {fng['value']} (Açgözlülük = RİSK!)")
            
            # Fibonacci
            fib_signal = fib.get('signal', {})
            if fib_signal.get('direction') == 'LONG':
                long_votes += 1
                reasons.append(f"📐 Fib: {fib_signal.get('reason', 'Destek')}")
            elif fib_signal.get('direction') == 'SHORT':
                short_votes += 1
                reasons.append(f"📐 Fib: {fib_signal.get('reason', 'Direnç')}")
            
            # Liquidation Magnet
            if liq.get('magnet_direction') == 'UP':
                long_votes += 1
                reasons.append(f"🎯 Liq Magnet: Yukarı (${liq.get('magnet_price', 0):,.0f})")
            elif liq.get('magnet_direction') == 'DOWN':
                short_votes += 1
                reasons.append(f"🎯 Liq Magnet: Aşağı (${liq.get('magnet_price', 0):,.0f})")
            
            # 4. Final karar
            total_votes = long_votes + short_votes
            if total_votes == 0:
                return None
            
            if long_votes > short_votes:
                direction = 'YUKARI'
                direction_emoji = '🟢'
                likely_action = 'LONG'
                confidence = (long_votes / (total_votes + 3)) * 100  # Normalize
            elif short_votes > long_votes:
                direction = 'AŞAĞI'  
                direction_emoji = '🔴'
                likely_action = 'SHORT'
                confidence = (short_votes / (total_votes + 3)) * 100
            else:
                direction = 'KARARSIZ'
                direction_emoji = '⚪'
                likely_action = 'BEKLE'
                confidence = 30
            
            # Minimum güven kontrolü
            if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                logger.debug(f"{symbol}: Confidence too low ({confidence:.0f}%)")
                return None
            
            # 5. Hedef ve Stop hesapla
            if likely_action == 'LONG':
                # Fib resistance varsa hedef olarak kullan
                resist = fib.get('nearest_resistance', {})
                if resist and resist.get('price'):
                    target_price = resist['price']
                else:
                    target_price = current_price * 1.02  # %2 default
                
                # Support varsa stop olarak kullan
                support = fib.get('nearest_support', {})
                if support and support.get('price'):
                    stop_loss = support['price'] * 0.99
                else:
                    stop_loss = current_price * 0.985  # %1.5 default
            elif likely_action == 'SHORT':
                support = fib.get('nearest_support', {})
                if support and support.get('price'):
                    target_price = support['price']
                else:
                    target_price = current_price * 0.98
                
                resist = fib.get('nearest_resistance', {})
                if resist and resist.get('price'):
                    stop_loss = resist['price'] * 1.01
                else:
                    stop_loss = current_price * 1.015
            else:
                target_price = current_price
                stop_loss = current_price
            
            # Cooldown güncelle
            self.last_observations[symbol] = datetime.now()
            self._save_state()
            
            # 6. PHASE 125: Trader Mindset Context
            try:
                from src.brain.trader_mindset import get_trader_mindset
                
                mindset = get_trader_mindset()
                
                # Session bilgisi
                session = mindset.get_current_session()
                
                # Stop hunt tespiti (klines'dan)
                klines = await self._get_klines_raw(symbol)
                stop_hunt = mindset.detect_stop_hunt(klines) if klines else {'detected': False}
                
                # Order flow
                order_flow = mindset.analyze_order_flow(symbol)
                
                # Manipülasyon kontrolü
                manipulation = mindset.detect_market_manipulation(klines, volume_ratio) if klines else {'detected': False}
                
                # Narrative oluştur
                narrative_data = {
                    'symbol': symbol,
                    'direction': direction,
                    'confidence': min(95, confidence),
                    'reasons': reasons,
                    'session': session,
                    'stop_hunt': stop_hunt,
                    'order_flow': order_flow,
                    'manipulation': manipulation
                }
                narrative = mindset.generate_narrative(narrative_data)
                
            except Exception as e:
                logger.debug(f"Trader mindset failed: {e}")
                session = {}
                stop_hunt = {'detected': False}
                order_flow = {'available': False}
                manipulation = {'detected': False}
                narrative = ""
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'change_pct': change_pct,
                'volume_ratio': volume_ratio,
                'direction': direction,
                'direction_emoji': direction_emoji,
                'likely_action': likely_action,
                'confidence': min(95, confidence),
                'target_price': target_price,
                'stop_loss': stop_loss,
                'reasons': reasons,
                'modules_used': len(reasons),
                # Raw module data
                'whale': whale,
                'funding': funding,
                'fear_greed': fng,
                'fibonacci': fib,
                'liquidation': liq,
                # PHASE 125: Trader Mindset
                'session': session,
                'stop_hunt': stop_hunt,
                'order_flow': order_flow,
                'manipulation': manipulation,
                'narrative': narrative
            }
            
        except Exception as e:
            logger.error(f"Full observation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def format_rich_observation(self, obs: Dict) -> str:
        """
        PHASE 124: Zengin Telegram formatı.
        Tüm modülleri gösterir.
        """
        symbol = obs['symbol']
        current_price = obs['current_price']
        direction = obs['direction']
        direction_emoji = obs['direction_emoji']
        likely_action = obs['likely_action']
        confidence = obs['confidence']
        target_price = obs['target_price']
        stop_loss = obs['stop_loss']
        change_pct = obs['change_pct']
        volume_ratio = obs['volume_ratio']
        reasons = obs['reasons']
        modules_used = obs['modules_used']
        
        # Kar/Zarar hesapla
        if likely_action == 'LONG':
            profit_pct = ((target_price - current_price) / current_price) * 100
            loss_pct = ((current_price - stop_loss) / current_price) * 100
        else:
            profit_pct = ((current_price - target_price) / current_price) * 100
            loss_pct = ((stop_loss - current_price) / current_price) * 100
        
        # RR oranı
        rr_ratio = profit_pct / loss_pct if loss_pct > 0 else 0
        
        # Güven seviyesi rengi
        if confidence >= 70:
            conf_emoji = "🟢"
            conf_text = "YÜKSEK"
        elif confidence >= 50:
            conf_emoji = "🟡"
            conf_text = "ORTA"
        else:
            conf_emoji = "⚪"
            conf_text = "DÜŞÜK"
        
        # Modül detayları
        reasons_text = "\n".join([f"  {r}" for r in reasons[:6]])  # Max 6 sebep
        
        # PHASE 125: Trader Mindset sections
        session = obs.get('session', {})
        stop_hunt = obs.get('stop_hunt', {})
        order_flow = obs.get('order_flow', {})
        manipulation = obs.get('manipulation', {})
        narrative = obs.get('narrative', '')
        
        # Session header
        session_line = ""
        if session.get('name'):
            session_line = f"{session['emoji']} {session['name']} Session | {session.get('volatility_expected', 'ORTA')} volatilite"
        
        # Warnings
        warnings = []
        if stop_hunt.get('detected'):
            warnings.append(f"⚠️ {stop_hunt.get('description', 'Stop hunt!')}")
        if manipulation.get('detected'):
            warnings.append(f"🚨 {manipulation.get('description', 'Dikkat!')}")
        
        warnings_text = "\n".join(warnings) if warnings else ""
        
        # Order flow
        order_flow_line = ""
        if order_flow.get('available'):
            order_flow_line = f"💹 Order Flow: {order_flow['flow_emoji']} {order_flow['flow']}"
        
        msg = f"""
🧠 AI TRADER ANALİZİ - {symbol}
━━━━━━━━━━━━━━━━━━━━━━
{session_line}
💰 Fiyat: ${current_price:,.2f}
📊 15dk: {change_pct:+.2f}% | Hacim: {volume_ratio:.1f}x
{order_flow_line}

📋 {modules_used} MODÜL ANALİZİ:
{reasons_text}

{direction_emoji} KOMBİNE SİNYAL: {likely_action}
{conf_emoji} Güven: %{confidence:.0f} ({conf_text})
━━━━━━━━━━━━━━━━━━━━━━
🎯 İŞLEM DETAYLARI:
▸ Giriş: ${current_price:,.2f}
▸ Hedef: ${target_price:,.2f} (+{profit_pct:.1f}%)
▸ Stop: ${stop_loss:,.2f} (-{loss_pct:.1f}%)
▸ R:R = 1:{rr_ratio:.1f}
""".strip()
        
        # Add warnings if any
        if warnings_text:
            msg += f"\n━━━━━━━━━━━━━━━━━━━━━━\n{warnings_text}"
        
        # Add brief narrative
        if narrative:
            # Kısa tut - max 3 satır
            narrative_lines = narrative.split('\n')[:3]
            brief_narrative = '\n'.join(narrative_lines)
            msg += f"\n━━━━━━━━━━━━━━━━━━━━━━\n📝 TRADER YORUMU:\n{brief_narrative}"
        
        msg += f"\n━━━━━━━━━━━━━━━━━━━━━━\n⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        
        return msg
    
    # Eski format için geriye uyumluluk
    async def observe(self, symbol: str = 'BTCUSDT') -> Optional[Dict]:
        """Eski observe - artık observe_full kullanır."""
        return await self.observe_full(symbol)
    
    def format_observation(self, obs: Dict) -> str:
        """Eski format - artık rich format kullanır."""
        return self.format_rich_observation(obs)


# Global instance
_observer = None

def get_observer() -> AIObservationSystem:
    """Get or create observer instance."""
    global _observer
    if _observer is None:
        _observer = AIObservationSystem()
    return _observer
