# -*- coding: utf-8 -*-
"""
DEMIR AI - Trader Mindset Engine
Piyasayı bir trader gibi yorumlar.

PHASE 125: Trader-Like Analysis
- Session Analysis (Asya/Avrupa/NY)
- Stop Hunt Detection
- Order Flow Reading
- Market Narrative Generation
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
import requests

logger = logging.getLogger("TRADER_MINDSET")


class TraderMindset:
    """
    Trader Düşünce Motoru
    
    Matematiksel analizi insan trader yorumuna dönüştürür.
    """
    
    # Trading Sessions (UTC)
    SESSIONS = {
        'ASIA': {'start': 0, 'end': 8, 'name': 'Asya', 'emoji': '🌏', 
                 'character': 'Düşük hacim, yavaş hareket, range-bound'},
        'EUROPE': {'start': 7, 'end': 16, 'name': 'Avrupa', 'emoji': '🌍',
                   'character': 'Orta hacim, trend başlangıçları'},
        'US': {'start': 13, 'end': 22, 'name': 'Amerika', 'emoji': '🌎',
               'character': 'Yüksek hacim, volatil, trend devamı veya kırılma'},
        'OVERLAP_EU_US': {'start': 13, 'end': 16, 'name': 'Avrupa/ABD Kesişimi', 'emoji': '⚡',
                         'character': 'EN YÜKSEK VOLATİLİTE - büyük hareketler'}
    }
    
    def __init__(self):
        logger.info("✅ Trader Mindset Engine initialized")
    
    def get_current_session(self) -> Dict:
        """Şu anki trading session'ı belirle."""
        utc_hour = datetime.now(timezone.utc).hour
        
        # Overlap kontrolü önce
        if 13 <= utc_hour < 16:
            return {
                **self.SESSIONS['OVERLAP_EU_US'],
                'is_overlap': True,
                'volatility_expected': 'YÜKSEK',
                'recommendation': 'Büyük hareketler beklenir - dikkatli ol veya fırsat kovala'
            }
        
        # Tek session
        if 0 <= utc_hour < 8:
            session = self.SESSIONS['ASIA']
            volatility = 'DÜŞÜK'
            rec = 'Range trading uygundur, breakout bekleme'
        elif 7 <= utc_hour < 16:
            session = self.SESSIONS['EUROPE']
            volatility = 'ORTA'
            rec = 'Trend başlangıçlarına dikkat et'
        else:
            session = self.SESSIONS['US']
            volatility = 'YÜKSEK'
            rec = 'Momentum takibi yap, büyük haberler olabilir'
        
        return {
            **session,
            'is_overlap': False,
            'volatility_expected': volatility,
            'recommendation': rec,
            'current_hour_utc': utc_hour
        }
    
    def detect_stop_hunt(self, klines: list) -> Dict:
        """
        Stop Hunt Tespiti
        
        Belirti: Ani spike (wick) + hızlı geri dönüş
        Bu genellikle market maker'ların stop'ları tetiklemesi
        """
        if len(klines) < 5:
            return {'detected': False}
        
        try:
            # Son 5 mum analizi
            recent = klines[-5:]
            
            for i, candle in enumerate(recent):
                open_p = float(candle[1])
                high = float(candle[2])
                low = float(candle[3])
                close = float(candle[4])
                
                body = abs(close - open_p)
                upper_wick = high - max(open_p, close)
                lower_wick = min(open_p, close) - low
                
                # Wick/Body oranı - büyük wick stop hunt işareti
                if body > 0:
                    upper_ratio = upper_wick / body
                    lower_ratio = lower_wick / body
                    
                    # Üst wick çok büyük = Yukarı stop hunt, sonra düşüş
                    if upper_ratio > 3:
                        return {
                            'detected': True,
                            'type': 'UPPER_HUNT',
                            'description': '⚠️ Yukarı stop hunt tespit! Long stop\'lar tetiklendi, geri çekilme beklenir',
                            'action': 'SHORT fırsatı olabilir',
                            'candle_index': i
                        }
                    
                    # Alt wick çok büyük = Aşağı stop hunt, sonra yükseliş
                    if lower_ratio > 3:
                        return {
                            'detected': True,
                            'type': 'LOWER_HUNT',
                            'description': '⚠️ Aşağı stop hunt tespit! Short stop\'lar tetiklendi, toparlanma beklenir',
                            'action': 'LONG fırsatı olabilir',
                            'candle_index': i
                        }
            
            return {'detected': False}
            
        except Exception as e:
            logger.debug(f"Stop hunt detection failed: {e}")
            return {'detected': False}
    
    def analyze_order_flow(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Order Flow Analizi
        
        Bid/Ask imbalance, taker buy/sell ratio
        """
        try:
            # 24h ticker stats
            resp = requests.get(
                f"https://api.binance.com/api/v3/ticker/24hr",
                params={'symbol': symbol},
                timeout=5
            )
            
            if resp.status_code != 200:
                return {'available': False}
            
            data = resp.json()
            
            # Volume analysis
            total_volume = float(data.get('volume', 0))
            quote_volume = float(data.get('quoteVolume', 0))
            
            # Trades count
            trades = int(data.get('count', 0))
            
            # Futures taker buy/sell ratio
            try:
                futures_resp = requests.get(
                    "https://fapi.binance.com/futures/data/takerlongshortRatio",
                    params={'symbol': symbol, 'period': '5m', 'limit': 1},
                    timeout=5
                )
                if futures_resp.status_code == 200:
                    futures_data = futures_resp.json()
                    if futures_data:
                        buy_sell_ratio = float(futures_data[0].get('buySellRatio', 1))
                    else:
                        buy_sell_ratio = 1
                else:
                    buy_sell_ratio = 1
            except:
                buy_sell_ratio = 1
            
            # Yorumla
            if buy_sell_ratio > 1.3:
                flow = 'ALICI BASKISI'
                flow_emoji = '🟢'
                interpretation = 'Taker alımları satımları geçiyor - yükseliş momentumu'
            elif buy_sell_ratio < 0.7:
                flow = 'SATICI BASKISI'
                flow_emoji = '🔴'
                interpretation = 'Taker satışları alımları geçiyor - düşüş momentumu'
            else:
                flow = 'DENGELİ'
                flow_emoji = '⚪'
                interpretation = 'Alıcı ve satıcılar dengede - yön belirsiz'
            
            return {
                'available': True,
                'buy_sell_ratio': buy_sell_ratio,
                'flow': flow,
                'flow_emoji': flow_emoji,
                'interpretation': interpretation,
                'total_trades_24h': trades,
                'quote_volume_24h': quote_volume
            }
            
        except Exception as e:
            logger.debug(f"Order flow analysis failed: {e}")
            return {'available': False}
    
    def detect_market_manipulation(self, klines: list, volume_ratio: float) -> Dict:
        """
        Piyasa Manipülasyonu Tespiti
        
        Belirtiler:
        - Ani hacim spike + küçük fiyat hareketi = wash trading
        - Büyük fiyat hareketi + düşük hacim = thin orderbook exploit
        - Cascade likidasyonları
        """
        if len(klines) < 10:
            return {'detected': False}
        
        try:
            # Son mumların analizi
            price_changes = []
            for i in range(-5, 0):
                open_p = float(klines[i][1])
                close = float(klines[i][4])
                change = ((close - open_p) / open_p) * 100
                price_changes.append(change)
            
            avg_change = sum(abs(c) for c in price_changes) / len(price_changes)
            
            # Anomali tespiti
            if volume_ratio > 3 and avg_change < 0.3:
                return {
                    'detected': True,
                    'type': 'WASH_TRADING',
                    'description': '🚨 Olası wash trading! Yüksek hacim ama düşük fiyat hareketi',
                    'severity': 'ORTA',
                    'recommendation': 'Dikkatli ol, gerçek talep olmayabilir'
                }
            
            if volume_ratio < 0.5 and avg_change > 2:
                return {
                    'detected': True,
                    'type': 'THIN_BOOK_EXPLOIT',
                    'description': '🚨 İnce orderbook kullanımı! Düşük hacim ama büyük fiyat hareketi',
                    'severity': 'YÜKSEK',
                    'recommendation': 'Fiyat hızla geri dönebilir, dikkat!'
                }
            
            return {'detected': False}
            
        except:
            return {'detected': False}
    
    def generate_narrative(self, obs_data: Dict) -> str:
        """
        Türkçe Piyasa Narratifi
        
        Tüm verileri birleştirip insan gibi yorum yapar.
        """
        symbol = obs_data.get('symbol', 'BTC')
        direction = obs_data.get('direction', 'BELİRSİZ')
        confidence = obs_data.get('confidence', 0)
        reasons = obs_data.get('reasons', [])
        session = obs_data.get('session', {})
        stop_hunt = obs_data.get('stop_hunt', {})
        order_flow = obs_data.get('order_flow', {})
        manipulation = obs_data.get('manipulation', {})
        
        parts = []
        
        # Session context
        if session.get('name'):
            parts.append(f"{session['emoji']} {session['name']} session'ındayız. {session.get('character', '')}")
        
        # Ana yön
        if direction == 'YUKARI':
            parts.append(f"📈 {symbol} yukarı yönlü sinyal veriyor.")
        elif direction == 'AŞAĞI':
            parts.append(f"📉 {symbol} aşağı yönlü baskı altında.")
        else:
            parts.append(f"↔️ {symbol} için net bir yön yok, bekle.")
        
        # Order flow
        if order_flow.get('available'):
            parts.append(f"💹 Order flow: {order_flow['interpretation']}")
        
        # Stop hunt uyarısı
        if stop_hunt.get('detected'):
            parts.append(f"⚠️ {stop_hunt['description']}")
        
        # Manipülasyon uyarısı
        if manipulation.get('detected'):
            parts.append(f"🚨 {manipulation['description']}")
        
        # Güven açıklaması
        if confidence >= 70:
            parts.append(f"✅ Güven yüksek (%{confidence:.0f}) - Bu iyi bir fırsat olabilir.")
        elif confidence >= 50:
            parts.append(f"🟡 Güven orta (%{confidence:.0f}) - Dikkatli gir, stop koy.")
        else:
            parts.append(f"⚪ Güven düşük (%{confidence:.0f}) - Bekle, daha iyi fırsat gelecek.")
        
        # Final tavsiye
        if direction == 'YUKARI' and confidence >= 60:
            parts.append("🎯 Sonuç: LONG pozisyon düşünülebilir, ama stop loss mutlaka koy!")
        elif direction == 'AŞAĞI' and confidence >= 60:
            parts.append("🎯 Sonuç: SHORT pozisyon düşünülebilir, risk yönetimini unutma!")
        else:
            parts.append("🎯 Sonuç: Şimdilik işlem yapma, piyasayı izle.")
        
        return "\n".join(parts)
    
    def get_full_context(self, symbol: str, klines: list, volume_ratio: float) -> Dict:
        """
        Tam trader bağlamı al.
        """
        session = self.get_current_session()
        stop_hunt = self.detect_stop_hunt(klines)
        order_flow = self.analyze_order_flow(symbol)
        manipulation = self.detect_market_manipulation(klines, volume_ratio)
        
        return {
            'session': session,
            'stop_hunt': stop_hunt,
            'order_flow': order_flow,
            'manipulation': manipulation
        }


# Global instance
_mindset = None

def get_trader_mindset() -> TraderMindset:
    """Get or create trader mindset instance."""
    global _mindset
    if _mindset is None:
        _mindset = TraderMindset()
    return _mindset
