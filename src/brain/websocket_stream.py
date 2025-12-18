# -*- coding: utf-8 -*-
"""
DEMIR AI - WebSocket Real-Time Stream
Anlık trade tespiti - Milisaniye içinde tepki.

PHASE 85: WebSocket Real-Time Stream
- Binance WebSocket'e bağlan
- Büyük trade'leri anında tespit et
- Ani hareket başladığında HEMEN bildir
"""
import logging
import asyncio
import json
import websockets
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, List
from collections import deque
import threading

logger = logging.getLogger("WEBSOCKET_STREAM")


class WebSocketStream:
    """
    Binance WebSocket Real-Time Stream
    
    Anlık trade akışını izler ve büyük hareketleri tespit eder.
    """
    
    BINANCE_WS = "wss://stream.binance.com:9443/ws"
    BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"
    
    def __init__(self, on_alert_callback: Optional[Callable] = None):
        self.on_alert = on_alert_callback
        self.running = False
        self.ws = None
        
        # Trade buffer (son 60 saniye)
        self.trade_buffer = deque(maxlen=1000)
        self.large_trades = deque(maxlen=100)
        
        # Thresholds
        self.large_trade_btc = 5  # 5+ BTC = Large trade
        self.volume_spike_multiplier = 3.0  # 3x normal = spike
        self.price_move_threshold = 0.3  # %0.3 = ani hareket
        
        # State
        self.last_price = 0
        self.avg_volume_per_min = 0
        self.last_alert_time = None
        self.alert_cooldown = timedelta(seconds=30)  # 30 saniye cooldown
        
        logger.info("✅ WebSocket Stream initialized")
    
    async def start(self, symbol: str = "btcusdt"):
        """WebSocket stream başlat."""
        self.running = True
        stream_name = f"{symbol.lower()}@aggTrade"
        url = f"{self.BINANCE_WS}/{stream_name}"
        
        logger.info(f"🔌 Connecting to WebSocket: {stream_name}")
        
        try:
            async with websockets.connect(url) as ws:
                self.ws = ws
                logger.info("✅ WebSocket connected!")
                
                while self.running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        await self._process_trade(json.loads(message))
                    except asyncio.TimeoutError:
                        # Heartbeat
                        await ws.ping()
                    except Exception as e:
                        logger.warning(f"WebSocket error: {e}")
                        await asyncio.sleep(1)
                        
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.running = False
    
    async def _process_trade(self, data: dict):
        """Her trade'i işle ve büyük hareketleri tespit et."""
        try:
            price = float(data.get('p', 0))
            qty = float(data.get('q', 0))
            is_buyer_maker = data.get('m', False)  # True = seller, False = buyer
            trade_time = data.get('T', 0)
            
            trade_value_btc = qty
            trade_value_usd = price * qty
            
            # Buffer'a ekle
            self.trade_buffer.append({
                'price': price,
                'qty': qty,
                'value_usd': trade_value_usd,
                'is_sell': is_buyer_maker,
                'time': trade_time
            })
            
            # 1. BÜYÜK TRADE TESPİTİ
            if trade_value_btc >= self.large_trade_btc:
                side = "SELL" if is_buyer_maker else "BUY"
                self.large_trades.append({
                    'price': price,
                    'qty': qty,
                    'side': side,
                    'time': datetime.now()
                })
                
                logger.info(f"🐋 LARGE TRADE: {side} {qty:.2f} BTC @ ${price:,.0f}")
                
                # Alert gönder
                await self._trigger_alert({
                    'type': 'LARGE_TRADE',
                    'side': side,
                    'qty': qty,
                    'price': price,
                    'value_usd': trade_value_usd
                })
            
            # 2. ANİ FİYAT HAREKETİ TESPİTİ
            if self.last_price > 0:
                price_change_pct = abs(price - self.last_price) / self.last_price * 100
                
                if price_change_pct >= self.price_move_threshold:
                    direction = "UP" if price > self.last_price else "DOWN"
                    logger.warning(f"⚡ SUDDEN MOVE: {direction} {price_change_pct:.2f}%")
                    
                    await self._trigger_alert({
                        'type': 'SUDDEN_MOVE',
                        'direction': direction,
                        'change_pct': price_change_pct,
                        'price': price,
                        'prev_price': self.last_price
                    })
            
            self.last_price = price
            
            # 3. HACİM SPIKE TESPİTİ (son 1 dakika)
            await self._check_volume_spike()
            
        except Exception as e:
            logger.debug(f"Trade processing error: {e}")
    
    async def _check_volume_spike(self):
        """Son 1 dakikadaki hacmi kontrol et."""
        now = datetime.now()
        one_min_ago = now - timedelta(minutes=1)
        
        # Son 1 dakikadaki trade'leri topla
        recent_volume = sum(
            t['qty'] for t in self.trade_buffer 
            if datetime.fromtimestamp(t['time']/1000) > one_min_ago
        )
        
        # Ortalama güncelle (basit moving average)
        if self.avg_volume_per_min == 0:
            self.avg_volume_per_min = recent_volume
        else:
            self.avg_volume_per_min = 0.9 * self.avg_volume_per_min + 0.1 * recent_volume
        
        # Spike tespit
        if self.avg_volume_per_min > 0 and recent_volume > self.avg_volume_per_min * self.volume_spike_multiplier:
            logger.warning(f"📊 VOLUME SPIKE: {recent_volume:.1f} BTC/min (Avg: {self.avg_volume_per_min:.1f})")
            
            await self._trigger_alert({
                'type': 'VOLUME_SPIKE',
                'current_volume': recent_volume,
                'avg_volume': self.avg_volume_per_min,
                'spike_ratio': recent_volume / self.avg_volume_per_min
            })
    
    async def _trigger_alert(self, alert_data: dict):
        """Alert callback'i çağır (cooldown ile)."""
        now = datetime.now()
        
        if self.last_alert_time and now - self.last_alert_time < self.alert_cooldown:
            return  # Cooldown içinde
        
        self.last_alert_time = now
        
        if self.on_alert:
            try:
                await self.on_alert(alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def stop(self):
        """WebSocket'i durdur."""
        self.running = False
        logger.info("🔌 WebSocket stopped")
    
    def get_recent_large_trades(self, minutes: int = 5) -> List[Dict]:
        """Son X dakikadaki büyük trade'leri döndür."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [t for t in self.large_trades if t['time'] > cutoff]
    
    def get_market_pressure(self) -> Dict:
        """
        Piyasa baskısını hesapla.
        
        Returns:
            {
                'buy_volume': 100,
                'sell_volume': 80,
                'net_pressure': 20,  # Pozitif = alış baskısı
                'pressure_ratio': 1.25,
                'direction': 'BUY_PRESSURE'
            }
        """
        now = datetime.now()
        five_min_ago = now - timedelta(minutes=5)
        
        recent_large = [t for t in self.large_trades if t['time'] > five_min_ago]
        
        buy_vol = sum(t['qty'] for t in recent_large if t['side'] == 'BUY')
        sell_vol = sum(t['qty'] for t in recent_large if t['side'] == 'SELL')
        
        net = buy_vol - sell_vol
        ratio = buy_vol / sell_vol if sell_vol > 0 else (2.0 if buy_vol > 0 else 1.0)
        
        if ratio > 1.5:
            direction = 'STRONG_BUY_PRESSURE'
        elif ratio > 1.1:
            direction = 'BUY_PRESSURE'
        elif ratio < 0.67:
            direction = 'STRONG_SELL_PRESSURE'
        elif ratio < 0.9:
            direction = 'SELL_PRESSURE'
        else:
            direction = 'NEUTRAL'
        
        return {
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'net_pressure': net,
            'pressure_ratio': ratio,
            'direction': direction
        }


# Singleton instance
_ws_stream = None

def get_websocket_stream(callback=None) -> WebSocketStream:
    """Get or create WebSocket stream instance."""
    global _ws_stream
    if _ws_stream is None:
        _ws_stream = WebSocketStream(on_alert_callback=callback)
    return _ws_stream


# Test
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def on_alert(data):
        print(f"🚨 ALERT: {data}")
    
    async def main():
        ws = WebSocketStream(on_alert_callback=on_alert)
        await ws.start("btcusdt")
    
    asyncio.run(main())
