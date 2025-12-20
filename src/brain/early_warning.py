# -*- coding: utf-8 -*-
"""
DEMIR AI - Early Warning System
================================
Ani piyasa hareketlerinden ÖNCE uyarı veren sistem.

Monitör edilen sinyaller:
1. Volatilite sıkışması (Bollinger squeeze)
2. Hacim anomalisi (normal x2+)
3. Funding rate extreme (>0.05% veya <-0.05%)
4. Liquidation yoğunlaşması
5. Whale hareketleri
"""
import logging
import asyncio
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("EARLY_WARNING")


@dataclass
class Warning:
    """Erken uyarı."""
    symbol: str
    warning_type: str  # SQUEEZE, VOLUME_SPIKE, FUNDING_EXTREME, WHALE, LIQUIDATION
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    direction_hint: str  # LONG, SHORT, UNKNOWN
    message: str
    data: Dict
    timestamp: datetime = field(default_factory=datetime.now)


class EarlyWarningSystem:
    """
    7/24 çalışan erken uyarı sistemi.
    
    Bu sistem SİNYAL VERMEZ, sadece potansiyel hareketleri önceden tespit eder.
    """
    
    # Eşik değerleri
    BOLLINGER_SQUEEZE_THRESHOLD = 2.5  # < 2.5% bandwidth = squeeze
    VOLUME_SPIKE_THRESHOLD = 2.0  # 2x normal hacim
    FUNDING_EXTREME_THRESHOLD = 0.05  # %0.05
    LS_RATIO_EXTREME_HIGH = 1.8
    LS_RATIO_EXTREME_LOW = 0.55
    
    # Uyarı cooldown (spam önleme)
    WARNING_COOLDOWN = 1800  # 30 dakika
    
    def __init__(self):
        self.last_warnings: Dict[str, datetime] = {}  # warning_key -> last time
        self.active_warnings: List[Warning] = []
    
    async def scan_all(self, symbols: List[str] = None) -> List[Warning]:
        """Tüm coinleri tara ve uyarıları topla."""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT']
        
        all_warnings = []
        
        for symbol in symbols:
            warnings = await self.scan_symbol(symbol)
            all_warnings.extend(warnings)
        
        # Severity'ye göre sırala
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        all_warnings.sort(key=lambda w: severity_order.get(w.severity, 4))
        
        self.active_warnings = all_warnings
        return all_warnings
    
    async def scan_symbol(self, symbol: str) -> List[Warning]:
        """Tek bir coini tara."""
        warnings = []
        
        try:
            # Veri topla
            price_data = await self._get_price_data(symbol)
            derivatives = await self._get_derivatives_data(symbol)
            
            if not price_data:
                return []
            
            # 1. Bollinger Squeeze kontrolü
            squeeze_warning = self._check_bollinger_squeeze(symbol, price_data)
            if squeeze_warning and self._can_warn(f"{symbol}_squeeze"):
                warnings.append(squeeze_warning)
                self.last_warnings[f"{symbol}_squeeze"] = datetime.now()
            
            # 2. Hacim anomalisi
            volume_warning = self._check_volume_spike(symbol, price_data)
            if volume_warning and self._can_warn(f"{symbol}_volume"):
                warnings.append(volume_warning)
                self.last_warnings[f"{symbol}_volume"] = datetime.now()
            
            # 3. Funding rate extreme
            if derivatives:
                funding_warning = self._check_funding_extreme(symbol, derivatives)
                if funding_warning and self._can_warn(f"{symbol}_funding"):
                    warnings.append(funding_warning)
                    self.last_warnings[f"{symbol}_funding"] = datetime.now()
                
                # 4. L/S ratio extreme
                ls_warning = self._check_ls_ratio_extreme(symbol, derivatives)
                if ls_warning and self._can_warn(f"{symbol}_ls"):
                    warnings.append(ls_warning)
                    self.last_warnings[f"{symbol}_ls"] = datetime.now()
            
            # 5. Whale hareketleri (orderbook imbalance)
            whale_warning = await self._check_orderbook_imbalance(symbol)
            if whale_warning and self._can_warn(f"{symbol}_whale"):
                warnings.append(whale_warning)
                self.last_warnings[f"{symbol}_whale"] = datetime.now()
                
        except Exception as e:
            logger.warning(f"Scan failed for {symbol}: {e}")
        
        return warnings
    
    # =========================================================================
    # CHECK FUNCTIONS
    # =========================================================================
    
    def _check_bollinger_squeeze(self, symbol: str, price_data: Dict) -> Optional[Warning]:
        """Volatilite sıkışması kontrolü."""
        closes = price_data.get('closes', [])
        if len(closes) < 20:
            return None
        
        # Bollinger Bands hesapla
        sma = np.mean(closes[-20:])
        std = np.std(closes[-20:])
        
        if sma == 0:
            return None
        
        bandwidth = (2 * std / sma) * 100  # % olarak
        
        if bandwidth < self.BOLLINGER_SQUEEZE_THRESHOLD:
            # Hangi yöne kırılacak?
            current = closes[-1]
            direction = 'LONG' if current > sma else 'SHORT'
            
            severity = 'HIGH' if bandwidth < 1.5 else 'MEDIUM'
            
            return Warning(
                symbol=symbol,
                warning_type='SQUEEZE',
                severity=severity,
                direction_hint=direction,
                message=f"⚡ {symbol}: Volatilite sıkışması! Bandwidth: {bandwidth:.1f}%. Yakında patlamaya hazır.",
                data={'bandwidth': bandwidth, 'sma': sma, 'current': current}
            )
        
        return None
    
    def _check_volume_spike(self, symbol: str, price_data: Dict) -> Optional[Warning]:
        """Hacim anomalisi kontrolü."""
        volumes = price_data.get('volumes', [])
        if len(volumes) < 10:
            return None
        
        avg_volume = np.mean(volumes[:-1])
        current_volume = volumes[-1]
        
        if avg_volume == 0:
            return None
        
        ratio = current_volume / avg_volume
        
        if ratio >= self.VOLUME_SPIKE_THRESHOLD:
            # Yön tahmini: son mum yönüne bak
            closes = price_data.get('closes', [])
            if len(closes) >= 2:
                direction = 'LONG' if closes[-1] > closes[-2] else 'SHORT'
            else:
                direction = 'UNKNOWN'
            
            severity = 'CRITICAL' if ratio > 3.0 else 'HIGH' if ratio > 2.5 else 'MEDIUM'
            
            return Warning(
                symbol=symbol,
                warning_type='VOLUME_SPIKE',
                severity=severity,
                direction_hint=direction,
                message=f"🔥 {symbol}: Hacim patlaması! {ratio:.1f}x normal. Büyük oyuncular aktif.",
                data={'ratio': ratio, 'current_volume': current_volume, 'avg_volume': avg_volume}
            )
        
        return None
    
    def _check_funding_extreme(self, symbol: str, derivatives: Dict) -> Optional[Warning]:
        """Funding rate extreme kontrolü."""
        funding = derivatives.get('funding_rate', 0)
        
        if abs(funding) >= self.FUNDING_EXTREME_THRESHOLD:
            # Yüksek funding = long'lar ödüyor = short bias
            # Düşük funding = short'lar ödüyor = long bias
            direction = 'SHORT' if funding > 0 else 'LONG'
            
            severity = 'HIGH' if abs(funding) > 0.1 else 'MEDIUM'
            
            side = 'Longlar' if funding > 0 else 'Shortlar'
            action = 'ödüyor' if funding > 0 else 'kazanıyor'
            
            return Warning(
                symbol=symbol,
                warning_type='FUNDING_EXTREME',
                severity=severity,
                direction_hint=direction,
                message=f"💰 {symbol}: Funding extreme! {funding:.3f}%. {side} {action}. Squeeze riski.",
                data={'funding_rate': funding}
            )
        
        return None
    
    def _check_ls_ratio_extreme(self, symbol: str, derivatives: Dict) -> Optional[Warning]:
        """Long/Short ratio extreme kontrolü."""
        ratio = derivatives.get('long_short_ratio', 1.0)
        
        if ratio >= self.LS_RATIO_EXTREME_HIGH:
            # Çok fazla long = short squeeze potansiyeli değil, LONG squeeze!
            return Warning(
                symbol=symbol,
                warning_type='LS_EXTREME',
                severity='MEDIUM',
                direction_hint='SHORT',  # Contrarian
                message=f"📊 {symbol}: L/S Ratio yüksek ({ratio:.2f}). Çok fazla long. Düşüş riski.",
                data={'long_short_ratio': ratio}
            )
        
        elif ratio <= self.LS_RATIO_EXTREME_LOW:
            # Çok fazla short = short squeeze potansiyeli
            return Warning(
                symbol=symbol,
                warning_type='LS_EXTREME',
                severity='HIGH',  # Short squeeze daha tehlikeli
                direction_hint='LONG',
                message=f"📊 {symbol}: L/S Ratio düşük ({ratio:.2f}). Çok fazla short. SHORT SQUEEZE riski!",
                data={'long_short_ratio': ratio}
            )
        
        return None
    
    async def _check_orderbook_imbalance(self, symbol: str) -> Optional[Warning]:
        """Order book dengesizliği (whale walls)."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/depth",
                params={'symbol': symbol, 'limit': 50},
                timeout=5
            )
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            # Toplam bid ve ask likiditesi
            bid_liquidity = sum(float(b[1]) for b in data.get('bids', []))
            ask_liquidity = sum(float(a[1]) for a in data.get('asks', []))
            
            if bid_liquidity == 0 or ask_liquidity == 0:
                return None
            
            imbalance = bid_liquidity / ask_liquidity
            
            if imbalance > 2.0:
                # Çok fazla alıcı = yukarı potansiyel
                return Warning(
                    symbol=symbol,
                    warning_type='WHALE',
                    severity='MEDIUM',
                    direction_hint='LONG',
                    message=f"🐋 {symbol}: Order book dengesiz. Alıcı duvarı {imbalance:.1f}x. Yukarı baskı.",
                    data={'imbalance': imbalance, 'bid_liq': bid_liquidity, 'ask_liq': ask_liquidity}
                )
            
            elif imbalance < 0.5:
                # Çok fazla satıcı = aşağı potansiyel
                return Warning(
                    symbol=symbol,
                    warning_type='WHALE',
                    severity='MEDIUM',
                    direction_hint='SHORT',
                    message=f"🐋 {symbol}: Order book dengesiz. Satıcı duvarı {1/imbalance:.1f}x. Aşağı baskı.",
                    data={'imbalance': imbalance, 'bid_liq': bid_liquidity, 'ask_liq': ask_liquidity}
                )
                
        except Exception as e:
            logger.debug(f"Orderbook check failed: {e}")
        
        return None
    
    # =========================================================================
    # DATA HELPERS
    # =========================================================================
    
    async def _get_price_data(self, symbol: str) -> Optional[Dict]:
        """Fiyat ve hacim verileri al."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '15m', 'limit': 30},
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            klines = resp.json()
            
            return {
                'closes': [float(k[4]) for k in klines],
                'volumes': [float(k[5]) for k in klines],
                'highs': [float(k[2]) for k in klines],
                'lows': [float(k[3]) for k in klines]
            }
            
        except Exception as e:
            logger.debug(f"Price data fetch failed: {e}")
            return None
    
    async def _get_derivatives_data(self, symbol: str) -> Optional[Dict]:
        """Derivatives verileri al."""
        result = {}
        
        try:
            # Funding rate
            fr_resp = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            if fr_resp.status_code == 200:
                data = fr_resp.json()
                if data:
                    result['funding_rate'] = float(data[0].get('fundingRate', 0)) * 100
            
            # L/S ratio
            ls_resp = requests.get(
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                params={'symbol': symbol, 'period': '1h', 'limit': 1},
                timeout=5
            )
            if ls_resp.status_code == 200:
                data = ls_resp.json()
                if data:
                    result['long_short_ratio'] = float(data[0].get('longShortRatio', 1.0))
                    
        except Exception as e:
            logger.debug(f"Derivatives fetch failed: {e}")
        
        return result if result else None
    
    def _can_warn(self, warning_key: str) -> bool:
        """Cooldown kontrolü."""
        if warning_key not in self.last_warnings:
            return True
        
        elapsed = (datetime.now() - self.last_warnings[warning_key]).total_seconds()
        return elapsed >= self.WARNING_COOLDOWN
    
    # =========================================================================
    # FORMATTING
    # =========================================================================
    
    def format_warnings(self, warnings: List[Warning]) -> str:
        """Telegram formatında uyarılar."""
        if not warnings:
            return ""
        
        severity_emoji = {
            'CRITICAL': '🚨',
            'HIGH': '⚠️',
            'MEDIUM': '📢',
            'LOW': 'ℹ️'
        }
        
        lines = ["🔔 ERKEN UYARI SİSTEMİ", "━━━━━━━━━━━━━━━━━━"]
        
        for w in warnings[:5]:  # Max 5 uyarı
            emoji = severity_emoji.get(w.severity, '📢')
            lines.append(f"{emoji} {w.message}")
        
        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append(f"⏰ {datetime.now().strftime('%H:%M')}")
        
        return "\n".join(lines)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_warning_system: Optional[EarlyWarningSystem] = None

def get_warning_system() -> EarlyWarningSystem:
    """Get or create warning system instance."""
    global _warning_system
    if _warning_system is None:
        _warning_system = EarlyWarningSystem()
    return _warning_system


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        system = get_warning_system()
        warnings = await system.scan_all()
        
        if warnings:
            print(system.format_warnings(warnings))
        else:
            print("No warnings detected")
        
        print(f"\nTotal warnings: {len(warnings)}")
    
    asyncio.run(test())
