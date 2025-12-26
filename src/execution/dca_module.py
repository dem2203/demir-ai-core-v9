# -*- coding: utf-8 -*-
"""
DEMIR AI - DCA (Dollar Cost Averaging) Module
==============================================
Terste kalınan pozisyonlarda ortalama düşürme stratejisi.

KULLANIM:
- Zarar eden pozisyonlarda ek alım yaparak ortalama maliyeti düşürür
- Risk limitleri ile kontrollü DCA
- Piyasa koşullarına göre akıllı DCA kararları

⚠️ UYARI: DCA yanlış kullanılırsa kayıpları artırabilir!
         Sadece güçlü temel analiz ile birlikte kullanılmalı.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("DCA_MODULE")


class DCAStrategy(Enum):
    """DCA Stratejileri"""
    FIXED = "fixed"           # Sabit miktarda alım
    MARTINGALE = "martingale" # Her seferinde 2x (RİSKLİ!)
    SCALED = "scaled"         # Kayıp oranına göre ölçekli
    SMART = "smart"           # Piyasa koşullarına göre dinamik


@dataclass
class DCALevel:
    """Tek bir DCA seviyesi"""
    level_number: int          # DCA seviyesi (1, 2, 3...)
    trigger_drop_pct: float   # Tetikleme için gereken düşüş %
    buy_amount_pct: float     # Pozisyon büyüklüğünün % kaçı alınacak
    executed: bool = False
    executed_at: Optional[datetime] = None
    executed_price: Optional[float] = None


@dataclass 
class DCAPosition:
    """DCA takibi yapılan pozisyon"""
    symbol: str
    entry_price: float
    current_avg_price: float
    total_quantity: float
    total_cost: float
    dca_levels: List[DCALevel] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_dca_at: Optional[datetime] = None
    dca_count: int = 0
    max_dca_reached: bool = False


class DCAModule:
    """
    Dollar Cost Averaging Module
    
    Terste kalınan pozisyonlarda kontrollü ortalama düşürme.
    """
    
    # Varsayılan DCA seviyeleri
    DEFAULT_LEVELS = [
        DCALevel(1, 5.0, 25.0),   # %5 düşüşte, pozisyonun %25'i kadar al
        DCALevel(2, 10.0, 25.0),  # %10 düşüşte, pozisyonun %25'i kadar al
        DCALevel(3, 15.0, 25.0),  # %15 düşüşte, pozisyonun %25'i kadar al
        DCALevel(4, 20.0, 25.0),  # %20 düşüşte, pozisyonun %25'i kadar al (MAX)
    ]
    
    def __init__(
        self,
        strategy: DCAStrategy = DCAStrategy.SCALED,
        max_dca_count: int = 4,
        min_dca_interval_hours: float = 4.0,
        max_position_multiplier: float = 3.0,
        enable_smart_filter: bool = True
    ):
        """
        Args:
            strategy: DCA stratejisi
            max_dca_count: Maksimum DCA sayısı
            min_dca_interval_hours: DCA'lar arası minimum süre
            max_position_multiplier: Başlangıç pozisyonunun max kaç katına çıkabilir
            enable_smart_filter: Akıllı filtre (VIX, trend kontrolü)
        """
        self.strategy = strategy
        self.max_dca_count = max_dca_count
        self.min_dca_interval = timedelta(hours=min_dca_interval_hours)
        self.max_position_multiplier = max_position_multiplier
        self.enable_smart_filter = enable_smart_filter
        
        # Aktif DCA pozisyonları
        self.positions: Dict[str, DCAPosition] = {}
        
        logger.info(f"💰 DCA Module initialized: Strategy={strategy.value}, MaxDCA={max_dca_count}")
    
    def create_position(
        self, 
        symbol: str, 
        entry_price: float, 
        quantity: float,
        custom_levels: Optional[List[DCALevel]] = None
    ) -> DCAPosition:
        """
        Yeni DCA takibi yapılacak pozisyon oluştur.
        """
        levels = custom_levels or [
            DCALevel(l.level_number, l.trigger_drop_pct, l.buy_amount_pct)
            for l in self.DEFAULT_LEVELS
        ]
        
        position = DCAPosition(
            symbol=symbol,
            entry_price=entry_price,
            current_avg_price=entry_price,
            total_quantity=quantity,
            total_cost=entry_price * quantity,
            dca_levels=levels[:self.max_dca_count]
        )
        
        self.positions[symbol] = position
        logger.info(f"📊 DCA Position created: {symbol} @ ${entry_price:.2f} x {quantity}")
        
        return position
    
    def check_dca_opportunity(
        self, 
        symbol: str, 
        current_price: float,
        market_data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        DCA fırsatı kontrolü yap.
        
        Args:
            symbol: Trading pair
            current_price: Güncel fiyat
            market_data: Opsiyonel piyasa verileri (VIX, trend, etc.)
            
        Returns:
            DCA önerisi dict veya None
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        # Max DCA'ya ulaşıldı mı?
        if position.dca_count >= self.max_dca_count:
            if not position.max_dca_reached:
                position.max_dca_reached = True
                logger.warning(f"⚠️ {symbol}: Max DCA limitine ulaşıldı ({self.max_dca_count})")
            return None
        
        # Minimum interval kontrolü
        if position.last_dca_at:
            time_since_last = datetime.now() - position.last_dca_at
            if time_since_last < self.min_dca_interval:
                remaining = self.min_dca_interval - time_since_last
                logger.debug(f"⏳ {symbol}: DCA interval beklemede ({remaining})")
                return None
        
        # Mevcut düşüş oranı
        drop_pct = ((position.entry_price - current_price) / position.entry_price) * 100
        
        if drop_pct <= 0:
            # Fiyat entry'nin üstünde, DCA gerek yok
            return None
        
        # Aktif edilmemiş en uygun DCA seviyesini bul
        triggered_level = None
        for level in position.dca_levels:
            if not level.executed and drop_pct >= level.trigger_drop_pct:
                triggered_level = level
        
        if not triggered_level:
            return None
        
        # Smart filter kontrolü
        if self.enable_smart_filter and market_data:
            if not self._smart_filter_check(market_data):
                logger.info(f"🛡️ {symbol}: Smart filter DCA'yı engelledi")
                return None
        
        # DCA miktarı hesapla
        dca_amount = self._calculate_dca_amount(position, triggered_level, current_price)
        
        return {
            'symbol': symbol,
            'action': 'DCA_BUY',
            'current_price': current_price,
            'entry_price': position.entry_price,
            'avg_price': position.current_avg_price,
            'drop_pct': drop_pct,
            'dca_level': triggered_level.level_number,
            'buy_quantity': dca_amount['quantity'],
            'buy_value_usd': dca_amount['value_usd'],
            'new_avg_price_estimate': dca_amount['new_avg_price'],
            'total_dca_count': position.dca_count + 1,
            'remaining_dca': self.max_dca_count - position.dca_count - 1
        }
    
    def execute_dca(
        self, 
        symbol: str, 
        executed_price: float, 
        executed_quantity: float
    ) -> bool:
        """
        DCA işlemini kaydet.
        """
        if symbol not in self.positions:
            logger.error(f"❌ {symbol}: DCA pozisyonu bulunamadı")
            return False
            
        position = self.positions[symbol]
        
        # Yeni ortalama hesapla
        new_total_cost = position.total_cost + (executed_price * executed_quantity)
        new_total_quantity = position.total_quantity + executed_quantity
        new_avg_price = new_total_cost / new_total_quantity
        
        # Pozisyonu güncelle
        position.total_cost = new_total_cost
        position.total_quantity = new_total_quantity
        position.current_avg_price = new_avg_price
        position.dca_count += 1
        position.last_dca_at = datetime.now()
        
        # Seviyeyi işaretle
        for level in position.dca_levels:
            if not level.executed:
                level.executed = True
                level.executed_at = datetime.now()
                level.executed_price = executed_price
                break
        
        logger.info(
            f"✅ DCA #{position.dca_count} EXECUTED: {symbol} "
            f"@ ${executed_price:.2f} x {executed_quantity} | "
            f"New Avg: ${new_avg_price:.2f}"
        )
        
        return True
    
    def close_position(self, symbol: str) -> Optional[Dict]:
        """
        DCA pozisyonunu kapat ve sonuçları döndür.
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions.pop(symbol)
        
        return {
            'symbol': symbol,
            'entry_price': position.entry_price,
            'final_avg_price': position.current_avg_price,
            'total_quantity': position.total_quantity,
            'total_cost': position.total_cost,
            'dca_count': position.dca_count,
            'duration': datetime.now() - position.created_at
        }
    
    def get_position_status(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Pozisyon durumu özeti.
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        unrealized_pnl = (current_price - position.current_avg_price) * position.total_quantity
        unrealized_pnl_pct = ((current_price / position.current_avg_price) - 1) * 100
        
        return {
            'symbol': symbol,
            'entry_price': position.entry_price,
            'avg_price': position.current_avg_price,
            'current_price': current_price,
            'quantity': position.total_quantity,
            'cost_basis': position.total_cost,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'dca_count': position.dca_count,
            'max_dca': self.max_dca_count,
            'remaining_dca': self.max_dca_count - position.dca_count
        }
    
    def _calculate_dca_amount(
        self, 
        position: DCAPosition, 
        level: DCALevel, 
        current_price: float
    ) -> Dict:
        """
        DCA alım miktarını hesapla.
        """
        if self.strategy == DCAStrategy.FIXED:
            # Sabit: İlk pozisyonun %'si kadar
            value_usd = position.entry_price * position.total_quantity * (level.buy_amount_pct / 100)
            
        elif self.strategy == DCAStrategy.MARTINGALE:
            # Martingale: Her seferinde 2x (RİSKLİ!)
            base_value = position.entry_price * position.total_quantity * 0.25
            value_usd = base_value * (2 ** position.dca_count)
            
        elif self.strategy == DCAStrategy.SCALED:
            # Ölçekli: Düşüş oranına göre
            drop_pct = ((position.entry_price - current_price) / position.entry_price) * 100
            scale_factor = min(2.0, 1 + (drop_pct / 20))  # Max 2x
            base_value = position.entry_price * position.total_quantity * (level.buy_amount_pct / 100)
            value_usd = base_value * scale_factor
            
        else:  # SMART
            # Akıllı: Volatilite ve trende göre
            base_value = position.entry_price * position.total_quantity * (level.buy_amount_pct / 100)
            value_usd = base_value  # Temel implementasyon
        
        # Max position limiti kontrolü
        max_allowed = position.entry_price * position.total_quantity * self.max_position_multiplier
        current_total = position.total_cost
        
        if current_total + value_usd > max_allowed:
            value_usd = max_allowed - current_total
            if value_usd <= 0:
                logger.warning(f"⚠️ Max position limit ({self.max_position_multiplier}x) reached")
                return {'quantity': 0, 'value_usd': 0, 'new_avg_price': position.current_avg_price}
        
        quantity = value_usd / current_price
        new_total_cost = position.total_cost + value_usd
        new_total_qty = position.total_quantity + quantity
        new_avg_price = new_total_cost / new_total_qty
        
        return {
            'quantity': quantity,
            'value_usd': value_usd,
            'new_avg_price': new_avg_price
        }
    
    def _smart_filter_check(self, market_data: Dict) -> bool:
        """
        Akıllı filtre: DCA yapmak mantıklı mı?
        """
        # VIX çok yüksekse DCA yapma
        vix = market_data.get('vix', 0)
        if vix > 35:
            logger.debug(f"Smart Filter: VIX too high ({vix})")
            return False
        
        # Trend aşağı ise dikkatli ol
        trend = market_data.get('trend', 'NEUTRAL')
        rsi = market_data.get('rsi', 50)
        
        # RSI aşırı düşükse (oversold) DCA mantıklı olabilir
        if rsi < 25:
            return True
        
        # Güçlü downtrend'de DCA yapma
        if trend == 'STRONG_DOWN' and rsi > 40:
            logger.debug(f"Smart Filter: Strong downtrend, RSI not oversold")
            return False
        
        return True
    
    def get_all_positions_summary(self) -> List[Dict]:
        """
        Tüm DCA pozisyonlarının özeti.
        """
        summaries = []
        for symbol, position in self.positions.items():
            summaries.append({
                'symbol': symbol,
                'entry': position.entry_price,
                'avg': position.current_avg_price,
                'quantity': position.total_quantity,
                'dca_count': f"{position.dca_count}/{self.max_dca_count}",
                'created': position.created_at.isoformat()
            })
        return summaries


# Global instance
_dca_module: Optional[DCAModule] = None


def get_dca_module() -> DCAModule:
    """Get or create DCA module instance."""
    global _dca_module
    if _dca_module is None:
        _dca_module = DCAModule()
    return _dca_module
