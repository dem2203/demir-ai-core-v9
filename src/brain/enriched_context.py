# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - ENRICHED MARKET CONTEXT
======================================
AI'lara gönderilecek zenginleştirilmiş piyasa bağlamı.

Sadece sayı değil, ANLAM da gönderir:
- Historical percentile (son 30 gün içindeki yeri)
- Trend direction (yükseliyor/düşüyor)
- Anomaly flags (normalden çok farklıysa)
- News summaries (son önemli haberler)
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("ENRICHED_CONTEXT")


@dataclass
class IndicatorContext:
    """Tek bir gösterge için zenginleştirilmiş bağlam"""
    name: str
    current_value: float
    percentile_30d: float  # Son 30 gün içinde yüzdelik dilim (0-100)
    trend: str  # RISING, FALLING, STABLE
    trend_strength: float  # -100 to +100
    is_extreme: bool  # %10 altı veya %90 üstü mü?
    is_anomaly: bool  # Normal dağılımdan 2 std sapma dışında mı?
    context_text: str  # İnsan okunabilir açıklama
    
    def to_prompt(self) -> str:
        """AI promptu için format"""
        emoji = "🔴" if self.is_extreme else "🟢"
        trend_emoji = "📈" if self.trend == "RISING" else "📉" if self.trend == "FALLING" else "➡️"
        
        return f"""
{self.name}: {self.current_value:.2f}
  {trend_emoji} Trend: {self.trend} ({self.trend_strength:+.0f}%)
  📊 30-Day Percentile: {self.percentile_30d:.0f}% {emoji}
  💡 Context: {self.context_text}
  {'⚠️ EXTREME VALUE!' if self.is_extreme else ''}
  {'🚨 ANOMALY DETECTED!' if self.is_anomaly else ''}
"""


@dataclass
class EnrichedMarketContext:
    """AI'lar için zenginleştirilmiş piyasa bağlamı"""
    symbol: str
    current_price: float
    timestamp: datetime
    
    # Enriched indicators
    rsi: Optional[IndicatorContext] = None
    fear_greed: Optional[IndicatorContext] = None
    funding_rate: Optional[IndicatorContext] = None
    ls_ratio: Optional[IndicatorContext] = None
    btc_dominance: Optional[IndicatorContext] = None
    orderbook_imbalance: Optional[IndicatorContext] = None
    whale_activity: Optional[IndicatorContext] = None
    oi_change: Optional[IndicatorContext] = None
    
    # Market structure
    market_regime: str = "UNKNOWN"
    regime_duration_hours: int = 0
    last_regime_change: Optional[datetime] = None
    
    # News & Events
    recent_news: List[Dict] = field(default_factory=list)
    upcoming_events: List[Dict] = field(default_factory=list)
    
    # Whale activity
    whale_movements: List[Dict] = field(default_factory=list)
    net_whale_flow_24h: float = 0.0
    
    # Technical levels
    key_resistance_levels: List[float] = field(default_factory=list)
    key_support_levels: List[float] = field(default_factory=list)
    distance_to_resistance_pct: float = 0.0
    distance_to_support_pct: float = 0.0
    
    # Validation
    data_quality_score: float = 100.0
    data_warnings: List[str] = field(default_factory=list)
    stale_data_flags: List[str] = field(default_factory=list)
    
    def to_ai_prompt(self) -> str:
        """AI için kapsamlı prompt oluştur"""
        prompt = f"""
## 📊 ENRICHED MARKET ANALYSIS - {self.symbol}
Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
Data Quality Score: {self.data_quality_score:.0f}/100
{' | '.join(self.data_warnings) if self.data_warnings else 'No warnings'}

### 💰 Price
Current: ${self.current_price:,.2f}
Distance to Resistance: {self.distance_to_resistance_pct:+.1f}%
Distance to Support: {self.distance_to_support_pct:+.1f}%
Key Resistance: {', '.join([f'${r:,.0f}' for r in self.key_resistance_levels[:3]])}
Key Support: {', '.join([f'${s:,.0f}' for s in self.key_support_levels[:3]])}

### 🎯 Market Regime
Current: {self.market_regime}
Duration: {self.regime_duration_hours} hours
{'⚠️ Regime may be changing soon!' if self.regime_duration_hours > 72 else ''}

### 📈 Technical Indicators (Enriched)
"""
        if self.rsi:
            prompt += self.rsi.to_prompt()
        if self.fear_greed:
            prompt += self.fear_greed.to_prompt()
        if self.funding_rate:
            prompt += self.funding_rate.to_prompt()
        if self.ls_ratio:
            prompt += self.ls_ratio.to_prompt()
        if self.orderbook_imbalance:
            prompt += self.orderbook_imbalance.to_prompt()
        if self.whale_activity:
            prompt += self.whale_activity.to_prompt()
        
        prompt += f"""
### 🐋 Whale Activity (Last 24h)
Net Flow: ${self.net_whale_flow_24h:,.0f}
"""
        for movement in self.whale_movements[:5]:
            prompt += f"  • {movement.get('description', 'Unknown movement')}\n"
        
        prompt += "\n### 📰 Recent News\n"
        for news in self.recent_news[:5]:
            sentiment_emoji = "🟢" if news.get('sentiment') == 'BULLISH' else "🔴" if news.get('sentiment') == 'BEARISH' else "⚪"
            prompt += f"  {sentiment_emoji} {news.get('title', 'Unknown')[:80]}\n"
        
        if self.upcoming_events:
            prompt += "\n### 📅 Upcoming Events\n"
            for event in self.upcoming_events[:3]:
                prompt += f"  ⏰ {event.get('name')} - {event.get('time')}\n"
        
        if self.stale_data_flags:
            prompt += f"\n### ⚠️ DATA WARNINGS\n"
            for flag in self.stale_data_flags:
                prompt += f"  🚨 {flag}\n"
        
        return prompt


class EnrichedContextBuilder:
    """Zenginleştirilmiş bağlam oluşturucu"""
    
    def __init__(self):
        self._historical_data: Dict[str, List[float]] = {}
        self._cache_duration = timedelta(hours=1)
        logger.info("📊 Enriched Context Builder initialized")
    
    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Değerin historical percentile'ını hesapla"""
        if not history:
            return 50.0
        
        below = sum(1 for h in history if h < value)
        return (below / len(history)) * 100
    
    def _detect_trend(self, history: List[float], window: int = 5) -> Tuple[str, float]:
        """Trend yönü ve gücü tespit et"""
        if len(history) < window:
            return "STABLE", 0.0
        
        recent = history[-window:]
        older = history[-window*2:-window] if len(history) >= window*2 else history[:window]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older) if older else recent_avg
        
        if older_avg == 0:
            return "STABLE", 0.0
        
        change_pct = ((recent_avg - older_avg) / older_avg) * 100
        
        if change_pct > 5:
            return "RISING", min(100, change_pct * 2)
        elif change_pct < -5:
            return "FALLING", max(-100, change_pct * 2)
        else:
            return "STABLE", change_pct
    
    def _is_anomaly(self, value: float, history: List[float], std_threshold: float = 2.0) -> bool:
        """Değer anomali mi kontrol et"""
        if len(history) < 10:
            return False
        
        mean = np.mean(history)
        std = np.std(history)
        
        if std == 0:
            return False
        
        z_score = abs(value - mean) / std
        return z_score > std_threshold
    
    def _generate_context_text(
        self, 
        name: str, 
        value: float, 
        percentile: float,
        trend: str,
        is_extreme: bool
    ) -> str:
        """İnsan okunabilir bağlam metni oluştur"""
        
        if name == "RSI":
            if value < 30:
                return f"OVERSOLD - Son 30 günün en düşük %{100-percentile:.0f}'lik diliminde. Potansiyel dip."
            elif value > 70:
                return f"OVERBOUGHT - Son 30 günün en yüksek %{percentile:.0f}'lik diliminde. Dikkat!"
            else:
                return f"Nötr bölgede. Son 30 günün %{percentile:.0f}'lik diliminde."
        
        elif name == "Fear & Greed":
            if value < 25:
                return f"EXTREME FEAR - Historik olarak bu seviyelerde genellikle dip oluşur."
            elif value > 75:
                return f"EXTREME GREED - Historik olarak bu seviyelerde genellikle tepe oluşur."
            else:
                return f"Normal aralıkta. Son 30 günün %{percentile:.0f}'lik diliminde."
        
        elif name == "Funding Rate":
            if value > 0.1:
                return f"Çok YÜKSEK funding - Aşırı long pozisyon. Short squeeze riski düşük, long liquidation riski yüksek."
            elif value < -0.05:
                return f"NEGATİF funding - Aşırı short pozisyon. Short squeeze potansiyeli!"
            else:
                return f"Normal funding. Piyasa dengeli."
        
        elif name == "L/S Ratio":
            if value > 2:
                return f"EXTREME LONG kalabalık ({value:.1f}x). Contrarian kısa vadeli short fırsatı?"
            elif value < 0.5:
                return f"EXTREME SHORT kalabalık ({value:.1f}x). Contrarian long fırsatı?"
            else:
                return f"Dengeli Long/Short oranı."
        
        elif name == "Orderbook":
            if value > 50:
                return f"Güçlü ALIM baskısı (+{value:.0f}%). Buyers kontrolde."
            elif value < -50:
                return f"Güçlü SATIŞ baskısı ({value:.0f}%). Sellers kontrolde."
            else:
                return f"Dengeli orderbook."
        
        else:
            return f"Son 30 günün %{percentile:.0f}'lik diliminde. Trend: {trend}"
    
    def build_indicator_context(
        self,
        name: str,
        current_value: float,
        history: List[float] = None
    ) -> IndicatorContext:
        """Tek gösterge için zenginleştirilmiş bağlam oluştur"""
        history = history or []
        
        percentile = self._calculate_percentile(current_value, history)
        trend, trend_strength = self._detect_trend(history)
        is_extreme = percentile < 10 or percentile > 90
        is_anomaly = self._is_anomaly(current_value, history)
        
        context_text = self._generate_context_text(
            name, current_value, percentile, trend, is_extreme
        )
        
        return IndicatorContext(
            name=name,
            current_value=current_value,
            percentile_30d=percentile,
            trend=trend,
            trend_strength=trend_strength,
            is_extreme=is_extreme,
            is_anomaly=is_anomaly,
            context_text=context_text
        )
    
    async def build_full_context(
        self,
        symbol: str,
        current_price: float,
        raw_data: Dict,
        history_data: Dict[str, List[float]] = None
    ) -> EnrichedMarketContext:
        """Tam zenginleştirilmiş bağlam oluştur"""
        
        history = history_data or {}
        
        context = EnrichedMarketContext(
            symbol=symbol,
            current_price=current_price,
            timestamp=datetime.now()
        )
        
        # Build enriched indicators
        if 'rsi' in raw_data:
            context.rsi = self.build_indicator_context(
                "RSI", raw_data['rsi'], history.get('rsi', [])
            )
        
        if 'fear_greed' in raw_data:
            context.fear_greed = self.build_indicator_context(
                "Fear & Greed", raw_data['fear_greed'], history.get('fear_greed', [])
            )
        
        if 'funding_rate' in raw_data:
            context.funding_rate = self.build_indicator_context(
                "Funding Rate", raw_data['funding_rate'] * 100, history.get('funding_rate', [])
            )
        
        if 'ls_ratio' in raw_data:
            context.ls_ratio = self.build_indicator_context(
                "L/S Ratio", raw_data['ls_ratio'], history.get('ls_ratio', [])
            )
        
        if 'orderbook_score' in raw_data:
            context.orderbook_imbalance = self.build_indicator_context(
                "Orderbook", raw_data['orderbook_score'], history.get('orderbook', [])
            )
        
        if 'whale_score' in raw_data:
            context.whale_activity = self.build_indicator_context(
                "Whale Activity", raw_data['whale_score'], history.get('whale', [])
            )
        
        # Market structure
        context.market_regime = raw_data.get('regime', 'UNKNOWN')
        
        # Resistance/Support
        context.key_resistance_levels = raw_data.get('resistance_levels', [])
        context.key_support_levels = raw_data.get('support_levels', [])
        
        if context.key_resistance_levels:
            nearest_resistance = min(context.key_resistance_levels, key=lambda x: x if x > current_price else float('inf'))
            if nearest_resistance > current_price:
                context.distance_to_resistance_pct = ((nearest_resistance - current_price) / current_price) * 100
        
        if context.key_support_levels:
            nearest_support = max(context.key_support_levels, key=lambda x: x if x < current_price else float('-inf'))
            if nearest_support < current_price:
                context.distance_to_support_pct = ((current_price - nearest_support) / current_price) * 100
        
        # News
        context.recent_news = raw_data.get('recent_news', [])
        
        # Whale movements
        context.whale_movements = raw_data.get('whale_movements', [])
        context.net_whale_flow_24h = raw_data.get('net_whale_flow', 0)
        
        logger.info(f"📊 Built enriched context for {symbol}")
        
        return context


# Singleton
_context_builder: Optional[EnrichedContextBuilder] = None

def get_enriched_context_builder() -> EnrichedContextBuilder:
    global _context_builder
    if _context_builder is None:
        _context_builder = EnrichedContextBuilder()
    return _context_builder
