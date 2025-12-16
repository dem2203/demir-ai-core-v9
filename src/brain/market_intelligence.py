"""
DEMIR AI - Market Intelligence Report System
Her 15 dakikada fırsat/risk tarar, bulursa bildirir.
1 saat içinde bulamazsa saatlik özet gönderir.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("MARKET_INTELLIGENCE")


class MarketIntelligence:
    """
    Akıllı Bildirim Sistemi:
    - Her 15dk fırsat/risk tarar
    - Bulursa anında Telegram'a gönderir
    - 1 saat boş geçerse saatlik özet gönderir
    """
    
    # Fırsat/Risk eşikleri
    OPPORTUNITY_THRESHOLDS = {
        'rsi_oversold': 30,        # RSI < 30 = Alım fırsatı
        'rsi_overbought': 70,      # RSI > 70 = Satış riski
        'ls_ratio_high': 1.5,      # L/S > 1.5 = Long kalabalığı (risk)
        'ls_ratio_low': 0.7,       # L/S < 0.7 = Short sıkışması (fırsat)
        'oi_change_pct': 5,        # OI %5+ değişim = Volatilite riski
        'funding_extreme': 0.05,   # Funding > 0.05% = Ekstrem
        'mtf_confluence': 0.7,     # MTF uyum > 70% = Güçlü sinyal
        'whale_imbalance': 0.3,    # Order book dengesizliği
    }
    
    def __init__(self):
        self.last_opportunity_time = datetime.now()
        self.last_15min_check = datetime.now()
        self.last_hourly_report = datetime.now() - timedelta(hours=1)  # İlk rapor hemen gider
        self.check_interval = 15  # dakika
        self.hourly_fallback = 60  # dakika
        
    def should_run_15min_check(self) -> bool:
        """15 dakika geçti mi kontrol et"""
        elapsed = (datetime.now() - self.last_15min_check).total_seconds() / 60
        return elapsed >= self.check_interval
    
    def should_send_hourly_fallback(self) -> bool:
        """1 saat fırsat bulunamadı mı kontrol et"""
        time_since_opportunity = (datetime.now() - self.last_opportunity_time).total_seconds() / 60
        time_since_hourly = (datetime.now() - self.last_hourly_report).total_seconds() / 60
        return time_since_opportunity >= self.hourly_fallback and time_since_hourly >= self.hourly_fallback
    
    def scan_for_opportunities(self, snapshots: Dict) -> List[Dict]:
        """
        Tüm coinler için fırsat ve risk taraması yap.
        
        Returns:
            List of opportunities/risks found
        """
        opportunities = []
        
        for symbol, data in snapshots.items():
            coin_opps = self._analyze_coin(symbol, data)
            opportunities.extend(coin_opps)
        
        if opportunities:
            self.last_opportunity_time = datetime.now()
            logger.info(f"🎯 {len(opportunities)} fırsat/risk bulundu!")
        
        self.last_15min_check = datetime.now()
        return opportunities
    
    def _analyze_coin(self, symbol: str, data: Dict) -> List[Dict]:
        """Tek coin için fırsat/risk analizi"""
        findings = []
        price = data.get('price', 0)
        
        # 1. RSI Analizi
        brain_state = data.get('brain_state', {})
        # RSI approximation from tech_attention
        
        # 2. SMC Analizi
        smc = data.get('smc', {})
        if smc:
            bias = smc.get('bias', 'NEUTRAL')
            strength = smc.get('strength', 0)
            
            if bias == 'BULLISH' and strength > 70:
                findings.append({
                    'symbol': symbol,
                    'type': 'OPPORTUNITY',
                    'category': 'SMC',
                    'title': f'🟢 {symbol} Güçlü Bullish SMC',
                    'detail': f'Bias: {bias}, Güç: {strength}%',
                    'action': f'Long pozisyon düşünülebilir',
                    'price': price
                })
            elif bias == 'BEARISH' and strength > 70:
                findings.append({
                    'symbol': symbol,
                    'type': 'RISK',
                    'category': 'SMC',
                    'title': f'🔴 {symbol} Güçlü Bearish SMC',
                    'detail': f'Bias: {bias}, Güç: {strength}%',
                    'action': f'Short veya bekle',
                    'price': price
                })
            
            # Order Block yakınlığı
            obs = smc.get('order_blocks', [])
            if obs:
                for ob in obs[:2]:  # İlk 2 OB
                    ob_price = ob.get('price', 0)
                    distance_pct = abs(price - ob_price) / price * 100 if price > 0 else 0
                    if distance_pct < 1:  # %1'den yakın
                        ob_type = ob.get('type', 'Unknown')
                        findings.append({
                            'symbol': symbol,
                            'type': 'OPPORTUNITY',
                            'category': 'ORDER_BLOCK',
                            'title': f'🎯 {symbol} Order Block Yakın!',
                            'detail': f'{ob_type} OB ${ob_price:,.0f} (%.{distance_pct:.1f} uzakta)',
                            'action': f'Reaksiyon için izle',
                            'price': price
                        })
        
        # 3. MTF Confluence
        mtf = data.get('mtf', {})
        if mtf:
            confluence = mtf.get('confluence_score', 0)
            if confluence > 75:
                trend_1h = mtf.get('trend_1h', 'N/A')
                findings.append({
                    'symbol': symbol,
                    'type': 'OPPORTUNITY',
                    'category': 'MTF',
                    'title': f'✅ {symbol} Yüksek MTF Uyumu',
                    'detail': f'Confluence: {confluence}%, Trend: {trend_1h}',
                    'action': f'Trend yönünde işlem güçlü',
                    'price': price
                })
        
        # 4. Whale Activity
        whale_support = data.get('whale_support', 0)
        whale_resistance = data.get('whale_resistance', 0)
        
        if whale_support > 0:
            support_distance = (price - whale_support) / price * 100 if price > 0 else 0
            if 0 < support_distance < 2:  # %2'den yakın
                findings.append({
                    'symbol': symbol,
                    'type': 'OPPORTUNITY',
                    'category': 'WHALE',
                    'title': f'🐋 {symbol} Whale Desteği Yakın',
                    'detail': f'Destek: ${whale_support:,.0f} (%.{support_distance:.1f} aşağıda)',
                    'action': f'Güçlü destek noktası',
                    'price': price
                })
        
        if whale_resistance > 0:
            resist_distance = (whale_resistance - price) / price * 100 if price > 0 else 0
            if 0 < resist_distance < 2:  # %2'den yakın
                findings.append({
                    'symbol': symbol,
                    'type': 'RISK',
                    'category': 'WHALE',
                    'title': f'🐋 {symbol} Whale Direnci Yakın',
                    'detail': f'Direnç: ${whale_resistance:,.0f} (%.{resist_distance:.1f} yukarıda)',
                    'action': f'Kar al noktası olabilir',
                    'price': price
                })
        
        # 5. Wyckoff Phase
        wyckoff = data.get('wyckoff_phase', '')
        if wyckoff == 'ACCUMULATION':
            findings.append({
                'symbol': symbol,
                'type': 'OPPORTUNITY',
                'category': 'WYCKOFF',
                'title': f'📊 {symbol} Birikim Fazı',
                'detail': f'Akıllı para alım yapıyor',
                'action': f'Orta vadeli long fırsatı',
                'price': price
            })
        elif wyckoff == 'DISTRIBUTION':
            findings.append({
                'symbol': symbol,
                'type': 'RISK',
                'category': 'WYCKOFF',
                'title': f'📊 {symbol} Dağıtım Fazı',
                'detail': f'Akıllı para satıyor',
                'action': f'Dikkatli ol, düşüş gelebilir',
                'price': price
            })
        
        return findings
    
    def format_opportunity_report(self, opportunities: List[Dict]) -> str:
        """Fırsat/risk listesini Telegram formatına çevir"""
        if not opportunities:
            return ""
        
        opps = [o for o in opportunities if o['type'] == 'OPPORTUNITY']
        risks = [o for o in opportunities if o['type'] == 'RISK']
        
        msg = "🔍 *15dk Piyasa Taraması*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        if opps:
            msg += "🟢 *FIRSATLAR:*\n"
            for o in opps[:5]:  # Max 5
                msg += f"• {o['title']}\n"
                msg += f"  _{o['detail']}_\n"
                msg += f"  💡 {o['action']}\n\n"
        
        if risks:
            msg += "🔴 *RİSKLER:*\n"
            for r in risks[:5]:  # Max 5
                msg += f"• {r['title']}\n"
                msg += f"  _{r['detail']}_\n"
                msg += f"  ⚠️ {r['action']}\n\n"
        
        msg += f"⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        return msg
    
    def format_hourly_status(self, snapshots: Dict, live_data: Dict = None) -> str:
        """Saatlik durum özeti - 4 coin değerleri"""
        msg = "📊 *Saatlik Piyasa Özeti*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        msg += "_Son 1 saatte önemli fırsat/risk bulunamadı_\n\n"
        
        msg += "💰 *Güncel Fiyatlar:*\n"
        for symbol, data in snapshots.items():
            price = data.get('price', 0)
            decision = data.get('ai_decision', 'NEUTRAL')
            confidence = data.get('ai_confidence', 0)
            
            emoji = "🟢" if decision == "BUY" else "🔴" if decision == "SELL" else "⚪"
            msg += f"{emoji} {symbol}: `${price:,.2f}`\n"
            msg += f"   AI: {decision} ({confidence:.0f}%)\n"
        
        # Derivatives data if available
        if live_data:
            msg += "\n📈 *Piyasa Durumu:*\n"
            oi = live_data.get('open_interest', 0)
            ls = live_data.get('long_short_ratio', 0)
            btc_d = live_data.get('btc_dominance', 0)
            
            if oi > 0:
                msg += f"• Open Interest: ${oi/1e9:.2f}B\n"
            if ls > 0:
                msg += f"• L/S Ratio: {ls:.2f}\n"
            if btc_d > 0:
                msg += f"• BTC Dominance: {btc_d:.1f}%\n"
        
        msg += f"\n⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        self.last_hourly_report = datetime.now()
        return msg
