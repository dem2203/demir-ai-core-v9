"""
DEMIR AI - Advanced Predictive Indicators
Fırsatları ve riskleri ÖNCE tespit eden gelişmiş göstergeler.

Indicators:
1. Momentum Spike Detector - Ani hacim artışı + durgun fiyat = Büyük hareket yakın
2. OI Velocity - Açık pozisyon değişim hızı
3. Funding Rate Extreme - Aşırı funding = Ters hareket riski
4. Whale Alert - Büyük transferler
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("PREDICTIVE_INDICATORS")


@dataclass
class PredictiveAlert:
    """Tahmine dayalı uyarı yapısı"""
    symbol: str
    alert_type: str  # MOMENTUM_SPIKE, OI_VELOCITY, FUNDING_EXTREME, WHALE_MOVE
    severity: str    # HIGH, MEDIUM, LOW
    direction: str   # BULLISH, BEARISH, NEUTRAL
    title: str
    detail: str
    action: str
    timestamp: datetime


class PredictiveIndicators:
    """
    Gelişmiş Tahmin Göstergeleri
    
    Fırsatları ve riskleri ÖNCE tespit eder.
    """
    
    # Eşik değerleri
    MOMENTUM_VOLUME_MULTIPLIER = 2.5   # Normal hacmin 2.5x üstü
    MOMENTUM_PRICE_THRESHOLD = 0.5     # Fiyat %0.5'den az değişmiş
    OI_VELOCITY_THRESHOLD = 5.0        # 1 saatte %5+ OI değişimi
    FUNDING_HIGH_THRESHOLD = 0.05      # %0.05+ funding (long aşırı)
    FUNDING_LOW_THRESHOLD = -0.03      # %-0.03'den düşük (short aşırı)
    WHALE_USD_THRESHOLD = 1_000_000    # 1M$ üstü transfer
    
    def __init__(self):
        self.last_check = datetime.now()
        self.historical_oi = {}  # Cache for OI history
        self.historical_funding = {}  # Cache for funding history
        
    def analyze_all(self, symbol: str, current_data: Dict, historical_data: List[Dict] = None) -> List[PredictiveAlert]:
        """
        Tüm öngörücü göstergeleri analiz et.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            current_data: Current market data dict
            historical_data: List of recent candles for momentum analysis
        
        Returns:
            List of PredictiveAlert objects
        """
        alerts = []
        
        # 1. Momentum Spike Detection
        momentum_alert = self.detect_momentum_spike(symbol, current_data, historical_data)
        if momentum_alert:
            alerts.append(momentum_alert)
        
        # 2. OI Velocity
        oi_alert = self.detect_oi_velocity(symbol, current_data)
        if oi_alert:
            alerts.append(oi_alert)
        
        # 3. Funding Rate Extreme
        funding_alert = self.detect_funding_extreme(symbol, current_data)
        if funding_alert:
            alerts.append(funding_alert)
        
        # 4. Whale Alert - Büyük alım/satım ve transferler
        whale_alerts = self.detect_whale_moves(symbol, current_data)
        alerts.extend(whale_alerts)
        
        return alerts
    
    # =========================================
    # 1. MOMENTUM SPIKE DETECTOR
    # =========================================
    def detect_momentum_spike(self, symbol: str, current_data: Dict, 
                               historical_data: List[Dict] = None) -> Optional[PredictiveAlert]:
        """
        Hacim ani artıyor ama fiyat durağan = Büyük hareket yaklaşıyor!
        
        Bu genellikle akıllı paranın sessizce pozisyon aldığı anlamına gelir.
        """
        try:
            # Get volume data
            current_volume = current_data.get('volume', 0)
            avg_volume = current_data.get('avg_volume_20', 0)
            
            if not avg_volume or avg_volume == 0:
                # Calculate from historical if available
                if historical_data and len(historical_data) >= 20:
                    avg_volume = sum(d.get('volume', 0) for d in historical_data[-20:]) / 20
                else:
                    return None
            
            # Calculate price change (last 5 candles)
            price_change = 0
            if historical_data and len(historical_data) >= 5:
                price_5_ago = historical_data[-5].get('close', 0)
                price_now = current_data.get('price', 0) or historical_data[-1].get('close', 0)
                if price_5_ago > 0:
                    price_change = abs((price_now - price_5_ago) / price_5_ago * 100)
            
            # Volume spike detection
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio > self.MOMENTUM_VOLUME_MULTIPLIER and price_change < self.MOMENTUM_PRICE_THRESHOLD:
                # Strong momentum spike with flat price = Breakout incoming!
                severity = "HIGH" if volume_ratio > 4 else "MEDIUM"
                
                # Try to determine direction from order book imbalance
                ob_imbalance = current_data.get('orderbook_imbalance', 0)
                if ob_imbalance > 0.2:
                    direction = "BULLISH"
                    action = "Yukarı kırılım bekleniyor - Long hazırlığı yap!"
                elif ob_imbalance < -0.2:
                    direction = "BEARISH"
                    action = "Aşağı kırılım bekleniyor - Short veya bekle!"
                else:
                    direction = "NEUTRAL"
                    action = "Yön belirsiz - Kırılımı bekle!"
                
                return PredictiveAlert(
                    symbol=symbol,
                    alert_type="MOMENTUM_SPIKE",
                    severity=severity,
                    direction=direction,
                    title=f"⚡ {symbol} Momentum Spike!",
                    detail=f"Hacim {volume_ratio:.1f}x normal, fiyat %{price_change:.1f} değişti",
                    action=action,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.warning(f"Momentum spike detection failed: {e}")
        
        return None
    
    # =========================================
    # 2. OI VELOCITY (Açık Pozisyon Değişim Hızı)
    # =========================================
    def detect_oi_velocity(self, symbol: str, current_data: Dict) -> Optional[PredictiveAlert]:
        """
        OI çok hızlı artıyorsa = Volatilite patlaması yakın!
        
        Trader'lar agresif pozisyon açıyor demektir.
        """
        try:
            derivatives = current_data.get('derivatives', {})
            current_oi = derivatives.get('open_interest', 0)
            
            if current_oi <= 0:
                return None
            
            # Check historical OI
            symbol_key = symbol.replace('/', '')
            
            # Get 1-hour ago OI from cache or API
            oi_1h_ago = self.historical_oi.get(f"{symbol_key}_1h", 0)
            
            if oi_1h_ago > 0:
                oi_change_pct = ((current_oi - oi_1h_ago) / oi_1h_ago) * 100
                
                if abs(oi_change_pct) > self.OI_VELOCITY_THRESHOLD:
                    if oi_change_pct > 0:
                        # OI increasing rapidly
                        severity = "HIGH" if oi_change_pct > 10 else "MEDIUM"
                        direction = "NEUTRAL"  # OI spike can go either way
                        
                        return PredictiveAlert(
                            symbol=symbol,
                            alert_type="OI_VELOCITY",
                            severity=severity,
                            direction=direction,
                            title=f"📊 {symbol} OI Spike! (+{oi_change_pct:.1f}%)",
                            detail=f"1 saatte ${current_oi/1e9:.2f}B'a yükseldi",
                            action="⚠️ Volatilite patlaması bekleniyor - Stop'ları geniş tut veya bekle!",
                            timestamp=datetime.now()
                        )
                    else:
                        # OI decreasing rapidly - positions closing
                        return PredictiveAlert(
                            symbol=symbol,
                            alert_type="OI_VELOCITY",
                            severity="MEDIUM",
                            direction="NEUTRAL",
                            title=f"📉 {symbol} OI Düşüyor ({oi_change_pct:.1f}%)",
                            detail=f"Trader'lar pozisyon kapatıyor",
                            action="Trend zayıflıyor - Yeni pozisyonlara dikkat!",
                            timestamp=datetime.now()
                        )
            
            # Update cache for next check
            self.historical_oi[f"{symbol_key}_1h"] = current_oi
            
        except Exception as e:
            logger.warning(f"OI velocity detection failed: {e}")
        
        return None
    
    # =========================================
    # 3. FUNDING RATE EXTREME
    # =========================================
    def detect_funding_extreme(self, symbol: str, current_data: Dict) -> Optional[PredictiveAlert]:
        """
        Aşırı yüksek funding = Long'lar ödeme yapıyor, düşüş riski
        Aşırı düşük funding = Short'lar ödeme yapıyor, yükseliş fırsatı
        """
        try:
            derivatives = current_data.get('derivatives', {})
            funding_rate = derivatives.get('funding_rate', 0)
            
            # Also check from snapshot
            if not funding_rate:
                funding_rate = current_data.get('funding_rate', 0)
            
            if funding_rate > self.FUNDING_HIGH_THRESHOLD:
                # Extreme positive funding - longs paying shorts
                return PredictiveAlert(
                    symbol=symbol,
                    alert_type="FUNDING_EXTREME",
                    severity="HIGH",
                    direction="BEARISH",
                    title=f"🔴 {symbol} Funding Extreme! ({funding_rate:.3f}%)",
                    detail=f"Long'lar short'lara %{funding_rate:.3f} ödüyor",
                    action="⚠️ Aşırı long kalabalığı - Düşüş riski yüksek! Kar al veya hedge yap!",
                    timestamp=datetime.now()
                )
            
            elif funding_rate < self.FUNDING_LOW_THRESHOLD:
                # Extreme negative funding - shorts paying longs
                return PredictiveAlert(
                    symbol=symbol,
                    alert_type="FUNDING_EXTREME",
                    severity="HIGH",
                    direction="BULLISH",
                    title=f"🟢 {symbol} Negatif Funding! ({funding_rate:.3f}%)",
                    detail=f"Short'lar long'lara %{abs(funding_rate):.3f} ödüyor",
                    action="✅ Aşırı short kalabalığı - Short squeeze fırsatı! Long düşün!",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.warning(f"Funding extreme detection failed: {e}")
        
        return None
    
    # =========================================
    # 4. WHALE ALERT (Büyük İşlem ve Transfer Takibi)
    # =========================================
    def detect_whale_moves(self, symbol: str, current_data: Dict = None) -> List[PredictiveAlert]:
        """
        Büyük cüzdan transferlerini ve işlemlerini tespit et.
        
        1. Binance aggTrades -> Büyük alım/satım emirleri
        2. whale-alert.io -> Exchange transferleri (premium özellik)
        
        Exchange'e giriş = Satış hazırlığı
        Exchange'den çıkış = HODL (bullish)
        """
        alerts = []
        
        try:
            clean_symbol = symbol.replace('/', '')
            
            # 1. Binance Aggravated Trades - Son 1 dakikadaki büyük işlemler
            whale_trades = self._fetch_large_trades(clean_symbol)
            
            if whale_trades:
                total_buy_volume = sum(t['qty'] for t in whale_trades if not t['isBuyerMaker'])
                total_sell_volume = sum(t['qty'] for t in whale_trades if t['isBuyerMaker'])
                
                # Check for significant imbalance
                if total_buy_volume > 0 or total_sell_volume > 0:
                    total = total_buy_volume + total_sell_volume
                    buy_ratio = total_buy_volume / total if total > 0 else 0
                    
                    if buy_ratio > 0.7:  # %70+ alım
                        price = current_data.get('price', 0) if current_data else 0
                        usd_volume = total_buy_volume * price if price else total_buy_volume
                        
                        if usd_volume > self.WHALE_USD_THRESHOLD:
                            alerts.append(PredictiveAlert(
                                symbol=symbol,
                                alert_type="WHALE_BUY",
                                severity="HIGH",
                                direction="BULLISH",
                                title=f"🐋 {symbol} Whale ALIM!",
                                detail=f"${usd_volume/1e6:.2f}M büyük alım tespit edildi",
                                action="✅ Balinalar alım yapıyor - Yükseliş sinyali!",
                                timestamp=datetime.now()
                            ))
                    
                    elif buy_ratio < 0.3:  # %70+ satım
                        price = current_data.get('price', 0) if current_data else 0
                        usd_volume = total_sell_volume * price if price else total_sell_volume
                        
                        if usd_volume > self.WHALE_USD_THRESHOLD:
                            alerts.append(PredictiveAlert(
                                symbol=symbol,
                                alert_type="WHALE_SELL",
                                severity="HIGH",
                                direction="BEARISH",
                                title=f"🐋 {symbol} Whale SATIŞ!",
                                detail=f"${usd_volume/1e6:.2f}M büyük satış tespit edildi",
                                action="⚠️ Balinalar satıyor - Dikkatli ol!",
                                timestamp=datetime.now()
                            ))
            
            # 2. Blockchain whale transfers (for BTC/ETH)
            if 'BTC' in symbol:
                transfer_alerts = self._check_btc_whale_transfers()
                alerts.extend(transfer_alerts)
            elif 'ETH' in symbol:
                transfer_alerts = self._check_eth_whale_transfers()
                alerts.extend(transfer_alerts)
                
        except Exception as e:
            logger.warning(f"Whale detection failed for {symbol}: {e}")
        
        return alerts
    
    def _fetch_large_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Binance'den son büyük işlemleri çek"""
        try:
            url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={symbol}&limit={limit}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                trades = response.json()
                
                # Filter for large trades (>$50K approximate)
                large_trades = []
                for trade in trades:
                    qty = float(trade['q'])
                    price = float(trade['p'])
                    usd_value = qty * price
                    
                    if usd_value > 50000:  # $50K+
                        large_trades.append({
                            'qty': qty,
                            'price': price,
                            'usd': usd_value,
                            'isBuyerMaker': trade['m'],  # True = Satış, False = Alım
                            'time': trade['T']
                        })
                
                return large_trades
                
        except Exception as e:
            logger.warning(f"Large trades fetch failed: {e}")
        
        return []
    
    def _check_btc_whale_transfers(self) -> List[PredictiveAlert]:
        """BTC whale transferlerini kontrol et (blockchain.info API)"""
        alerts = []
        
        try:
            # Check unconfirmed transactions > 100 BTC
            url = "https://blockchain.info/unconfirmed-transactions?format=json"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                txs = data.get('txs', [])
                
                for tx in txs[:50]:  # Son 50 tx
                    # Calculate total output
                    total_btc = sum(out.get('value', 0) for out in tx.get('out', [])) / 1e8
                    
                    if total_btc > 100:  # 100+ BTC transfer
                        # Check if going to exchange (simplified - would need address labeling)
                        alerts.append(PredictiveAlert(
                            symbol="BTC/USDT",
                            alert_type="WHALE_TRANSFER",
                            severity="MEDIUM",
                            direction="NEUTRAL",
                            title=f"🐋 BTC Büyük Transfer: {total_btc:.0f} BTC",
                            detail=f"~${total_btc * 100000:,.0f} değerinde transfer",
                            action="Takip et - Exchange'e girerse satış baskısı!",
                            timestamp=datetime.now()
                        ))
                        break  # Only one alert per check
                        
        except Exception as e:
            logger.debug(f"BTC whale transfer check failed: {e}")
        
        return alerts
    
    def _check_eth_whale_transfers(self) -> List[PredictiveAlert]:
        """ETH whale transferlerini kontrol et"""
        alerts = []
        
        try:
            # Use Etherscan free API (rate limited)
            # For now, we'll skip this as it requires API key
            # In production, use etherscan.io API or Alchemy
            pass
            
        except Exception as e:
            logger.debug(f"ETH whale transfer check failed: {e}")
        
        return alerts
    
    # =========================================
    # HELPER: Fetch Real-Time Funding Rate
    # =========================================
    def fetch_funding_rate(self, symbol: str) -> float:
        """Binance Futures'tan funding rate çek"""
        try:
            clean_symbol = symbol.replace('/', '')
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={clean_symbol}&limit=1"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return float(data[0]['fundingRate']) * 100  # Convert to percentage
        except Exception as e:
            logger.warning(f"Funding rate fetch failed: {e}")
        return 0
    
    # =========================================
    # HELPER: Fetch OI Change
    # =========================================
    def fetch_oi_history(self, symbol: str) -> Dict:
        """Binance Futures'tan OI geçmişi çek"""
        try:
            clean_symbol = symbol.replace('/', '')
            url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={clean_symbol}&period=1h&limit=2"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if len(data) >= 2:
                    return {
                        'current': float(data[-1]['sumOpenInterest']),
                        'previous': float(data[-2]['sumOpenInterest']),
                        'change_pct': ((float(data[-1]['sumOpenInterest']) - float(data[-2]['sumOpenInterest'])) 
                                      / float(data[-2]['sumOpenInterest']) * 100) if float(data[-2]['sumOpenInterest']) > 0 else 0
                    }
        except Exception as e:
            logger.warning(f"OI history fetch failed: {e}")
        return {}
    
    # =========================================
    # FORMAT FOR TELEGRAM
    # =========================================
    def format_alerts_for_telegram(self, alerts: List[PredictiveAlert]) -> str:
        """Telegram için uyarıları formatla"""
        if not alerts:
            return ""
        
        msg = "🔮 *ÖNGÖRÜCÜLERNALİZ*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for alert in alerts:
            severity_emoji = "🔴" if alert.severity == "HIGH" else "🟡" if alert.severity == "MEDIUM" else "🟢"
            direction_emoji = "📈" if alert.direction == "BULLISH" else "📉" if alert.direction == "BEARISH" else "↔️"
            
            msg += f"{severity_emoji} *{alert.title}*\n"
            msg += f"  {direction_emoji} _{alert.detail}_\n"
            msg += f"  💡 {alert.action}\n\n"
        
        msg += f"⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        return msg
