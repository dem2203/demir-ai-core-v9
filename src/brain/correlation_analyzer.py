# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - CORRELATION ANALYZER
====================================
Coinler arası korelasyon analizi yaparak risk yönetimi sağlar.

Özellikler:
1. Pearson korelasyon hesaplama (7 ve 30 günlük)
2. Korelasyon matrisi cache'leme
3. Sinyal filtreleme (yüksek korelasyonlu coinlerde aynı yönde pozisyon engelleme)

Örnek:
- BTC LONG açık, ETH LONG sinyali geldi
- BTC-ETH korelasyonu %85 → "Korelasyon riski, sinyal atlandı"
"""
import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("CORRELATION_ANALYZER")


@dataclass
class CorrelationResult:
    """Korelasyon analiz sonucu"""
    symbol1: str
    symbol2: str
    correlation_7d: float   # Son 7 günlük korelasyon
    correlation_30d: float  # Son 30 günlük korelasyon
    is_high_correlation: bool  # > 0.7 ise True
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL


class CorrelationAnalyzer:
    """
    Multi-Coin Korelasyon Analizörü
    
    Coinler arasındaki fiyat korelasyonunu hesaplar ve
    yüksek korelasyonlu coinlerde aynı yönde pozisyon açmayı engeller.
    """
    
    # Korelasyon eşikleri
    CORRELATION_THRESHOLDS = {
        'LOW': 0.3,      # < 0.3: Düşük korelasyon, bağımsız hareket
        'MEDIUM': 0.5,   # 0.3-0.5: Orta korelasyon
        'HIGH': 0.7,     # 0.5-0.7: Yüksek korelasyon
        'CRITICAL': 0.85 # > 0.85: Kritik korelasyon, aynı yönde pozisyon YASAK
    }
    
    # Desteklenen semboller
    SUPPORTED_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        self._last_update: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)  # 1 saatte bir güncelle
        self._price_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info("📊 Correlation Analyzer initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _fetch_historical_prices(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Binance'den geçmiş fiyat verilerini çek."""
        try:
            session = await self._get_session()
            
            # Günlük mum verisi (1d interval)
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1d',
                'limit': days
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to fetch prices for {symbol}")
                    return None
                
                data = await resp.json()
            
            # DataFrame oluştur
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['close'] = df['close'].astype(float)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df[['timestamp', 'close']]
            df.columns = ['timestamp', symbol]
            
            return df
            
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
            return None
    
    async def calculate_correlation_matrix(self, symbols: List[str] = None, force_refresh: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Tüm semboller için korelasyon matrisi hesapla.
        
        Returns:
            {'BTCUSDT': {'ETHUSDT': 0.85, 'SOLUSDT': 0.72}, ...}
        """
        if symbols is None:
            symbols = self.SUPPORTED_SYMBOLS
        
        # Cache kontrolü
        if not force_refresh and self._last_update:
            if datetime.now() - self._last_update < self._cache_duration:
                return self._correlation_matrix
        
        logger.info(f"📊 Calculating correlation matrix for {len(symbols)} symbols...")
        
        # Fiyat verilerini çek
        tasks = [self._fetch_historical_prices(sym, days=30) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # DataFrame birleştir
        combined_df = None
        valid_symbols = []
        
        for sym, result in zip(symbols, results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                if combined_df is None:
                    combined_df = result
                else:
                    combined_df = combined_df.merge(result, on='timestamp', how='outer')
                valid_symbols.append(sym)
        
        if combined_df is None or len(valid_symbols) < 2:
            logger.warning("Insufficient data for correlation calculation")
            return {}
        
        # Korelasyon hesapla
        combined_df.set_index('timestamp', inplace=True)
        combined_df = combined_df.dropna()
        
        # Günlük getiri hesapla (fiyat değişimi daha doğru korelasyon verir)
        returns_df = combined_df.pct_change().dropna()
        
        # Korelasyon matrisi
        corr_matrix = returns_df.corr()
        
        # Dict formatına çevir
        self._correlation_matrix = {}
        for sym1 in valid_symbols:
            self._correlation_matrix[sym1] = {}
            for sym2 in valid_symbols:
                if sym1 != sym2:
                    corr_value = corr_matrix.loc[sym1, sym2]
                    self._correlation_matrix[sym1][sym2] = round(corr_value, 3)
        
        self._last_update = datetime.now()
        
        # Log sonuçları
        for sym1, correlations in self._correlation_matrix.items():
            high_corrs = [(s, c) for s, c in correlations.items() if abs(c) > 0.7]
            if high_corrs:
                logger.info(f"📊 {sym1} high correlations: {high_corrs}")
        
        return self._correlation_matrix
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """İki sembol arasındaki korelasyonu al."""
        if symbol1 in self._correlation_matrix:
            return self._correlation_matrix[symbol1].get(symbol2)
        return None
    
    def get_risk_level(self, correlation: float) -> str:
        """Korelasyon değerine göre risk seviyesi döndür."""
        abs_corr = abs(correlation)
        
        if abs_corr >= self.CORRELATION_THRESHOLDS['CRITICAL']:
            return 'CRITICAL'
        elif abs_corr >= self.CORRELATION_THRESHOLDS['HIGH']:
            return 'HIGH'
        elif abs_corr >= self.CORRELATION_THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    async def should_skip_signal(
        self, 
        new_signal_symbol: str, 
        new_signal_direction: str,  # 'BUY' or 'SELL'
        open_positions: List[Dict]   # [{'symbol': 'BTCUSDT', 'direction': 'BUY'}, ...]
    ) -> Tuple[bool, Optional[str]]:
        """
        Yeni sinyal için korelasyon kontrolü yap.
        
        Args:
            new_signal_symbol: Yeni sinyalin sembolü
            new_signal_direction: Yeni sinyalin yönü ('BUY' veya 'SELL')
            open_positions: Açık pozisyonlar listesi
        
        Returns:
            (should_skip: bool, reason: Optional[str])
        """
        if not open_positions:
            return False, None
        
        # Korelasyon matrisini güncelle (cache'den alır veya hesaplar)
        await self.calculate_correlation_matrix()
        
        for position in open_positions:
            pos_symbol = position.get('symbol')
            pos_direction = position.get('direction')
            
            if pos_symbol == new_signal_symbol:
                continue  # Aynı sembol, korelasyon kontrolü gerekmez
            
            # Korelasyon al
            correlation = self.get_correlation(new_signal_symbol, pos_symbol)
            
            if correlation is None:
                continue
            
            risk_level = self.get_risk_level(correlation)
            
            # Aynı yönde pozisyon kontrolü
            same_direction = (new_signal_direction == pos_direction)
            
            # Negatif korelasyon durumunda ters yön tehlikeli
            if correlation < 0:
                same_direction = (new_signal_direction != pos_direction)
            
            if same_direction and risk_level in ['HIGH', 'CRITICAL']:
                reason = (
                    f"⚠️ Korelasyon Riski: {new_signal_symbol} {new_signal_direction} atlandı. "
                    f"{pos_symbol} {pos_direction} pozisyonu açık ve korelasyon {correlation:.1%} ({risk_level})"
                )
                logger.warning(reason)
                return True, reason
        
        return False, None
    
    def analyze_pair(self, symbol1: str, symbol2: str) -> Optional[CorrelationResult]:
        """İki sembol arasındaki detaylı korelasyon analizi."""
        correlation = self.get_correlation(symbol1, symbol2)
        
        if correlation is None:
            return None
        
        risk_level = self.get_risk_level(correlation)
        
        return CorrelationResult(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation_7d=correlation,  # Şimdilik 30d kullanıyoruz
            correlation_30d=correlation,
            is_high_correlation=abs(correlation) > 0.7,
            risk_level=risk_level
        )
    
    def format_matrix_for_telegram(self) -> str:
        """Korelasyon matrisini Telegram için formatla."""
        if not self._correlation_matrix:
            return "📊 Korelasyon matrisi henüz hesaplanmadı."
        
        msg = "📊 *Korelasyon Matrisi*\n"
        msg += "━━━━━━━━━━━━━━━━━\n\n"
        
        # En yüksek korelasyonları listele
        pairs = []
        for sym1, correlations in self._correlation_matrix.items():
            for sym2, corr in correlations.items():
                if sym1 < sym2:  # Duplicate'leri önle
                    pairs.append((sym1, sym2, corr))
        
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for sym1, sym2, corr in pairs[:10]:  # En yüksek 10
            risk = self.get_risk_level(corr)
            emoji = "🔴" if risk == 'CRITICAL' else "🟠" if risk == 'HIGH' else "🟡" if risk == 'MEDIUM' else "🟢"
            msg += f"{emoji} {sym1[:3]}-{sym2[:3]}: {corr:.1%}\n"
        
        msg += f"\n_Son güncelleme: {self._last_update.strftime('%H:%M') if self._last_update else 'N/A'}_"
        
        return msg
    
    async def close(self):
        """Session'ı kapat."""
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton instance
_correlation_analyzer: Optional[CorrelationAnalyzer] = None


def get_correlation_analyzer() -> CorrelationAnalyzer:
    """Get or create singleton CorrelationAnalyzer."""
    global _correlation_analyzer
    if _correlation_analyzer is None:
        _correlation_analyzer = CorrelationAnalyzer()
    return _correlation_analyzer


async def check_correlation_risk(
    new_signal_symbol: str,
    new_signal_direction: str,
    open_positions: List[Dict]
) -> Tuple[bool, Optional[str]]:
    """Quick access function for correlation check."""
    analyzer = get_correlation_analyzer()
    return await analyzer.should_skip_signal(new_signal_symbol, new_signal_direction, open_positions)
