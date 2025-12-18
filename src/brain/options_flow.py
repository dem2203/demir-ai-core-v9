# -*- coding: utf-8 -*-
"""
DEMIR AI - Options Flow Analyzer
Deribit ve diğer kaynaklardan opsiyon verisi analizi.

Options Flow = "Smart Money" göstergesi
- Yüksek Call/Put oranı = Bullish
- Yüksek Put/Call oranı = Bearish
- Max Pain = Fiyatın çekileceği seviye
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("OPTIONS_FLOW")


@dataclass
class OptionsData:
    """Opsiyon verileri"""
    call_volume: float
    put_volume: float
    call_oi: float  # Open Interest
    put_oi: float
    call_put_ratio: float
    max_pain: float
    iv_rank: float  # Implied Volatility Rank
    timestamp: datetime


class OptionsFlowAnalyzer:
    """
    Opsiyon Akış Analizi
    
    Veri Kaynakları:
    - Deribit API (ücretsiz)
    - Laevitas (ücretsiz tier)
    
    Sinyaller:
    - Call/Put > 1.5: Bullish
    - Put/Call > 1.5: Bearish
    - Max Pain'e yaklaşma: Fiyat çekimi
    """
    
    # Deribit API
    DERIBIT_API = "https://www.deribit.com/api/v2/public"
    
    # Laevitas (backup)
    LAEVITAS_API = "https://api.laevitas.ch"
    
    def __init__(self):
        self.cache: Dict[str, OptionsData] = {}
        self.cache_time = 300  # 5 dakika cache
    
    def get_deribit_options(self, currency: str = "BTC") -> Optional[Dict]:
        """Deribit'ten opsiyon verilerini çek."""
        try:
            # Get instruments
            instruments_url = f"{self.DERIBIT_API}/get_instruments"
            resp = requests.get(instruments_url, params={
                'currency': currency,
                'kind': 'option',
                'expired': 'false'
            }, timeout=10)
            
            if resp.status_code != 200:
                return None
            
            instruments = resp.json().get('result', [])
            
            # Sadece yakın vadeli opsiyonları al (7 gün içinde expire olan)
            now = datetime.now()
            week_later = now + timedelta(days=7)
            
            near_term = []
            for inst in instruments:
                exp_ts = inst.get('expiration_timestamp', 0) / 1000
                exp_date = datetime.fromtimestamp(exp_ts) if exp_ts > 0 else now
                if exp_date <= week_later:
                    near_term.append(inst)
            
            # Call ve Put ayrımı
            calls = [i for i in near_term if i.get('option_type') == 'call']
            puts = [i for i in near_term if i.get('option_type') == 'put']
            
            # Open Interest hesapla
            call_oi = sum(i.get('open_interest', 0) for i in calls)
            put_oi = sum(i.get('open_interest', 0) for i in puts)
            
            # Call/Put ratio
            cp_ratio = call_oi / put_oi if put_oi > 0 else 1
            
            return {
                'call_count': len(calls),
                'put_count': len(puts),
                'call_oi': call_oi,
                'put_oi': put_oi,
                'call_put_ratio': cp_ratio,
                'source': 'deribit'
            }
            
        except Exception as e:
            logger.warning(f"Deribit fetch failed: {e}")
            return None
    
    def get_btc_index_price(self) -> float:
        """Deribit BTC index fiyatı."""
        try:
            resp = requests.get(
                f"{self.DERIBIT_API}/get_index_price",
                params={'index_name': 'btc_usd'},
                timeout=5
            )
            if resp.status_code == 200:
                return resp.json().get('result', {}).get('index_price', 0)
        except:
            pass
        return 0
    
    def calculate_max_pain(self, currency: str = "BTC") -> float:
        """
        Max Pain hesapla.
        Max Pain = Opsiyon yazarlarının en az zarar ettiği fiyat.
        Fiyat genellikle max pain'e doğru çekilir.
        """
        try:
            resp = requests.get(
                f"{self.DERIBIT_API}/get_instruments",
                params={'currency': currency, 'kind': 'option', 'expired': 'false'},
                timeout=10
            )
            
            if resp.status_code != 200:
                return 0
            
            instruments = resp.json().get('result', [])
            
            # En yakın expiry'yi bul
            now = datetime.now()
            
            # Strike fiyatlarını topla
            strikes = set()
            for inst in instruments:
                strike = inst.get('strike', 0)
                if strike > 0:
                    strikes.add(strike)
            
            if not strikes:
                return 0
            
            # Her strike için toplam pain hesapla
            current_price = self.get_btc_index_price()
            if current_price == 0:
                return 0
            
            pain_by_strike = {}
            
            for strike in strikes:
                total_pain = 0
                
                for inst in instruments:
                    inst_strike = inst.get('strike', 0)
                    oi = inst.get('open_interest', 0)
                    opt_type = inst.get('option_type')
                    
                    if opt_type == 'call':
                        # Call pain: max(0, strike - settlement)
                        pain = max(0, strike - inst_strike) * oi
                    else:
                        # Put pain: max(0, settlement - strike)
                        pain = max(0, inst_strike - strike) * oi
                    
                    total_pain += pain
                
                pain_by_strike[strike] = total_pain
            
            # En düşük pain'e sahip strike = Max Pain
            max_pain_strike = min(pain_by_strike, key=pain_by_strike.get)
            
            return max_pain_strike
            
        except Exception as e:
            logger.warning(f"Max pain calculation failed: {e}")
            return 0
    
    def get_iv_rank(self, currency: str = "BTC") -> float:
        """
        IV Rank (0-100)
        Implied Volatility'nin son 1 yıldaki yerine göre sıralaması.
        """
        try:
            resp = requests.get(
                f"{self.DERIBIT_API}/get_historical_volatility",
                params={'currency': currency},
                timeout=5
            )
            
            if resp.status_code != 200:
                return 50
            
            data = resp.json().get('result', [])
            
            if not data:
                return 50
            
            # Son değer ve tarihsel aralık
            vols = [d[1] for d in data if len(d) > 1]
            
            if not vols:
                return 50
            
            current = vols[-1]
            min_vol = min(vols)
            max_vol = max(vols)
            
            if max_vol == min_vol:
                return 50
            
            iv_rank = ((current - min_vol) / (max_vol - min_vol)) * 100
            
            return iv_rank
            
        except Exception as e:
            logger.warning(f"IV rank calculation failed: {e}")
            return 50
    
    def analyze(self, currency: str = "BTC") -> Dict:
        """Tam opsiyon analizi."""
        options = self.get_deribit_options(currency)
        max_pain = self.calculate_max_pain(currency)
        iv_rank = self.get_iv_rank(currency)
        current_price = self.get_btc_index_price()
        
        if options is None:
            return {
                'available': False,
                'reason': 'Could not fetch options data'
            }
        
        cp_ratio = options['call_put_ratio']
        
        # Sinyal belirle - BOOSTED with Max Pain analysis
        confidence = 40  # Base confidence
        signal = 'NEUTRAL'
        bias = 'NEUTRAL'
        
        # C/P Ratio signal
        if cp_ratio > 1.5:
            signal = 'LONG'
            bias = 'BULLISH'
            confidence += (cp_ratio - 1) * 15  # +15 for each 0.1 above 1
        elif cp_ratio < 0.67:  # Put/Call > 1.5
            signal = 'SHORT'
            bias = 'BEARISH'
            confidence += ((1/cp_ratio) - 1) * 15
        
        # Max Pain etkisi - MAJOR BOOST
        max_pain_distance = 0
        max_pain_direction = 'NEUTRAL'
        if max_pain > 0 and current_price > 0:
            max_pain_distance = ((max_pain - current_price) / current_price) * 100
            
            # Max Pain çok farklıysa güçlü sinyal
            if max_pain_distance > 5:
                max_pain_direction = 'LONG'
                confidence += 20  # Strong pull up
                if signal == 'NEUTRAL':
                    signal = 'LONG'
                    bias = 'BULLISH'
            elif max_pain_distance > 2:
                max_pain_direction = 'LONG'
                confidence += 10
                if signal == 'NEUTRAL':
                    signal = 'LONG'
                    bias = 'BULLISH'
            elif max_pain_distance < -5:
                max_pain_direction = 'SHORT'
                confidence += 20  # Strong pull down
                if signal == 'NEUTRAL':
                    signal = 'SHORT'
                    bias = 'BEARISH'
            elif max_pain_distance < -2:
                max_pain_direction = 'SHORT'
                confidence += 10
                if signal == 'NEUTRAL':
                    signal = 'SHORT'
                    bias = 'BEARISH'
        
        # IV Rank boost - high IV = confidence boost
        if iv_rank > 60:
            confidence += 10
            iv_status = 'HIGH'
            iv_note = 'Volatilite yüksek'
        elif iv_rank < 20:
            confidence += 5  # Low IV = potential breakout
            iv_status = 'LOW'
            iv_note = 'Volatilite düşük - büyük hareket bekleniyor'
        else:
            iv_status = 'NORMAL'
            iv_note = 'Volatilite normal seviyede'
        
        # Clamp confidence
        confidence = max(45, min(80, confidence))
        
        return {
            'available': True,
            'signal': signal,
            'bias': bias,
            'confidence': confidence,
            'call_put_ratio': cp_ratio,
            'call_oi': options['call_oi'],
            'put_oi': options['put_oi'],
            'max_pain': max_pain,
            'max_pain_distance_pct': max_pain_distance,
            'max_pain_direction': max_pain_direction,
            'iv_rank': iv_rank,
            'iv_status': iv_status,
            'iv_note': iv_note,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_signal_for_orchestrator(self) -> Dict:
        """SignalOrchestrator için sinyal."""
        analysis = self.analyze()
        
        if not analysis.get('available'):
            return {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'reason': 'Options data unavailable'
            }
        
        return {
            'direction': analysis['signal'],
            'confidence': analysis['confidence'],
            'reason': f"C/P Ratio: {analysis['call_put_ratio']:.2f}, Max Pain: ${analysis['max_pain']:,.0f}"
        }


# Convenience functions
def get_options_signal() -> Dict:
    """Hızlı opsiyon sinyali."""
    analyzer = OptionsFlowAnalyzer()
    return analyzer.get_signal_for_orchestrator()


def get_options_analysis() -> Dict:
    """Tam opsiyon analizi."""
    analyzer = OptionsFlowAnalyzer()
    return analyzer.analyze()
