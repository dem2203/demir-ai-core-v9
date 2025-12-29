# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - LLM BRAIN
========================
Claude Haiku ile bağlamsal analiz yapan AI beyin.

Tüm verileri alır, Claude'a sorar, akıllı analiz döner.
"""
import os
import logging
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger("LLM_BRAIN")


@dataclass
class LLMAnalysis:
    """LLM analiz sonucu"""
    direction: str = "HOLD"  # BUY, SELL, HOLD
    confidence: int = 50
    reasoning: str = ""
    entry_price: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    risk_reward: float = 0
    key_factors: list = None
    timestamp: str = ""
    
    def __post_init__(self):
        if self.key_factors is None:
            self.key_factors = []
    
    def to_dict(self) -> Dict:
        return {
            'direction': self.direction,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward': self.risk_reward,
            'key_factors': self.key_factors,
            'timestamp': self.timestamp
        }


class LLMBrain:
    """Claude Haiku ile trading analizi"""
    
    MODEL = "claude-3-5-haiku-20241022"  # Fastest and cheapest
    
    SYSTEM_PROMPT = """Sen profesyonel bir kripto trader'sın. Verilen piyasa verilerini analiz edip net trading kararları veriyorsun.

KURALLAR:
1. Her zaman net bir yön belirt: BUY, SELL veya HOLD
2. Güven seviyesini 0-100 arası ver
3. Kararını destekleyen 3-5 ana faktör belirt
4. Entry, Stop Loss ve Take Profit seviyeleri öner
5. Risk/Reward oranını hesapla
6. Türkçe yanıt ver

ÇIKTI FORMATI (JSON):
{
  "direction": "BUY" | "SELL" | "HOLD",
  "confidence": 0-100,
  "reasoning": "Ana neden (1-2 cümle)",
  "entry_price": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "key_factors": ["faktör1", "faktör2", "faktör3"]
}"""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        self._enabled = False
        
        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                self._enabled = True
                logger.info("🧠 LLM Brain initialized with Claude Haiku")
            except ImportError:
                logger.warning("⚠️ anthropic package not installed. Run: pip install anthropic")
            except Exception as e:
                logger.error(f"❌ LLM Brain init error: {e}")
        else:
            logger.warning("⚠️ ANTHROPIC_API_KEY not set. LLM Brain disabled.")
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None
    
    async def analyze(
        self,
        symbol: str,
        current_price: float,
        technical_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        onchain_data: Dict[str, Any] = None,
        sentiment_data: Dict[str, Any] = None
    ) -> LLMAnalysis:
        """
        Tüm verileri Claude'a gönder ve analiz al.
        
        Args:
            symbol: Trading pair (örn: BTCUSDT)
            current_price: Şu anki fiyat
            technical_data: RSI, LSTM, patterns vb.
            macro_data: BTC.D, Fear Index, Market Cap
            onchain_data: Whale, exchange flow (opsiyonel)
            sentiment_data: News, social sentiment (opsiyonel)
        
        Returns:
            LLMAnalysis: Claude'un analiz sonucu
        """
        
        if not self.is_enabled:
            logger.debug("LLM Brain disabled, returning fallback")
            return self._fallback_analysis(symbol, current_price, technical_data, macro_data)
        
        try:
            # Build context prompt
            prompt = self._build_prompt(
                symbol, current_price, technical_data, 
                macro_data, onchain_data, sentiment_data
            )
            
            # Call Claude
            message = self._client.messages.create(
                model=self.MODEL,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse response
            response_text = message.content[0].text
            analysis = self._parse_response(response_text, current_price)
            
            logger.info(f"🧠 LLM Analysis for {symbol}: {analysis.direction} ({analysis.confidence}%)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return self._fallback_analysis(symbol, current_price, technical_data, macro_data)
    
    def _build_prompt(
        self,
        symbol: str,
        current_price: float,
        technical_data: Dict,
        macro_data: Dict,
        onchain_data: Dict = None,
        sentiment_data: Dict = None
    ) -> str:
        """Analiz promptu oluştur"""
        
        prompt = f"""# {symbol} ANALİZİ

## GÜNCEL FİYAT
${current_price:,.2f}

## TEKNİK GÖSTERGELER
- LSTM Tahmin: {technical_data.get('lstm_direction', 'N/A')} ({technical_data.get('lstm_change', 0):+.2f}%)
- LSTM Güven: {technical_data.get('lstm_confidence', 0):.0f}%
- RSI (1h): {technical_data.get('rsi', 50):.0f}
- Order Book Skor: {technical_data.get('orderbook_score', 0):+.0f}
- Wyckoff Phase: {technical_data.get('wyckoff_phase', 'N/A')}
- Volatility: {technical_data.get('volatility_state', 'NORMAL')}

## MAKRO VERİLER
- Fear & Greed Index: {macro_data.get('fear_greed_index', 50)} ({macro_data.get('fear_greed_label', 'Neutral')})
- BTC Dominance: {macro_data.get('btc_dominance', 0):.1f}%
- BTC.D Değişim (24h): {macro_data.get('btc_dominance_change_24h', 0):+.2f}%
- Toplam Market Cap: ${macro_data.get('total_market_cap', 0)/1e12:.2f}T
- MC Değişim (24h): {macro_data.get('total_market_cap_change_24h', 0):+.2f}%
"""
        
        if onchain_data:
            prompt += f"""
## ON-CHAIN VERİLER
- Whale Net Flow: {onchain_data.get('whale_flow', 0):+.0f}
- Exchange Inflow/Outflow: {onchain_data.get('exchange_flow', 'N/A')}
- Funding Rate: {onchain_data.get('funding_rate', 0):.4f}%
"""
        
        if sentiment_data:
            prompt += f"""
## SENTIMENT
- Haber Sentiment: {sentiment_data.get('news_sentiment', 'NEUTRAL')}
- Social Sentiment: {sentiment_data.get('social_sentiment', 'N/A')}
"""
        
        prompt += """
## GÖREV
Yukarıdaki verileri analiz et ve JSON formatında trading kararı ver.
Sadece JSON döndür, başka bir şey yazma."""
        
        return prompt
    
    def _parse_response(self, response: str, current_price: float) -> LLMAnalysis:
        """Claude yanıtını parse et"""
        
        try:
            # JSON'u bul ve parse et
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Calculate R/R if not provided
                entry = data.get('entry_price', current_price)
                sl = data.get('stop_loss', 0)
                tp = data.get('take_profit', 0)
                
                if sl > 0 and tp > 0 and entry > 0:
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)
                    rr = reward / risk if risk > 0 else 0
                else:
                    rr = 0
                
                return LLMAnalysis(
                    direction=data.get('direction', 'HOLD'),
                    confidence=int(data.get('confidence', 50)),
                    reasoning=data.get('reasoning', ''),
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    risk_reward=rr,
                    key_factors=data.get('key_factors', []),
                    timestamp=datetime.now().isoformat()
                )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
        except Exception as e:
            logger.error(f"Response parse error: {e}")
        
        # Fallback
        return LLMAnalysis(
            direction="HOLD",
            confidence=30,
            reasoning="Parse error - conservative hold",
            timestamp=datetime.now().isoformat()
        )
    
    def _fallback_analysis(
        self,
        symbol: str,
        current_price: float,
        technical_data: Dict,
        macro_data: Dict
    ) -> LLMAnalysis:
        """LLM çalışmadığında fallback analiz"""
        
        # Simple rule-based fallback
        score = 0
        factors = []
        
        # LSTM direction
        if technical_data.get('lstm_direction') == 'UP':
            score += 30
            factors.append("LSTM UP")
        elif technical_data.get('lstm_direction') == 'DOWN':
            score -= 30
            factors.append("LSTM DOWN")
        
        # Fear & Greed
        fear = macro_data.get('fear_greed_index', 50)
        if fear < 25:
            score += 20
            factors.append("Extreme Fear (contrarian buy)")
        elif fear > 75:
            score -= 20
            factors.append("Extreme Greed (contrarian sell)")
        
        # RSI
        rsi = technical_data.get('rsi', 50)
        if rsi < 30:
            score += 15
            factors.append("RSI oversold")
        elif rsi > 70:
            score -= 15
            factors.append("RSI overbought")
        
        # Direction
        if score > 20:
            direction = "BUY"
            sl = current_price * 0.98
            tp = current_price * 1.03
        elif score < -20:
            direction = "SELL"
            sl = current_price * 1.02
            tp = current_price * 0.97
        else:
            direction = "HOLD"
            sl = tp = current_price
        
        return LLMAnalysis(
            direction=direction,
            confidence=min(abs(score) + 30, 90),
            reasoning="Fallback analysis (LLM disabled)",
            entry_price=current_price,
            stop_loss=sl,
            take_profit=tp,
            key_factors=factors,
            timestamp=datetime.now().isoformat()
        )


# Singleton
_llm_brain: Optional[LLMBrain] = None


def get_llm_brain() -> LLMBrain:
    global _llm_brain
    if _llm_brain is None:
        _llm_brain = LLMBrain()
    return _llm_brain
