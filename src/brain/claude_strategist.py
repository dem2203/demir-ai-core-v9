import anthropic
import logging
from src.config import Config
from src.utils.retry import async_retry  # FIX 2.1

logger = logging.getLogger("CLAUDE_STRATEGIST")

class ClaudeStrategist:
    """
    Uses Claude for deep macro reasoning and strategy formulation.
    """
    def __init__(self):
        if Config.ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        else:
            self.client = None
            logger.warning("⚠️ Claude API Key missing. Strategic reasoning disabled.")
    
    @async_retry(max_attempts=3, base_delay=2)  # FIX 2.1: Add retry logic
    async def _call_claude_api(self, prompt: str) -> str:
        """Claude API call with retry logic"""
        # Claude 3.5 was deprecated Oct 2025 - use Claude Sonnet 4
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",  # Latest available
            max_tokens=1500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    
    async def formulate_strategy(self, macro_data: dict, chart_analysis: dict, news_sentiment: dict, performance_feedback: str = "") -> dict:
        """
        Ask Claude to act as a hedge fund manager and formulate trading strategy.
        Now includes historical performance feedback for self-learning.
        """
        if not self.client:
            return {"directive": "NEUTRAL", "reasoning": "Claude unavailable"}
            
        try:
            # Build context
            context = f"""**Market Macro Data:**
- VIX (Fear Index): {macro_data.get('vix', 'N/A')}
- DXY (Dollar Index): {macro_data.get('dxy', 'N/A')}
- BTC Dominance: {macro_data.get('btc_dominance', 'N/A')}%
- Market Regime: {macro_data.get('regime', 'UNKNOWN')}

**Chart Technical Analysis (Gemini Vision):**
{chart_analysis.get('analysis', 'No analysis available')}

**News Sentiment:**
{news_sentiment.get('summary', 'No news data')}
Sentiment Score: {news_sentiment.get('sentiment', 'NEUTRAL')}

{performance_feedback if performance_feedback else ""}"""

            prompt = f"""Sen profesyonel bir kripto trader'sın. 10 yıldır futures piyasasında trade ediyorsun. 

{context}

**ŞU ANKİ FİYAT DATA (PROFESYONEL):**
- Order Book Imbalance: {chart_analysis.get('orderbook_summary', 'N/A')}
- Funding Rate: {chart_analysis.get('funding_summary', 'N/A')}  
- CVD (Volume Delta): {chart_analysis.get('cvd_summary', 'N/A')}
- Volume Profile: {chart_analysis.get('volume_profile_summary', 'N/A')}

SENİN GÖREVİN: Bu verilere dayanarak SPESIFIK bir trade analizi yap.

KURALLARIN:
1. "Eğer" veya "ise" kullanma - KEŞİN analiz yap
2. Spesifik fiyat seviyeleri ver (entry, stop, target)
3. Risk/Reward hesapla
4. Conviction ver (1-10)
5. ACTIONABLE ol - trader bu analizi okuyup hemen işlem açabilmeli

ÖNEMLİ MANTIK KURALLARI:
- RSI > 75 (Overbought) ise: LONG pozisyon önerme, sadece breakout teyidi varsa (volume spike vs). Genelde CASH veya SHORT (Reversal) düşün.
- RSI < 25 (Oversold) ise: SHORT pozisyon önerme. Genelde CASH veya LONG (Bounce) düşün.
- MACRO vs TEKNİK çelişkisi: Eğer DXY çok güçlü (>105) ama Teknik Bullish ise, Conviction'ı düşür (<5/10) veya CASH kal.
- Asla şunu yapma: "Teknik çok güçlü, RSI overbought... SHORT". Trend tersine dönmediyse trendin karşısına çıkma.

FORMAT (JSON):
{{
  "market_view": "Kısa açıklama - ne görüyorsun (1 cümle)",
  "position": "LONG" veya "SHORT" veya "NEUTRAL",
  "conviction": 1-10 arası sayı,
  "entry_price": "Şu anki fiyat veya spesifik seviye ($90,450 gibi)",
  "stop_loss": "Spesifik stop seviyesi ($91,100 gibi)",
  "target_1": "İlk hedef ($89,800 gibi)",
  "target_2": "İkinci hedef ($89,100 gibi)",
  "risk_reward": "1:2.5 gibi",
  "reasoning": "NEDEN bu trade? Spesifik data points (Funding 12% APR, Order book $8M sell wall gibi)",
  "risk_level": "HIGH", "MEDIUM" veya "LOW"
}}

ÖRNEK CEVAP:
{{
  "market_view": "BTC overleveraged longs, squeeze bekliyorum",
  "position": "SHORT",
  "conviction": 8,
  "entry_price": "$90,450",
  "stop_loss": "$91,100", 
  "target_1": "$89,800",
  "target_2": "$89,100",
  "risk_reward": "1:2.5",
  "reasoning": "Funding rate 12% APR (aşırı yüksek, long squeeze riski). Order book'ta $90,200'de $8M sell wall. CVD son 4 saatte -$45M (güçlü satış). MACD bearish crossover. DXY güçlü = kripto baskı altında. Çoklu confluence → yüksek probability short.",
  "risk_level": "MEDIUM"
}}

TÜRKÇE cevap ver. Sadece JSON, ekstra açıklama yok."""

            # Claude 3.5 was deprecated Oct 2025 - use Claude Sonnet 4
            # FIX 2.1: Use retry-wrapped API call
            response_text = await self._call_claude_api(prompt)
            
            # Parse JSON
            try:
                import json
                if "{" in response_text and "}" in response_text:
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    result = json.loads(response_text[start:end])
                else:
                    result = {"reasoning": response_text, "position": "WAIT"}
            except:
                result = {"reasoning": response_text, "position": "WAIT"}
                
            logger.info(f"🧠 Claude Strategy: {result.get('position', 'WAIT')} - {result.get('assessment', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {"directive": "ERROR", "reasoning": str(e)}
