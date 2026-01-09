import anthropic
import logging
from src.config import Config

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

            prompt = f"""Sen profesyonel bir hedge fund yöneticisisin ve kripto işlemlerinde uzmanlaşmışsın. 

{context}

Bu verilere (ve varsa geçmiş performansına) dayanarak BTC/ETH için stratejik tavsiyeni ver:

1. **Genel Piyasa Değerlendirmesi**: Mevcut piyasa ortamı nedir? (Risk-On/Risk-Off/Belirsiz)
2. **Önerilen Pozisyon**: LONG, SHORT mı pozisyon almalıyız yoksa CASH'te mi kalmalıyız?
3. **Risk Seviyesi**: YÜKSEK, ORTA veya DÜŞÜK risk ortamı mı?
4. **Mantık**: Mantığını adım adım açıkla (makro → sentiment → teknik → sonuç)
5. **Giriş Koşulları**: İşleme girmeden önce hangi spesifik koşullar sağlanmalı?
6. **Stop Loss Stratejisi**: Koruyucu stop'lar nereye konulmalı?

ÖNEMLİ: Geçmiş performans belirli güven seviyelerinin başarısız olduğunu gösteriyorsa, stratejini buna göre ayarla.

TÜRKÇE cevap ver. JSON formatında şu anahtarlarla: assessment, position, risk_level, reasoning, entry_conditions, stop_strategy"""

            # Claude 3.5 was deprecated Oct 2025 - use Claude Sonnet 4
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",  # Latest available
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text
            
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
