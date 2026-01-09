import google.generativeai as genai
import logging
import base64
from src.config import Config

logger = logging.getLogger("GEMINI_VISION")

class GeminiVisionAnalyst:
    """
    Uses Gemini Vision API to analyze chart images.
    """
    def __init__(self):
        if Config.GOOGLE_API_KEY:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            # Use full path for v1beta API compatibility
            self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        else:
            self.model = None
            logger.warning("⚠️ Gemini API Key missing. Vision analysis disabled.")
    
    async def analyze_chart(self, chart_path: str, symbol: str) -> dict:
        """
        Sends chart image to Gemini Vision for analysis.
        """
        if not self.model:
            return {"analysis": "Gemini API unavailable", "trend": "UNKNOWN"}
            
        try:
            # Read image
            with open(chart_path, 'rb') as f:
                image_data = f.read()
            
            # Prepare prompt
            prompt = f"""You are a professional crypto trader analyzing {symbol} chart.

Analyze this 1-hour chart and provide:
1. **Trend**: Clearly state BULLISH, BEARISH, or NEUTRAL
2. **Key Levels**: Identify important support/resistance levels
3. **Volume Analysis**: Is volume confirming the price action?
4. **Pattern Recognition**: Any chart patterns (breakout, consolidation, reversal)?
5. **Trade Recommendation**: Should we BUY, SELL, or WAIT?

Be concise but precise. Format your response as JSON with keys: trend, key_levels, volume_signal, pattern, recommendation, reasoning."""

            # Call Gemini Vision
            import PIL.Image
            import io
            img = PIL.Image.open(io.BytesIO(image_data))
            
            response = self.model.generate_content([prompt, img])
            analysis_text = response.text
            
            # Parse response (try to extract JSON, fallback to text)
            try:
                import json
                # Try to find JSON block in response
                if "{" in analysis_text and "}" in analysis_text:
                    start = analysis_text.find("{")
                    end = analysis_text.rfind("}") + 1
                    json_str = analysis_text[start:end]
                    result = json.loads(json_str)
                else:
                    # Fallback: parse manually
                    result = {
                        "trend": "BULLISH" if "bullish" in analysis_text.lower() else 
                                 "BEARISH" if "bearish" in analysis_text.lower() else "NEUTRAL",
                        "analysis": analysis_text,
                        "recommendation": "WAIT"
                    }
            except:
                result = {"analysis": analysis_text, "trend": "UNKNOWN"}
                
            logger.info(f"🤖 Gemini Vision: {symbol} → {result.get('trend', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Gemini Vision error: {e}")
            return {"analysis": f"Error: {str(e)}", "trend": "ERROR"}
