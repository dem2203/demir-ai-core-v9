import logging
import os
import io
import json
import base64
from typing import Dict, Optional, List
import google.generativeai as genai
from openai import OpenAI
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

from src.config.settings import Config

logger = logging.getLogger("VISION_ANALYST")

class VisionAnalyst:
    """
    DUAL VISION CORTEX - Multi-Model AI Vision Analysis
    
    Capabilities:
    1. Generate technical charts from raw OHLCV data
    2. Analyze chart images using DUAL Multimodal AIs:
       - Google Gemini (Fast, wide context)
       - OpenAI GPT-4o (Precise, technical)
    3. Cross-validate patterns and trends between models
    4. Identify visual patterns that mathematical indicators might miss
    """
    
    def __init__(self):
        self.google_key = os.getenv("GOOGLE_API_KEY") or getattr(Config, 'GEMINI_API_KEY', None)
        self.openai_key = os.getenv("OPENAI_API_KEY") or getattr(Config, 'OPENAI_API_KEY', None)
        
        self.gemini_active = False
        self.openai_active = False
        
        # Initialize Gemini
        if self.google_key:
            try:
                genai.configure(api_key=self.google_key)
                # List of models to try in order of preference (Speed > Quality for this use case)
                self.gemini_models_list = [
                    "gemini-1.5-flash", 
                    "gemini-1.5-flash-latest",
                    "gemini-1.5-pro",
                    "gemini-pro-vision"
                ]
                self.gemini_active = True
                logger.info("👁️ Gemini Vision: ACTIVE (Fallback Mode Ready)")
            except Exception as e:
                logger.error(f"Gemini initialization failed: {e}")
        else:
            logger.warning("⚠️ GOOGLE_API_KEY not found. Gemini disabled.")
            
        # Initialize OpenAI
        if self.openai_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_key)
                self.openai_active = True
                logger.info("👁️ OpenAI (GPT-4o) Vision: ACTIVE")
            except Exception as e:
                logger.error(f"OpenAI initialization failed: {e}")
        else:
            logger.warning("⚠️ OPENAI_API_KEY not found. GPT-4o disabled.")
            
        if not self.gemini_active and not self.openai_active:
            logger.warning("⚠️ NO VISION APIs AVAILABLE. Visual Cortex disabled.")
            
    def analyze_chart(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Generate chart image and analyze it using DUAL AI Vision (Gemini + GPT-4o).
        Returns combined consensus analysis.
        """
        if (not self.gemini_active and not self.openai_active) or df.empty:
            return self._empty_analysis("No Vision APIs active or empty DataFrame")
            
        try:
            # 1. Generate Chart Image
            img_bytes = self._generate_chart_image(symbol, df)
            
            if not img_bytes:
                return self._empty_analysis("Chart generation failed")
            
            # 2. Query BOTH AIs (if available)
            analyses = []
            errors = []
            
            if self.gemini_active:
                gemini_result, gemini_error = self._query_gemini(img_bytes)
                if gemini_result:
                    analyses.append(("Gemini", gemini_result))
                elif gemini_error:
                    errors.append(f"Gemini Error: {gemini_error}")
                    
            if self.openai_active:
                openai_result, openai_error = self._query_openai(img_bytes)
                if openai_result:
                    analyses.append(("GPT-4o", openai_result))
                elif openai_error:
                    errors.append(f"GPT-4o Error: {openai_error}")
            
            # 3. Combine Results
            if len(analyses) == 0:
                combined_error = " | ".join(errors) if errors else "Both AI models failed silently"
                return self._empty_analysis(combined_error)
            elif len(analyses) == 1:
                # Single model available
                _, result = analyses[0]
                result['dual_vision'] = False
                result['agreement'] = 'N/A'
                # Append other errors to reasoning if one failed and other succeeded?
                # Maybe useful but let's keep it clean for now.
                return result
            else:
                # Dual Vision: Compare and Combine
                return self._combine_analyses(analyses)
            
        except Exception as e:
            logger.error(f"Visual Analysis failed for {symbol}: {e}")
            return self._empty_analysis(f"Analysis Error: {str(e)}")
            
    def _query_gemini(self, img_bytes: bytes) -> tuple[Optional[Dict], Optional[str]]:
        """Query Google Gemini Vision API with Fallback Logic. Returns (result, error_message)"""
        
        last_error = None
        
        for model_name in self.gemini_models_list:
            try:
                model = genai.GenerativeModel(model_name)
                
                prompt = self._get_analysis_prompt()
                image_part = {"mime_type": "image/png", "data": img_bytes}
                
                response = model.generate_content([prompt, image_part])
                
                # If successful, parse and return
                return self._parse_response(response.text, f"Gemini ({model_name})"), None
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Gemini model '{model_name}' failed: {error_str}")
                last_error = error_str
                
                # If it's a quote error, stop trying? No, usually 404 or 400.
                if "429" in error_str: # Rate limit
                    break 
                    
        # If all failed
        return None, f"All Gemini models failed. Last error: {last_error}"
            
    def _query_openai(self, img_bytes: bytes) -> tuple[Optional[Dict], Optional[str]]:
        """Query OpenAI GPT-4o Vision API. Returns (result, error_message)"""
        try:
            prompt = self._get_analysis_prompt()
            
            # Convert to base64
            import base64
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return self._parse_response(response.choices[0].message.content, "GPT-4o"), None
            
        except Exception as e:
            logger.error(f"OpenAI query failed: {e}")
            return None, str(e)
            
    def _combine_analyses(self, analyses: List[tuple]) -> Dict:
        """
        Combine analyses from multiple models.
        Returns consensus analysis with agreement flag.
        """
        gemini_data = next((data for name, data in analyses if name == "Gemini"), None)
        gpt_data = next((data for name, data in analyses if name == "GPT-4o"), None)
        
        # Average scores
        avg_score = (gemini_data['visual_score'] + gpt_data['visual_score']) // 2
        
        # Check trend agreement
        gemini_trend = gemini_data['trend']
        gpt_trend = gpt_data['trend']
        
        if gemini_trend == gpt_trend:
            consensus_trend = gemini_trend
            agreement = "STRONG"
        elif (gemini_trend == "NEUTRAL" or gpt_trend == "NEUTRAL"):
            consensus_trend = gemini_trend if gemini_trend != "NEUTRAL" else gpt_trend
            agreement = "MODERATE"
        else:
            # Disagreement
            consensus_trend = "NEUTRAL"
            agreement = "CONFLICT"
        
        # Combine reasoning
        combined_reasoning = f"🟢 Gemini: {gemini_data['reasoning']}\n\n🔵 GPT-4o: {gpt_data['reasoning']}"
        
        return {
            "trend": consensus_trend,
            "pattern": gemini_data.get('pattern', 'None'),  # Prefer first model's pattern
            "visual_score": avg_score,
            "reasoning": combined_reasoning,
            "dual_vision": True,
            "agreement": agreement,
            "gemini_score": gemini_data['visual_score'],
            "gpt_score": gpt_data['visual_score'],
            "timestamp": datetime.now().isoformat()
        }
            
    def _generate_chart_image(self, symbol: str, df: pd.DataFrame) -> Optional[bytes]:
        """Convert DataFrame to candlestick chart image bytes using Matplotlib (Server-side safe)"""
        try:
            import matplotlib.pyplot as plt
            # Removed mpf import as it requires 'mplfinance' package which wasn't added.
            # We use standard plt for reliability and speed.
            # However, for simplicity and dependency safety, we will draw a detailed line chart with volume
            # OR we can manually draw candles with matplotlib.patches which is robust.
            
            # Let's use a robust standard Matplotlib approach for OHLC to avoid extra dependencies like mplfinance if possible,
            # BUT mplfinance is standard for this. Let's try standard plt plotting Close price + SMA for simplicity and robustness.
            # AI can understand price action from Line Charts too, but Candles are better.
            # Let's verify if we can stick to simple Line Chart + Volume for reliability.
            
            # Setup Figure
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot Price (Line is safer than complex candle drawing without library)
            ax1.plot(df.index, df['close'], color='#00ff00', linewidth=1.5, label='Close')
            ax1.plot(df.index, df['high'], color='#333333', linewidth=0.5, alpha=0.5) # Shadow high
            ax1.plot(df.index, df['low'], color='#333333', linewidth=0.5, alpha=0.5)  # Shadow low
            
            # Add simple MA if enough data
            if len(df) > 20:
                ax1.plot(df.index, df['close'].rolling(20).mean(), color='yellow', linewidth=0.8, alpha=0.7, label='SMA20')
            
            ax1.set_title(f"{symbol} Price Action (H1)", fontsize=12, color='white')
            ax1.grid(True, alpha=0.2)
            ax1.legend(loc='upper left')
            
            # Plot Volume
            ax2.bar(df.index, df['volume'], color='cyan', alpha=0.5)
            ax2.grid(True, alpha=0.2)
            ax2.set_ylabel("Volume")
            
            # Save to Bytes
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100, facecolor='#0e1117')
            plt.close(fig)
            
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Matplotlib Chart generation failed: {e}")
            return None
            
    def _get_analysis_prompt(self) -> str:
        """Standard prompt for both models"""
        return """
You are an expert technical analyst. I will provide a cryptocurrency price chart.
Analyze the visual patterns and price action.

Focus on:
1. The immediate trend (Bullish/Bearish/Neutral)
2. Any visible chart patterns (Bull Flag, Head & Shoulders, Double Bottom, etc.)
3. Key mental levels (Support/Resistance)

Respond ONLY with a JSON object in this exact format:
{
    "trend": "BULLISH/BEARISH/NEUTRAL",
    "pattern": "Pattern Name or None",
    "visual_score": 0-100 (Where 0 is extremely bearish, 100 is extremely bullish, 50 neutral),
    "reasoning": "Brief explanation of what you see"
}
Do not include markdown formatting or extra text. Just the JSON.
"""
            
    def _parse_response(self, text: str, model_name: str) -> Dict:
        """Parse JSON response from AI"""
        try:
            # Clean up potential markdown code blocks
            clean_text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            
            # Validate fields
            return {
                "trend": data.get("trend", "NEUTRAL").upper(),
                "pattern": data.get("pattern", "None"),
                "visual_score": int(data.get("visual_score", 50)),
                "reasoning": data.get("reasoning", "Analysis unavailable"),
                "timestamp": datetime.now().isoformat(),
                "model": model_name
            }
        except Exception as e:
            logger.error(f"Failed to parse {model_name} response: {text[:100]}... Error: {e}")
            return {
                "trend": "NEUTRAL",
                "pattern": "None",
                "visual_score": 50,
                "reasoning": f"{model_name} parsing failed",
                "timestamp": datetime.now().isoformat(),
                "model": model_name
            }
            
    def _empty_analysis(self, error_msg: str = "Visual analysis disabled or failed") -> Dict:
        return {
            "trend": "NEUTRAL",
            "pattern": "None",
            "visual_score": 50,
            "reasoning": error_msg,
            "dual_vision": False,
            "agreement": "N/A",
            "timestamp": datetime.now().isoformat()
        }
