# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - AI COUNCIL (Yapay Zeka Konseyi)
==============================================
4 Güçlü AI Modeli Birlikte Karar Veriyor:
- Claude Haiku (Anthropic)
- GPT-4o-mini (OpenAI)
- Gemini Flash (Google)
- DeepSeek V3 (DeepSeek)

Her AI aynı market verisini analiz eder, sonra oylama yapılır.
Çoğunluk kazanır. Eşitlik durumunda Claude'un kararı geçerli.
"""
import os
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger("AI_COUNCIL")


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class AIAnalysis:
    """Tek bir AI'ın analiz sonucu"""
    model_name: str
    direction: str  # BUY, SELL, HOLD
    confidence: int  # 0-100
    reasoning: str
    key_factors: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    response_time_ms: int = 0
    error: Optional[str] = None


@dataclass
class CouncilDecision:
    """AI Council'ın nihai kararı"""
    final_direction: str  # BUY, SELL, HOLD
    final_confidence: int  # Ortalama güven
    vote_count: Dict[str, int]  # {'BUY': 3, 'SELL': 1, 'HOLD': 0}
    unanimous: bool  # Oybirliği var mı?
    individual_analyses: List[AIAnalysis]
    combined_reasoning: str
    veto_active: bool = False
    veto_reason: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_telegram_format(self) -> str:
        """Telegram mesajı formatı"""
        votes = " | ".join([f"{k}: {v}" for k, v in self.vote_count.items() if v > 0])
        
        models_summary = []
        for analysis in self.individual_analyses:
            emoji = "🟢" if analysis.direction == "BUY" else "🔴" if analysis.direction == "SELL" else "⚪"
            models_summary.append(f"{emoji} {analysis.model_name}: {analysis.direction} ({analysis.confidence}%)")
        
        msg = f"""🤖 *AI COUNCIL KARARI*
━━━━━━━━━━━━━━━━━━━━

📊 *OYLAMA:* {votes}
{'✅ OYBİRLİĞİ!' if self.unanimous else ''}

*Model Kararları:*
{chr(10).join(models_summary)}

📍 *FİNAL:* {self.final_direction} ({self.final_confidence}%)

💭 *ÖZET:*
{self.combined_reasoning[:300]}...
"""
        return msg


# ============================================================
# BASE ANALYZER CLASS
# ============================================================

class AIAnalyzer(ABC):
    """AI Analizör Base Class"""
    
    MODEL_NAME = "BaseAI"
    
    def __init__(self):
        self.enabled = False
        self.api_key = None
        self.call_count = 0
        self.total_cost = 0.0
    
    @abstractmethod
    async def analyze(
        self,
        symbol: str,
        current_price: float,
        market_data: Dict
    ) -> AIAnalysis:
        """Market verisini analiz et"""
        pass
    
    def _build_prompt(self, symbol: str, current_price: float, market_data: Dict) -> str:
        """Tüm AI'lar için ortak prompt"""
        return f"""Sen profesyonel bir kripto trader'sın. Aşağıdaki verileri analiz et ve trading kararı ver.

## MARKET VERİLERİ

**Sembol:** {symbol}
**Güncel Fiyat:** ${current_price:,.2f}

### Teknik Göstergeler
- RSI (1h): {market_data.get('rsi', 'N/A')}
- LSTM Tahmin: {market_data.get('lstm_direction', 'N/A')} ({market_data.get('lstm_change', 0):.2f}%)
- Trend: {market_data.get('trend', 'N/A')}
- Volatilite: {market_data.get('volatility_state', 'NORMAL')}

### Order Book
- İmbalance: {market_data.get('orderbook_score', 0):+.0f}%
- Whale Aktivite: {market_data.get('whale_score', 0):+.0f}

### Makro Veriler
- Fear & Greed: {market_data.get('fear_greed', 50)}
- BTC Dominance: {market_data.get('btc_dominance', 0):.1f}%
- Market Regime: {market_data.get('regime', 'UNKNOWN')}
- Haber Sentimenti: {market_data.get('news_sentiment', 'NEUTRAL')}
- Opsiyon Sentimenti: {market_data.get('options_sentiment', 'NEUTRAL')}

### Likidasyon Verileri
- Funding Rate: {market_data.get('funding_rate', 0):.4f}%
- Long/Short Ratio: {market_data.get('ls_ratio', 1.0):.2f}
- OI Değişim: {market_data.get('oi_change', 0):.1f}%

### Pivot Noktaları
- Daily R1: ${market_data.get('daily_r1', 0):,.0f}
- Daily S1: ${market_data.get('daily_s1', 0):,.0f}

## GÖREV

Yukarıdaki verileri analiz et ve şu formatta JSON döndür:

```json
{{
    "direction": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "reasoning": "Kısa açıklama (max 200 karakter)",
    "key_factors": ["Faktör 1", "Faktör 2", "Faktör 3"],
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "price_target": hedef_fiyat_veya_null,
    "stop_loss": stop_fiyat_veya_null
}}
```

SADECE JSON döndür, başka bir şey yazma."""
    
    def _parse_response(self, response_text: str, model_name: str) -> AIAnalysis:
        """AI yanıtını parse et"""
        try:
            # JSON bloğunu bul
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            data = json.loads(json_str)
            
            return AIAnalysis(
                model_name=model_name,
                direction=data.get("direction", "HOLD").upper(),
                confidence=min(100, max(0, int(data.get("confidence", 50)))),
                reasoning=data.get("reasoning", "")[:500],
                key_factors=data.get("key_factors", [])[:5],
                risk_level=data.get("risk_level", "MEDIUM").upper(),
                price_target=data.get("price_target"),
                stop_loss=data.get("stop_loss")
            )
        except Exception as e:
            logger.error(f"{model_name} parse error: {e}")
            return AIAnalysis(
                model_name=model_name,
                direction="HOLD",
                confidence=30,
                reasoning=f"Parse error: {str(e)[:100]}",
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# CLAUDE ANALYZER
# ============================================================

class ClaudeAnalyzer(AIAnalyzer):
    """Claude Haiku Analizör"""
    
    MODEL_NAME = "Claude"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.enabled = bool(self.api_key)
        self.model = "claude-3-haiku-20240307"
        
        if self.enabled:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"✅ {self.MODEL_NAME} initialized")
            except Exception as e:
                logger.error(f"❌ Claude init error: {e}")
                self.enabled = False
    
    async def analyze(self, symbol: str, current_price: float, market_data: Dict) -> AIAnalysis:
        if not self.enabled:
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning="Claude API not available",
                key_factors=[],
                risk_level="HIGH",
                error="API disabled"
            )
        
        import time
        start = time.time()
        
        try:
            prompt = self._build_prompt(symbol, current_price, market_data)
            
            # Sync call in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            
            response_text = response.content[0].text
            self.call_count += 1
            
            analysis = self._parse_response(response_text, self.MODEL_NAME)
            analysis.response_time_ms = int((time.time() - start) * 1000)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning=str(e)[:200],
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# GPT-4 ANALYZER
# ============================================================

class GPTAnalyzer(AIAnalyzer):
    """GPT-4o-mini Analizör"""
    
    MODEL_NAME = "GPT-4"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = bool(self.api_key)
        self.model = "gpt-4o-mini"
        
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"✅ {self.MODEL_NAME} initialized")
            except Exception as e:
                logger.error(f"❌ GPT init error: {e}")
                self.enabled = False
    
    async def analyze(self, symbol: str, current_price: float, market_data: Dict) -> AIAnalysis:
        if not self.enabled:
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning="OpenAI API not available",
                key_factors=[],
                risk_level="HIGH",
                error="API disabled"
            )
        
        import time
        start = time.time()
        
        try:
            prompt = self._build_prompt(symbol, current_price, market_data)
            
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
            )
            
            response_text = response.choices[0].message.content
            self.call_count += 1
            
            analysis = self._parse_response(response_text, self.MODEL_NAME)
            analysis.response_time_ms = int((time.time() - start) * 1000)
            
            return analysis
            
        except Exception as e:
            logger.error(f"GPT analysis error: {e}")
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning=str(e)[:200],
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# GEMINI ANALYZER
# ============================================================

class GeminiAnalyzer(AIAnalyzer):
    """Google Gemini Flash Analizör"""
    
    MODEL_NAME = "Gemini"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.enabled = bool(self.api_key)
        self.model = "gemini-1.5-flash"
        
        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"✅ {self.MODEL_NAME} initialized")
            except Exception as e:
                logger.error(f"❌ Gemini init error: {e}")
                self.enabled = False
    
    async def analyze(self, symbol: str, current_price: float, market_data: Dict) -> AIAnalysis:
        if not self.enabled:
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning="Google API not available",
                key_factors=[],
                risk_level="HIGH",
                error="API disabled"
            )
        
        import time
        start = time.time()
        
        try:
            prompt = self._build_prompt(symbol, current_price, market_data)
            
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(prompt)
            )
            
            response_text = response.text
            self.call_count += 1
            
            analysis = self._parse_response(response_text, self.MODEL_NAME)
            analysis.response_time_ms = int((time.time() - start) * 1000)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning=str(e)[:200],
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# DEEPSEEK ANALYZER
# ============================================================

class DeepSeekAnalyzer(AIAnalyzer):
    """DeepSeek V3 Analizör"""
    
    MODEL_NAME = "DeepSeek"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.enabled = bool(self.api_key)
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com"
        
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                logger.info(f"✅ {self.MODEL_NAME} initialized")
            except Exception as e:
                logger.error(f"❌ DeepSeek init error: {e}")
                self.enabled = False
    
    async def analyze(self, symbol: str, current_price: float, market_data: Dict) -> AIAnalysis:
        if not self.enabled:
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning="DeepSeek API not available",
                key_factors=[],
                risk_level="HIGH",
                error="API disabled"
            )
        
        import time
        start = time.time()
        
        try:
            prompt = self._build_prompt(symbol, current_price, market_data)
            
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
            )
            
            response_text = response.choices[0].message.content
            self.call_count += 1
            
            analysis = self._parse_response(response_text, self.MODEL_NAME)
            analysis.response_time_ms = int((time.time() - start) * 1000)
            
            return analysis
            
        except Exception as e:
            logger.error(f"DeepSeek analysis error: {e}")
            return AIAnalysis(
                model_name=self.MODEL_NAME,
                direction="HOLD",
                confidence=0,
                reasoning=str(e)[:200],
                key_factors=[],
                risk_level="HIGH",
                error=str(e)
            )


# ============================================================
# AI COUNCIL - MAIN ORCHESTRATOR
# ============================================================

class AICouncil:
    """
    AI Council - 4 AI Modeli Birlikte Karar Veriyor
    
    Oylama Kuralları:
    1. Çoğunluk kazanır (3/4 veya 2/4)
    2. Eşitlik durumunda Claude'un kararı geçerli
    3. Tüm AI'lar HOLD derse → Sinyal yok
    4. Oybirliği varsa güven %10 artar
    """
    
    def __init__(self):
        self.analyzers: List[AIAnalyzer] = []
        self.enabled = False
        self._initialize_analyzers()
        
        # Budget tracking
        self.daily_budget = float(os.getenv("AI_DAILY_BUDGET", "5.0"))
        self.daily_cost = 0.0
        self.last_reset = datetime.now().date()
        
        logger.info(f"🤖 AI Council initialized with {len([a for a in self.analyzers if a.enabled])} active models")
    
    def _initialize_analyzers(self):
        """Tüm AI analizörleri başlat"""
        self.analyzers = [
            ClaudeAnalyzer(),
            GPTAnalyzer(),
            GeminiAnalyzer(),
            DeepSeekAnalyzer()
        ]
        
        active = [a.MODEL_NAME for a in self.analyzers if a.enabled]
        self.enabled = len(active) >= 1
        
        if active:
            logger.info(f"🟢 Active AI Models: {', '.join(active)}")
        else:
            logger.warning("⚠️ No AI models available!")
    
    async def analyze(
        self,
        symbol: str,
        current_price: float,
        market_data: Dict
    ) -> CouncilDecision:
        """
        Tüm AI'lardan analiz al ve oylama yap
        """
        if not self.enabled:
            return self._empty_decision("No AI models available")
        
        # Budget check
        self._check_budget()
        
        # Paralel olarak tüm AI'ları çağır
        active_analyzers = [a for a in self.analyzers if a.enabled]
        
        logger.info(f"🗳️ AI Council voting for {symbol}...")
        
        tasks = [
            analyzer.analyze(symbol, current_price, market_data)
            for analyzer in active_analyzers
        ]
        
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Hataları filtrele
        valid_analyses = []
        for i, result in enumerate(analyses):
            if isinstance(result, Exception):
                logger.error(f"AI {active_analyzers[i].MODEL_NAME} failed: {result}")
                valid_analyses.append(AIAnalysis(
                    model_name=active_analyzers[i].MODEL_NAME,
                    direction="HOLD",
                    confidence=0,
                    reasoning=str(result)[:100],
                    key_factors=[],
                    risk_level="HIGH",
                    error=str(result)
                ))
            else:
                valid_analyses.append(result)
        
        # Oylama
        return self._vote(valid_analyses)
    
    def _vote(self, analyses: List[AIAnalysis]) -> CouncilDecision:
        """Oylama yap"""
        vote_count = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for analysis in analyses:
            if analysis.direction in vote_count:
                vote_count[analysis.direction] += 1
        
        # Kazananı bul
        max_votes = max(vote_count.values())
        winners = [d for d, v in vote_count.items() if v == max_votes]
        
        if len(winners) == 1:
            final_direction = winners[0]
        else:
            # Eşitlik - Claude'un kararı geçerli
            claude_analysis = next((a for a in analyses if a.model_name == "Claude"), None)
            if claude_analysis:
                final_direction = claude_analysis.direction
            else:
                final_direction = "HOLD"
        
        # Ortalama güven
        direction_analyses = [a for a in analyses if a.direction == final_direction]
        if direction_analyses:
            avg_confidence = sum(a.confidence for a in direction_analyses) // len(direction_analyses)
        else:
            avg_confidence = 50
        
        # Oybirliği bonusu
        unanimous = max_votes == len(analyses) and len(analyses) > 1
        if unanimous:
            avg_confidence = min(100, avg_confidence + 10)
        
        # VETO kontrolü - Çoğunluk HOLD derse
        veto_active = False
        veto_reason = ""
        if vote_count["HOLD"] >= len(analyses) // 2 + 1:
            veto_active = True
            veto_reason = f"AI Council: {vote_count['HOLD']}/{len(analyses)} HOLD oyu"
        
        # Combined reasoning
        key_reasons = []
        for a in analyses:
            if a.reasoning and a.direction == final_direction:
                key_reasons.append(f"[{a.model_name}] {a.reasoning[:100]}")
        combined = " | ".join(key_reasons[:3])
        
        decision = CouncilDecision(
            final_direction=final_direction,
            final_confidence=avg_confidence,
            vote_count=vote_count,
            unanimous=unanimous,
            individual_analyses=analyses,
            combined_reasoning=combined,
            veto_active=veto_active,
            veto_reason=veto_reason
        )
        
        logger.info(
            f"🗳️ Council Decision: {final_direction} ({avg_confidence}%) | "
            f"Votes: BUY={vote_count['BUY']}, SELL={vote_count['SELL']}, HOLD={vote_count['HOLD']} | "
            f"{'UNANIMOUS!' if unanimous else ''}"
        )
        
        return decision
    
    def _empty_decision(self, reason: str) -> CouncilDecision:
        """Boş karar döndür"""
        return CouncilDecision(
            final_direction="HOLD",
            final_confidence=0,
            vote_count={"BUY": 0, "SELL": 0, "HOLD": 0},
            unanimous=False,
            individual_analyses=[],
            combined_reasoning=reason,
            veto_active=True,
            veto_reason=reason
        )
    
    def _check_budget(self):
        """Günlük bütçe kontrolü"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_cost = 0.0
            self.last_reset = today
        
        if self.daily_cost >= self.daily_budget:
            logger.warning(f"⚠️ Daily AI budget exceeded: ${self.daily_cost:.2f} >= ${self.daily_budget:.2f}")
    
    def get_stats(self) -> Dict:
        """İstatistikler"""
        return {
            "active_models": [a.MODEL_NAME for a in self.analyzers if a.enabled],
            "total_calls": sum(a.call_count for a in self.analyzers),
            "daily_cost": self.daily_cost,
            "daily_budget": self.daily_budget
        }


# ============================================================
# SINGLETON
# ============================================================

_ai_council: Optional[AICouncil] = None


def get_ai_council() -> AICouncil:
    """Get or create singleton AICouncil"""
    global _ai_council
    if _ai_council is None:
        _ai_council = AICouncil()
    return _ai_council


async def council_analyze(symbol: str, current_price: float, market_data: Dict) -> CouncilDecision:
    """Quick access function"""
    council = get_ai_council()
    return await council.analyze(symbol, current_price, market_data)
