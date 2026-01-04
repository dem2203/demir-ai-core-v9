# -*- coding: utf-8 -*-
"""
DEMIR AI v11.1 - Advanced Signal Enhancer Tests
================================================
Whale, Liquidation ve Sentiment modüllerinin entegrasyon testleri.

Test Kapsamı:
1. Modül başlatma (initialization)
2. Score hesaplama (-0.15 to 0.15 arası)
3. Boost mekanizması
4. Block mekanizması
5. Entegrasyon testi

Run: pytest tests/test_advanced_signals.py -v
"""
import pytest
import asyncio
import sys
sys.path.insert(0, '.')

# pytest-asyncio configuration
pytest_plugins = ('pytest_asyncio',)

from src.execution.advanced_signals import AdvancedSignalEnhancer, get_advanced_enhancer


class TestAdvancedSignalEnhancer:
    """AdvancedSignalEnhancer birim testleri."""
    
    @pytest.fixture
    def enhancer(self):
        """Test için enhancer instance'ı."""
        return AdvancedSignalEnhancer()
    
    # ==================== INITIALIZATION ====================
    
    def test_enhancer_creation(self, enhancer):
        """Enhancer başarıyla oluşturulmalı."""
        assert enhancer is not None
        assert enhancer.initialized == False
        assert enhancer.whale_tracker is None
        assert enhancer.liq_hunter is None
        assert enhancer.sentiment_analyzer is None
    
    @pytest.mark.asyncio
    async def test_enhancer_initialization(self, enhancer):
        """Initialize çağrıldığında modüller yüklenmeli."""
        await enhancer.initialize()
        assert enhancer.initialized == True
        # Not: Modüller yüklenemese de initialized=True olacak
    
    def test_singleton_pattern(self):
        """get_advanced_enhancer() singleton döndürmeli."""
        e1 = get_advanced_enhancer()
        e2 = get_advanced_enhancer()
        assert e1 is e2
    
    # ==================== SCORE CALCULATIONS ====================
    
    @pytest.mark.asyncio
    async def test_whale_score_range(self, enhancer):
        """Whale skoru -0.10 ile 0.10 arasında olmalı."""
        score = await enhancer.get_whale_score("BTCUSDT")
        assert -0.10 <= score <= 0.10
    
    @pytest.mark.asyncio
    async def test_liquidation_score_range(self, enhancer):
        """Liquidation skoru -0.10 ile 0.10 arasında olmalı."""
        score = await enhancer.get_liquidation_score("BTCUSDT")
        assert -0.10 <= score <= 0.10
    
    @pytest.mark.asyncio
    async def test_sentiment_score_range(self, enhancer):
        """Sentiment skoru -0.10 ile 0.10 arasında olmalı."""
        score = await enhancer.get_sentiment_score("BTCUSDT")
        assert -0.10 <= score <= 0.10
    
    @pytest.mark.asyncio
    async def test_boost_score_range(self, enhancer):
        """Total boost -0.15 ile 0.15 arasında olmalı."""
        boost = await enhancer.get_boost_score("BTCUSDT")
        assert -0.15 <= boost <= 0.15
    
    # ==================== BLOCKING MECHANISM ====================
    
    @pytest.mark.asyncio
    async def test_block_returns_tuple(self, enhancer):
        """should_block_signal() (bool, str) döndürmeli."""
        result = await enhancer.should_block_signal("BTCUSDT", "BUY")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
    
    # ==================== ENHANCED SIGNAL DATA ====================
    
    @pytest.mark.asyncio
    async def test_enhanced_signal_data_structure(self, enhancer):
        """get_enhanced_signal_data() doğru yapıda dict döndürmeli."""
        data = await enhancer.get_enhanced_signal_data("BTCUSDT", "BUY", 0.85)
        
        required_keys = ['blocked', 'block_reason', 'boost', 'final_confidence',
                         'whale_score', 'liq_score', 'sentiment_score']
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
    
    @pytest.mark.asyncio
    async def test_final_confidence_capped(self, enhancer):
        """Final güven 1.0'ı geçmemeli."""
        data = await enhancer.get_enhanced_signal_data("BTCUSDT", "BUY", 0.95)
        assert data['final_confidence'] <= 1.0
    
    # ==================== CACHING ====================
    
    @pytest.mark.asyncio
    async def test_cache_works(self, enhancer):
        """İkinci çağrıda cache'den dönmeli."""
        # İlk çağrı
        score1 = await enhancer.get_whale_score("BTCUSDT")
        # İkinci çağrı (cache'den gelmeli)
        score2 = await enhancer.get_whale_score("BTCUSDT")
        # Aynı değer olmalı (cache hit)
        assert score1 == score2


class TestSignalGeneratorIntegration:
    """SignalGenerator entegrasyon testleri."""
    
    @pytest.mark.asyncio
    async def test_signal_generator_with_advanced(self):
        """SignalGenerator advanced modla başlatılabilmeli."""
        from src.execution.signal_generator import SignalGenerator
        
        gen = SignalGenerator(["BTCUSDT"], use_advanced=True)
        assert gen.use_advanced == True
        assert gen.enhancer is not None
    
    @pytest.mark.asyncio
    async def test_signal_generator_without_advanced(self):
        """SignalGenerator basic modla başlatılabilmeli."""
        from src.execution.signal_generator import SignalGenerator
        
        gen = SignalGenerator(["BTCUSDT"], use_advanced=False)
        assert gen.use_advanced == False
        assert gen.enhancer is None


# ==================== RUN TESTS ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
