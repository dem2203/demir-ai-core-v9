import logging
from src.utils.signal_tracker import SignalPerformanceTracker

logger = logging.getLogger("ADAPTIVE_WEIGHTS")

class AdaptiveModuleWeightManager:
    """
    Dynamically adjusts AI module weights based on historical performance.
    Uses signal_tracker data to learn which modules are most accurate.
    """
    
    def __init__(self, signal_tracker: SignalPerformanceTracker):
        self.tracker = signal_tracker
        
        # Default weights (used when no history)
        self.default_weights = {
            "Gemini Vision": 3.0,
            "Macro Brain": 1.0,
            "Technical Analysis": 1.0,
            "Price Action": 1.0,
            "Order Book": 1.0,
            "Funding Rate": 1.0,
            "Volume Profile": 1.0,
            "CVD": 1.0,
            "Claude Strategist": 1.0,
            "GPT-4 News": 1.0
        }
    
    def get_adaptive_weight(self, module_name: str) -> float:
        """
        Get adaptive weight for a module based on historical win rate.
        
        Args:
            module_name: Name of the AI module
        
        Returns:
            weight: Multiplier for this module's vote (0.5 to 4.0)
        """
        # Get module performance stats
        module_perf = self.tracker.get_module_performance()
        
        # If no history or module not in stats, use default
        if not module_perf or module_name not in module_perf:
            default = self.default_weights.get(module_name, 1.0)
            logger.debug(f"{module_name}: Using default weight {default:.1f} (no history)")
            return default
        
        stats = module_perf[module_name]
        win_rate = stats['win_rate']
        total_votes = stats['total_votes']
        
        # Need minimum sample size for reliability
        if total_votes < 10:
            default = self.default_weights.get(module_name, 1.0)
            logger.debug(f"{module_name}: Using default weight {default:.1f} (only {total_votes} samples)")
            return default
        
        # Calculate adaptive weight based on win rate
        # Win rate 50% = neutral (weight 1.0)
        # Win rate 70%+ = excellent (weight 3.0-4.0)
        # Win rate 30%- = poor (weight 0.5)
        
        if win_rate >= 75:
            weight = 4.0  # Exceptional - quadruple weight!
        elif win_rate >= 70:
            weight = 3.5  # Excellent
        elif win_rate >= 65:
            weight = 3.0  # Very good
        elif win_rate >= 60:
            weight = 2.0  # Good
        elif win_rate >= 55:
            weight = 1.5  # Above average
        elif win_rate >= 50:
            weight = 1.0  # Average/neutral
        elif win_rate >= 45:
            weight = 0.8  # Below average
        elif win_rate >= 40:
            weight = 0.6  # Poor
        else:
            weight = 0.5  # Very poor - downweight significantly
        
        logger.info(f"🧠 {module_name}: Win rate {win_rate:.1f}% ({total_votes} trades) → Weight {weight:.1f}x")
        
        return weight
    
    def get_all_weights(self) -> dict:
        """Get adaptive weights for all modules"""
        weights = {}
        for module_name in self.default_weights.keys():
            weights[module_name] = self.get_adaptive_weight(module_name)
        return weights
    
    def get_performance_report(self) -> str:
        """Generate human-readable performance report"""
        module_perf = self.tracker.get_module_performance()
        
        if not module_perf:
            return "📊 No historical data yet for adaptive weighting."
        
        weights = self.get_all_weights()
        
        report = "🧠 **ADAPTIVE MODULE WEIGHTS**\\n"
        report += "━━━━━━━━━━━━━━━━━━\\n"
        
        # Sort by weight (descending)
        sorted_modules = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for module_name, weight in sorted_modules:
            if module_name in module_perf:
                stats = module_perf[module_name]
                emoji = "🔥" if weight >= 3.0 else "✅" if weight >= 1.5 else "⚠️" if weight >= 1.0 else "🔻"
                report += f"{emoji} **{module_name}**: {stats['win_rate']:.1f}% WR | Weight {weight:.1f}x\\n"
            else:
                report += f"🆕 **{module_name}**: No data | Weight {weight:.1f}x (default)\\n"
        
        return report
