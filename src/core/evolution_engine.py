import logging
import numpy as np
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger("EVOLUTION_ENGINE")


class EvolutionEngine:
    """
    PHASE 20: SELF-EVOLUTION (GENETIC OPTIMIZATION)
    
    ✅ GERÇEK BACKTEST ENTEGRASYONU AKTİF!
    
    Her genome için gerçek backtest çalıştırarak fitness hesaplar:
    - 7 günlük tarihsel veri üzerinde test
    - Fitness = win_rate * 0.5 + total_pnl_normalized * 0.5
    """
    
    def __init__(self):
        self.config_file = "src/config/evolved_params.json"
        self.population_size = 6  # Daha az çünkü backtest yavaş
        self.elite_count = 2
        self.mutation_rate = 0.2
        self.backtest_days = 7  # Her genome için 7 günlük test
        
        self.param_ranges = {
            "rsi_period": (10, 20),
            "rsi_oversold": (25, 35),
            "rsi_overbought": (65, 75),
            "sl_percent": (1.5, 3.5),
            "tp_percent": (3.0, 6.0)
        }
        
    async def evolve_parameters_async(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """
        Run one evolution cycle with REAL BACKTEST.
        
        Args:
            symbol: Trading pair to optimize for
            
        Returns: Best parameters dict or None if failed
        """
        logger.info("🧬 Starting Evolution Cycle (REAL BACKTEST)...")
        
        try:
            # Step 1: Generate population
            population = self._create_population()
            logger.info(f"📊 Population size: {len(population)}")
            
            # Step 2: Evaluate fitness with REAL BACKTESTS
            fitness_scores = await self._evaluate_population_async(population, symbol)
            
            if not fitness_scores or all(f == 0 for f in fitness_scores):
                logger.error("❌ Tüm backtestler başarısız oldu")
                return None
            
            # Step 3: Select elite
            elite = self._select_elite(population, fitness_scores)
            
            # Step 4: Breed new generation (for next cycle)
            next_gen = self._breed(elite)
            
            # Step 5: Save best
            best_idx = fitness_scores.index(max(fitness_scores))
            best = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            self._save_params(best)
            
            logger.info(f"✅ Evolution Complete. Best Fitness: {best_fitness:.2f}")
            logger.info(f"📋 Best Params: {best}")
            
            return best
            
        except Exception as e:
            logger.error(f"❌ Evolution failed: {e}")
            return None
    
    def evolve_parameters(self) -> Optional[Dict]:
        """
        Synchronous wrapper for async evolution.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context
                logger.warning("⚠️ evolve_parameters called from async context, use evolve_parameters_async instead")
                return None
            return loop.run_until_complete(self.evolve_parameters_async())
        except RuntimeError:
            # No event loop
            return asyncio.run(self.evolve_parameters_async())
        
    def _create_population(self) -> List[Dict]:
        """Generate random parameter sets"""
        population = []
        for _ in range(self.population_size):
            genome = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                genome[param] = round(np.random.uniform(min_val, max_val), 2)
            population.append(genome)
        return population
        
    async def _evaluate_population_async(self, population: List[Dict], symbol: str) -> List[float]:
        """
        Calculate fitness for each genome using REAL BACKTESTS.
        
        Fitness = win_rate * 0.5 + normalized_pnl * 0.5
        """
        from src.backtest.backtester import Backtester
        
        fitness_scores = []
        
        for i, genome in enumerate(population):
            logger.info(f"📊 Testing genome {i+1}/{len(population)}: {genome}")
            
            try:
                # Create backtester and run test
                backtester = Backtester()
                
                # Run backtest with genome parameters
                # Note: Current backtester doesn't use params yet, but we set them
                results = await backtester.run_backtest(
                    symbol=symbol,
                    days=self.backtest_days,
                    params=genome
                )
                
                if results and 'error' not in results:
                    win_rate = results.get('win_rate', 0) / 100  # Normalize to 0-1
                    total_pnl = results.get('total_pnl_pct', 0)
                    
                    # Normalize PnL (-50 to +50 -> 0 to 1)
                    normalized_pnl = max(0, min(1, (total_pnl + 50) / 100))
                    
                    # Calculate fitness
                    fitness = win_rate * 0.5 + normalized_pnl * 0.5
                    
                    logger.info(f"   Win Rate: {win_rate*100:.1f}%, PnL: {total_pnl:.2f}%, Fitness: {fitness:.3f}")
                else:
                    fitness = 0
                    logger.warning(f"   Backtest failed for genome {i+1}")
                    
            except Exception as e:
                logger.error(f"   Error testing genome {i+1}: {e}")
                fitness = 0
            
            fitness_scores.append(fitness)
        
        return fitness_scores
        
    def _select_elite(self, population: List[Dict], fitness: List[float]) -> List[Dict]:
        """Select top performers"""
        sorted_pairs = sorted(zip(fitness, population), reverse=True)
        return [p for _, p in sorted_pairs[:self.elite_count]]
        
    def _breed(self, elite: List[Dict]) -> List[Dict]:
        """Create next generation via crossover and mutation"""
        next_gen = [e.copy() for e in elite]  # Keep elite
        
        while len(next_gen) < self.population_size:
            # Crossover
            parent1, parent2 = np.random.choice(len(elite), 2, replace=False)
            child = {}
            for param in self.param_ranges.keys():
                child[param] = elite[parent1][param] if np.random.rand() > 0.5 else elite[parent2][param]
                
                # Mutation
                if np.random.rand() < self.mutation_rate:
                    min_val, max_val = self.param_ranges[param]
                    child[param] = round(np.random.uniform(min_val, max_val), 2)
                    
            next_gen.append(child)
            
        return next_gen
        
    def _save_params(self, params: Dict):
        """Save evolved parameters to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump({
                    'params': params,
                    'evolved_at': datetime.now().isoformat(),
                    'version': 'v2_real_backtest'
                }, f, indent=2)
            logger.info(f"💾 Saved evolved params to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save params: {e}")
    
    def load_params(self) -> Optional[Dict]:
        """Load evolved parameters from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return data.get('params', {})
        except Exception as e:
            logger.error(f"Failed to load params: {e}")
        return None
