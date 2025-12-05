import logging
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger("EVOLUTION_ENGINE")

class EvolutionEngine:
    """
    PHASE 20: SELF-EVOLUTION (GENETIC OPTIMIZATION)
    
    Uses genetic algorithms to auto-tune trading parameters:
    - RSI period
    - MACD parameters
    - Stop Loss %
    - Take Profit %
    
    Process:
    1. Create 10 parameter variants (genes)
    2. Backtest each on last 7 days
    3. "Survive" top 3 performers
    4. Mutate and breed for next generation
    """
    
    def __init__(self):
        self.config_file = "src/config/evolved_params.json"
        self.population_size = 10
        self.elite_count = 3
        self.mutation_rate = 0.2
        
        self.param_ranges = {
            "rsi_period": (10, 20),
            "rsi_oversold": (25, 35),
            "rsi_overbought": (65, 75),
            "sl_percent": (1.5, 3.5),
            "tp_percent": (3.0, 6.0)
        }
        
    def evolve_parameters(self) -> Dict:
        """
        Run one evolution cycle.
        Returns best parameters.
        """
        logger.info("🧬 Starting Evolution Cycle...")
        
        # Step 1: Generate population
        population = self._create_population()
        
        # Step 2: Evaluate fitness (would run backtests here)
        fitness_scores = self._evaluate_population(population)
        
        # Step 3: Select elite
        elite = self._select_elite(population, fitness_scores)
        
        # Step 4: Breed new generation
        next_gen = self._breed(elite)
        
        # Step 5: Save best
        best = elite[0]
        self._save_params(best)
        
        logger.info(f"✅ Evolution Complete. Best Fitness: {fitness_scores[0]:.2f}")
        return best
        
    def _create_population(self) -> List[Dict]:
        """Generate random parameter sets"""
        population = []
        for _ in range(self.population_size):
            genome = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                genome[param] = np.random.uniform(min_val, max_val)
            population.append(genome)
        return population
        
    def _evaluate_population(self, population: List[Dict]) -> List[float]:
        """
        Calculate fitness for each genome.
        In real implementation, this would run backtests.
        """
        # Placeholder: Random fitness
        return [np.random.uniform(0.5, 1.5) for _ in population]
        
    def _select_elite(self, population: List[Dict], fitness: List[float]) -> List[Dict]:
        """Select top performers"""
        sorted_pop = [x for _, x in sorted(zip(fitness, population), reverse=True)]
        return sorted_pop[:self.elite_count]
        
    def _breed(self, elite: List[Dict]) -> List[Dict]:
        """Create next generation via crossover and mutation"""
        next_gen = elite.copy()  # Keep elite
        
        while len(next_gen) < self.population_size:
            # Crossover
            parent1, parent2 = np.random.choice(elite, 2, replace=False)
            child = {}
            for param in self.param_ranges.keys():
                child[param] = parent1[param] if np.random.rand() > 0.5 else parent2[param]
                
                # Mutation
                if np.random.rand() < self.mutation_rate:
                    min_val, max_val = self.param_ranges[param]
                    child[param] = np.random.uniform(min_val, max_val)
                    
            next_gen.append(child)
            
        return next_gen
        
    def _save_params(self, params: Dict):
        """Save evolved parameters to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(params, f, indent=2)
            logger.info(f"💾 Saved evolved params to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save params: {e}")
