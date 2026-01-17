"""
Self-Learning RL Agent (Q-Learning)

Goal: Optimize AI Voting Weights automatically based on performance.
Philosophy: "If Technical Analysis lost money yesterday in Choppy market, reduce its weight today."

State Space (4 regimes):
0: LOW_VOL_RANGE (Chop)
1: HIGH_VOL_RANGE (Volatile)
2: BULL_TREND
3: BEAR_TREND

Action Space:
- Increase/Decrease weight of specific AI agents (Gemini, Claude, Technical, etc.)

Algorithm: Q-Learning (Offline, updates daily)
"""

import logging
import json
import os
import random
import numpy as np
from typing import Dict, List

logger = logging.getLogger("RL_AGENT")

class RLAgent:
    def __init__(self, data_dir="data/rl_brain"):
        self.data_dir = data_dir
        self.q_table_path = os.path.join(data_dir, "q_table.json")
        self.os_createdir()
        
        # Hyperparameters
        self.alpha = 0.1  # Learning Rate
        self.gamma = 0.9  # Discount Factor
        self.epsilon = 0.1 # Exploration Rate
        
        # State Definitions
        self.regimes = ["LOW_VOL_RANGE", "HIGH_VOL_RANGE", "BULL_TREND", "BEAR_TREND"]
        
        # Actions: (Agent, Adjustment)
        self.agents = ["Technical", "Gemini", "Claude", "Grok", "Macro"]
        self.actions = []
        for agent in self.agents:
            self.actions.append((agent, +0.2)) # Increase weight
            self.actions.append((agent, -0.2)) # Decrease weight
            
        # Initialize Q-Table
        self.q_table = self._load_q_table()
        
    def os_createdir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_state(self, volatility: float, trend_strength: float) -> str:
        """
        Map market metrics to discrete state
        """
        is_volatile = volatility > 0.02 # 2% daily moves
        is_trending = trend_strength > 25 # ADX > 25
        
        if is_trending:
            # Simplified: Assume trend direction handled by strategy, here we just care if it's trending
            return "BULL_TREND" if True else "BEAR_TREND" # Placeholder logic
            # Better specific logic:
            # We just return TREND or RANGE for simplicity in this version
            return "BULL_TREND" if is_trending else "LOW_VOL_RANGE"
            
        if is_volatile:
            return "HIGH_VOL_RANGE"
        else:
            return "LOW_VOL_RANGE"

    def choose_action(self, state: str) -> int:
        """Epsilon-Greedy Action Selection"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        
        # Exploit: Best action for current state
        state_actions = self.q_table.get(state, [0.0] * len(self.actions))
        return int(np.argmax(state_actions))

    def update(self, state: str, action_idx: int, reward: float, next_state: str):
        """Q-Learning Update Rule"""
        old_val = self.q_table.get(state, [0.0] * len(self.actions))[action_idx]
        next_max = max(self.q_table.get(next_state, [0.0] * len(self.actions)))
        
        # Bellman Equation
        new_val = old_val + self.alpha * (reward + self.gamma * next_max - old_val)
        
        # Save
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)
        
        self.q_table[state][action_idx] = new_val
        self._save_q_table()
        
    def get_optimized_weights(self, state: str) -> Dict[str, float]:
        """Convert learned Q-values to weight multipliers"""
        # Default weights
        weights = {agent: 1.0 for agent in self.agents}
        
        # Apply learned policies (Softmax or best action)
        # For simplicity: Use best action to verify logic
        best_action_idx = int(np.argmax(self.q_table.get(state, [0.0] * len(self.actions))))
        agent, adjustment = self.actions[best_action_idx]
        
        weights[agent] += adjustment
        logger.info(f"🧠 RL Optimization ({state}): Adjusted {agent} weight by {adjustment:+.1f}")
        
        return weights

    def _load_q_table(self):
        try:
            with open(self.q_table_path, 'r') as f:
                return json.load(f)
        except:
            return {r: [0.0] * len(self.actions) for r in self.regimes}

    def _save_q_table(self):
        with open(self.q_table_path, 'w') as f:
            json.dump(self.q_table, f)
