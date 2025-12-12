import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger("RL_TRADING_ENV")

class TradingEnv(gym.Env):
    """
    Gym-Compatible Trading Environment for RL Agent
    (RL Ajanı için Gym Uyumlu Trading Ortamı)
    
    State (Durum): 28 features (fiyat + makro + pozisyon bilgisi)
    Action (Aksiyon): 0=HOLD, 1=BUY, 2=SELL
    Reward (Ödül): PnL değişimi + Sharpe bonusu - işlem cezası
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self, 
        data: np.ndarray,  # Shape: (N, features) - Tarihsel OHLCV + indicator data
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,  # 0.1% fee (komisyon)
        max_position_size: float = 1.0  # 100% of balance (bakiyenin %100'ü)
    ):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position_size = max_position_size
        
        # State space: 28 features (özellik)
        # [price_features(20), macro(5), position_info(3)]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(28,), 
            dtype=np.float32
        )
        
        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # Episode tracking (Bölüm takibi)
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Trading state (Trading durumu)
        self.balance = initial_balance
        self.position = 0.0  # 0 = no position, >0 = long, <0 = short
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = [initial_balance]
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state
        (Ortamı başlangıç durumuna sıfırla)
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step
        (Bir trading adımı çalıştır)
        
        Returns:
            observation (durum), reward (ödül), terminated (bitti mi), truncated, info
        """
        # Get current price (Mevcut fiyat)
        current_price = self._get_current_price()
        
        # Execute action (Aksiyonu çalıştır)
        reward = self._execute_action(action, current_price)
        
        # Move to next step (Sonraki adıma geç)
        self.current_step += 1
        
        # Check if episode done (Bölüm bitti mi?)
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Update equity curve (Sermaye eğrisini güncelle)
        current_equity = self._calculate_equity(current_price)
        self.equity_curve.append(current_equity)
        
        # Get new observation (Yeni durumu al)
        obs = self._get_observation()
        
        # Info dict
        info = {
            'balance': self.balance,
            'position': self.position,
            'equity': current_equity,
            'sharpe': self._calculate_sharpe() if terminated else 0
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Build state vector from current market data
        (Mevcut piyasa verisinden durum vektörü oluştur)
        """
        # Price features (20): OHLCV ratios, returns, indicators
        # Simplified for now - in real impl, use FeatureExtractor
        price_features = self.data[self.current_step, :20]
        
        # Macro features (5): DXY, VIX, etc.
        macro_features = self.data[self.current_step, 20:25]
        
        # Position info (3): is_in_position, entry_delta, unrealized_pnl
        current_price = self._get_current_price()
        position_info = np.array([
            1.0 if self.position != 0 else 0.0,  # In position?
            (current_price - self.entry_price) / current_price if self.entry_price > 0 else 0.0,
            self._calculate_unrealized_pnl(current_price) / self.balance if self.balance > 0 else 0.0
        ])
        
        obs = np.concatenate([price_features, macro_features, position_info])
        return obs.astype(np.float32)
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Execute trading action and return reward
        (Trading aksiyonunu çalıştır ve ödülü döndür)
        """
        prev_equity = self._calculate_equity(current_price)
        
        if action == 1:  # BUY (AL)
            if self.position <= 0:  # Open long or close short
                self._open_position(1.0, current_price)
        elif action == 2:  # SELL (SAT)
            if self.position >= 0:  # Open short or close long
                self._close_position(current_price)
        # action == 0: HOLD (do nothing)
        
        new_equity = self._calculate_equity(current_price)
        
        # Calculate reward (Ödülü hesapla)
        pnl_change = (new_equity - prev_equity) / prev_equity * 100 if prev_equity > 0 else 0
        
        reward = pnl_change
        reward -= 0.01 * abs(action - 1)  # Small penalty for trading (İşlem cezası)
        
        return reward
    
    def _open_position(self, size: float, price: float):
        """Open a new position (Yeni pozisyon aç)"""
        if self.position != 0:
            self._close_position(price)  # Close existing first
        
        self.position = size * self.max_position_size
        self.entry_price = price
        self.balance -= self.balance * self.transaction_fee  # Fee
    
    def _close_position(self, price: float):
        """Close current position (Mevcut pozisyonu kapat)"""
        if self.position == 0:
            return
        
        pnl = (price - self.entry_price) / self.entry_price * self.position
        self.balance += self.balance * pnl
        self.balance -= self.balance * self.transaction_fee  # Fee
        
        self.trades.append({
            'entry': self.entry_price,
            'exit': price,
            'pnl': pnl,
            'balance': self.balance
        })
        
        self.position = 0.0
        self.entry_price = 0.0
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate total equity (balance + unrealized PnL)"""
        unrealized = self._calculate_unrealized_pnl(current_price)
        return self.balance + unrealized
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss"""
        if self.position == 0 or self.entry_price == 0:
            return 0.0
        return self.balance * ((current_price - self.entry_price) / self.entry_price) * self.position
    
    def _get_current_price(self) -> float:
        """Get current close price from data"""
        # Assuming column 3 is 'close' price
        return self.data[self.current_step, 3]
    
    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio from equity curve"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
