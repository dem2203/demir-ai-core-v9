import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("TRADING_ENV_PRO")

class TradingEnv(gym.Env):
    """
    DEMIR AI V15.0 - RISK AVERSE ENVIRONMENT
    Ödül Fonksiyonu: Sadece kâr değil, riske göre ayarlanmış getiri (Sortino/Sharpe benzeri).
    Drawdown (Sermaye erimesi) ağır cezalandırılır.
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: Tüm feature sütunları
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0 # 0: Yok, 1: Long (Short yok şimdilik)
        self.entry_price = 0
        self.max_balance = self.initial_balance # Drawdown hesabı için
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Şu anki satırı döndür
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # --- İŞLEM MANTIĞI ---
        reward = 0
        
        # 1. BUY (Eğer pozisyon yoksa al)
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            # İşlem ücreti cezası (Komisyon simülasyonu)
            reward -= 0.001 
            
        # 2. SELL (Eğer pozisyon varsa sat)
        elif action == 2 and self.position == 1:
            self.position = 0
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # ÖDÜL FONKSİYONU (RISK AVERSE)
            if pnl_pct > 0:
                # Kâr varsa ödül, ama kârın büyüklüğüne göre logaritmik (aşırı hırsı engelle)
                reward += pnl_pct * 10 
            else:
                # Zarar varsa CEZA x 2 (Kaybetmekten nefret etmeli)
                reward += pnl_pct * 20 
                
            self.balance *= (1 + pnl_pct)

        # 3. HOLD (Pozisyonu koru veya bekle)
        elif action == 0:
            if self.position == 1:
                # Pozisyondaysa ve fiyat artıyorsa ufak ödül (Trend takibi)
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                if unrealized_pnl > 0:
                    reward += 0.0001
            else:
                # Nakitteyse ve piyasa düşüyorsa ödül (Düşüşten korunma)
                # Bunu hesaplamak için bir önceki fiyata bakmak lazım ama şimdilik basit tutalım.
                pass

        # --- DRAWDOWN CEZASI ---
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        
        drawdown = (self.max_balance - self.balance) / self.max_balance
        if drawdown > 0.05: # %5'ten fazla erime varsa
            reward -= drawdown * 5 # Ağır ceza

        # Sonraki adım
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # İflas kontrolü
        if self.balance < self.initial_balance * 0.5: # %50 kayıp
            done = True
            reward -= 100 # Oyun bitti cezası

        info = {'balance': self.balance}
        
        return self._next_observation(), reward, done, False, info
