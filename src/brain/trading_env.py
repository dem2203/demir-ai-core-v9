import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

class TradingEnv(gym.Env):
    """
    YAPAY ZEKA OYUN ALANI (PROFESSIONAL REWARD EDITION)
    
    AI Ajanı burada eğitilir.
    Ödül Fonksiyonu güncellendi: Sadece karı değil, riski ve istikrarı da ödüllendirir.
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # Aksiyonlar: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # Gözlem Alanı: Fiyatlar ve indikatörler
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32
        )
        
        # Durum Değişkenleri
        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.fees = 0.001 # %0.1 Komisyon (Binance Standart)

    def reset(self, seed=None, options=None):
        """Ortamı sıfırlar"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs, {}

    def step(self, action):
        """Bir sonraki adıma geç"""
        
        current_price = self.df.iloc[self.current_step]['close']
        
        # --- İŞLEM MANTIĞI ---
        
        # 1: BUY (Al)
        if action == 1 and self.balance > 0:
            amount_to_buy = self.balance / current_price
            cost = amount_to_buy * current_price * (1 + self.fees)
            
            if self.balance >= cost:
                self.balance -= cost
                self.crypto_held += amount_to_buy

        # 2: SELL (Sat)
        elif action == 2 and self.crypto_held > 0:
            sale_value = self.crypto_held * current_price
            fee = sale_value * self.fees
            
            self.balance += (sale_value - fee)
            self.crypto_held = 0

        # --- SONRAKİ ADIM ---
        self.current_step += 1
        
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        if terminated:
            next_obs = self.df.iloc[self.current_step].values.astype(np.float32)
        else:
            next_obs = self.df.iloc[self.current_step].values.astype(np.float32)

        # --- GELİŞMİŞ ÖDÜL SİSTEMİ (REWARD ENGINEERING) ---
        
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + (self.crypto_held * current_price)
        
        # 1. Temel Ödül: Net Varlık Değişimi
        reward = self.net_worth - prev_net_worth
        
        # 2. Ceza: İşlem Komisyonu (Sürekli al-sat yapmasın, eminse yapsın)
        if action == 1 or action == 2:
            reward -= (self.net_worth * 0.0005) 

        # 3. Ceza: Büyük Düşüş (Drawdown)
        # Eğer ana paradan %5 aşağı düşerse canı çok yansın (-100 puan)
        if self.net_worth < self.initial_balance * 0.95:
            reward -= 100 

        # 4. Ödül: Yeni Zirve (High Water Mark)
        # Eğer portföy rekor kırarsa bonus ver (+10 puan)
        if self.net_worth > self.max_net_worth:
            reward += 10
            self.max_net_worth = self.net_worth
            
        # 5. Ceza: Hareketsizlik (Hold) ama piyasa düşüyorsa
        # Eğer elinde mal var ve fiyat düşüyorsa ekstra ceza (Stop Loss eğitimi)
        if self.crypto_held > 0 and reward < 0:
            reward *= 1.1 # Cezayı %10 artır

        return next_obs, reward, terminated, truncated, {}

    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}')
