import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

class TradingEnv(gym.Env):
    """
    YAPAY ZEKA OYUN ALANI (GYM ENVIRONMENT)
    
    AI Ajanı burada eğitilir.
    - Gözlem (State): Fiyat, RSI, MACD, Trend, DXY, VIX (44 Veri)
    - Aksiyon (Action): 0=Bekle (Hold), 1=Al (Buy), 2=Sat (Sell)
    - Ödül (Reward): Cüzdandaki kar/zarar değişimi.
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # Aksiyonlar: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # Gözlem Alanı: Mevcut piyasa verileri (Features)
        # Fiyatlar ve indikatörler (Sonsuz sayı olabilir)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32
        )
        
        # Durum Değişkenleri
        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.fees = 0.001 # %0.1 Komisyon

    def reset(self, seed=None, options=None):
        """Ortamı sıfırlar (Yeni oyun başlar)"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        # İlk gözlemi döndür
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs, {}

    def step(self, action):
        """Bir sonraki adıma geç (AI bir hamle yaptı)"""
        
        # Mevcut veriler
        current_price = self.df.iloc[self.current_step]['close']
        
        # --- AKSİYONLAR ---
        # 0: HOLD (Bekle) -> Hiçbir şey yapma
        
        # 1: BUY (Al)
        if action == 1 and self.balance > 0:
            # Tüm parayla al (Basitleştirilmiş)
            amount_to_buy = self.balance / current_price
            cost = amount_to_buy * current_price * (1 + self.fees)
            
            if self.balance >= cost:
                self.balance -= cost
                self.crypto_held += amount_to_buy

        # 2: SELL (Sat)
        elif action == 2 and self.crypto_held > 0:
            # Hepsini sat
            sale_value = self.crypto_held * current_price
            fee = sale_value * self.fees
            
            self.balance += (sale_value - fee)
            self.crypto_held = 0

        # --- SONRAKİ ADIM ---
        self.current_step += 1
        
        # Oyun bitti mi?
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        if terminated:
            next_obs = self.df.iloc[self.current_step].values.astype(np.float32)
        else:
            next_obs = self.df.iloc[self.current_step].values.astype(np.float32)

        # --- ÖDÜL SİSTEMİ (REWARD ENGINEERING) ---
        # AI'ya ne zaman aferin diyeceğiz?
        
        current_net_worth = self.balance + (self.crypto_held * current_price)
        
        # Ödül: Net varlıktaki değişim
        reward = current_net_worth - self.net_worth
        
        # Ekstra Ceza: Sürekli al-sat yapıp komisyon yemesin
        if action != 0: 
            reward -= (current_net_worth * 0.0005) 

        self.net_worth = current_net_worth
        
        return next_obs, reward, terminated, truncated, {}

    def render(self):
        """Ekrana durumu yazdır"""
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}')
