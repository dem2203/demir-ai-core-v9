import numpy as np
import logging
from typing import Optional, Dict
from pathlib import Path
import torch
import torch.nn as nn

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

logger = logging.getLogger("RL_AGENT")

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer-based Feature Extractor for RL Policy
    (RL Policy için Transformer tabanlı Özellik Çıkarıcı)
    
    Uses Multi-Head Attention (Çoklu-Başlı Dikkat) to process temporal features
    Matches TimeNet architecture (TimeNet mimarisiyle uyumlu)
    """
    
    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        n_input_features = observation_space.shape[0]  # 37 (FIXED)
        
        # Input projection (Giriş projeksiyon)
        self.input_proj = nn.Linear(n_input_features, 64)
        
        # Transformer Encoder Layer (Transformatör Kodlayıcı Katmanı)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,  # Model boyutu
            nhead=4,  # Dikkat başlığı sayısı
            dim_feedforward=128,  # Feed-forward katman boyutu
            dropout=0.1,
            activation='gelu',  # GELU activation (GPT-style)
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2  # 2 Transformer katmanı
        )
        
        # Output projection (Çıkış projeksiyon)
        self.output_proj = nn.Linear(64, features_dim)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer
        (Transformer'dan ileri geçiş)
        
        Args:
            observations: (batch, 37) - State vector (FIXED)
        Returns:
            features: (batch, 64) - Encoded features
        """
        # Add sequence dimension for Transformer (1 timestep)
        # (batch, 28) -> (batch, 1, 28)
        x = observations.unsqueeze(1)
        
        # Project to hidden dim (Gizli boyuta projele)
        # (batch, 1, 28) -> (batch, 1, 64)
        x = self.input_proj(x)
        
        # Transformer encoding (Transformatör kodlama)
        # (batch, 1, 64) -> (batch, 1, 64)
        x = self.transformer(x)
        
        # Remove sequence dim and project (Boyutu kaldır ve projele)
        # (batch, 1, 64) -> (batch, 64) -> (batch, features_dim)
        x = x.squeeze(1)
        x = self.output_proj(x)
        
        return x


class RLAgent:
    """
    Reinforcement Learning Trading Agent using PPO
    (PPO kullanan Pekiştirmeli Öğrenme Trading Ajanı)
    
    Architecture (Mimari):
        State → Transformer Encoder → Actor (Policy Network - Politika Ağı)
                                   → Critic (Value Network - Değer Ağı)
    
    The agent learns to maximize cumulative reward (Sharpe ratio-weighted PnL)
    (Ajan kümülatif ödülü (Sharpe ile ağırlıklandırılmış kar/zarar) maksimize etmeyi öğrenir)
    """
    
    def __init__(self, storage_dir: str = "src/brain/rl_agent/storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[PPO] = None
        self.is_trained = False
        
    def create_model(self, env, learning_rate: float = 3e-4):
        """
        Create PPO model with Transformer policy
        (Transformer politikası ile PPO modeli oluştur)
        
        Args:
            env: TradingEnv instance (Trading ortamı örneği)
            learning_rate: Adam optimizer LR (Adam optimize edici öğrenme oranı)
        """
        # Custom policy with Transformer feature extractor
        # (Transformatör özellik çıkarıcılı özel politika)
        policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            net_arch=dict(
                pi=[32],  # Actor network (Aktör ağı) - Policy çıkışı
                vf=[32]   # Critic network (Kritik ağı) - Value çıkışı
            ),
            activation_fn=nn.ReLU
        )
        
        self.model = PPO(
            policy=ActorCriticPolicy,
            env=env,
            learning_rate=learning_rate,
            n_steps=2048,  # Batch boyutu (her güncelleme için)
            batch_size=64,  # Mini-batch boyutu
            n_epochs=10,  # Her rollout için epoch sayısı
            gamma=0.99,  # Discount factor (İndirim faktörü)
            gae_lambda=0.95,  # GAE lambda (Genelleştirilmiş Avantaj Tahmini)
            clip_range=0.2,  # PPO clip range (PPO kırpma aralığı)
            ent_coef=0.01,  # Entropy coefficient (Entropi katsayısı) - Keşif teşvik eder
            vf_coef=0.5,  # Value function coefficient (Değer fonksiyonu katsayısı)
            max_grad_norm=0.5,  # Gradient clipping (Gradyan kırpma)
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='auto'  # GPU varsa kullan
        )
        
        logger.info("✅ RL Agent model created with Transformer architecture")
        
    def train(self, total_timesteps: int = 100_000, tb_log_name: str = "ppo_trader"):
        """
        Train the agent on the environment
        (Ajanı ortamda eğit)
        
        Args:
            total_timesteps: Total training steps (Toplam eğitim adımı)
            tb_log_name: TensorBoard log name (TensorBoard log adı)
        """
        if self.model is None:
            raise ValueError("Model not created! Call create_model() first.")
        
        logger.info(f"🚀 Starting RL training for {total_timesteps} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=True
        )
        
        self.is_trained = True
        logger.info("✅ Training complete!")
        
    def save(self, filename: str = "ppo_btcusdt_v1"):
        """Save trained model (Eğitilmiş modeli kaydet)"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        save_path = self.storage_dir / f"{filename}.zip"
        self.model.save(save_path)
        logger.info(f"💾 Model saved to {save_path}")
        
    def load(self, filename: str = "ppo_btcusdt_v1"):
        """Load trained model (Eğitilmiş modeli yükle)"""
        load_path = self.storage_dir / f"{filename}.zip"
        
        if not load_path.exists():
            logger.warning(f"⚠️ Model file not found: {load_path}")
            return False
        
        try:
            self.model = PPO.load(load_path)
            self.is_trained = True
            logger.info(f"✅ Model loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> tuple[int, float]:
        """
        Predict action from current state
        (Mevcut durumdan aksiyon tahmin et)
        
        Args:
            state: (37,) state vector (durum vektörü) - FIXED
            deterministic: Use deterministic policy (Deterministik politika kullan)
        
        Returns:
            action: 0=HOLD, 1=BUY, 2=SELL
            confidence: Action probability (Aksiyon olasılığı) 0-100%
        """
        if self.model is None or not self.is_trained:
            # Fallback: return neutral action (Yedek: nötr aksiyon döndür)
            return 0, 0.0
        
        action, _states = self.model.predict(state, deterministic=deterministic)
        
        # Get action probabilities for confidence (Güven için aksiyon olasılıklarını al)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]
        
        confidence = float(probs[action] * 100)  # Convert to percentage
        
        return int(action), confidence
    
    def get_metrics(self) -> Dict:
        """Get training metrics (Eğitim metriklerini al)"""
        if self.model is None:
            return {}
        
        return {
            'is_trained': self.is_trained,
            'n_steps': self.model.n_steps,
            'learning_rate': self.model.learning_rate,
            'gamma': self.model.gamma
        }
