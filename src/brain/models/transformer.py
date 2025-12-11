import tensorflow as tf
from tensorflow.keras import layers, models
import logging

logger = logging.getLogger("TIMENET_ARCHITECT")

class TimeNet:
    """
    TimeNet: A Lightweight Time-Series Transformer for Financial Forecasting.
    Replaces traditional LSTM with Self-Attention mechanisms to capture 
    long-range dependencies and non-linear patterns in market data.
    
    Architecture:
    - Input Embedding (via Dense projection)
    - Positional Encoding (Learnable)
    - Multi-Head Self-Attention Blocks
    - Global Average Pooling
    - Dense Output Head
    """
    
    def __init__(self, input_shape, num_heads=4, head_size=64, num_layers=2, dropout=0.1):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.dropout = dropout
        
    def _transformer_encoder(self, inputs):
        """
        Single Transformer Encoder Block with Residual Connection.
        """
        # Attention Layer
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )(x, x)
        x = layers.Dropout(self.dropout)(x)
        res = x + inputs # Residual Connection 1
        
        # Feed Forward Layer
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Dense(self.head_size * 2, activation="relu")(x) # Expand
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(inputs.shape[-1])(x) # Project back
        return x + res # Residual Connection 2

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # 1. Input Projection (Embedding)
        # Financial data is continuous, so we project it to a higher dim space
        x = layers.Dense(self.head_size)(inputs)
        
        # 2. Positional Encoding (Learnable)
        # We simply add a learnable vector to give time context
        positions = layers.Embedding(
            input_dim=self.input_shape[0], output_dim=self.head_size
        )(tf.range(start=0, limit=self.input_shape[0], delta=1))
        x = x + positions
        
        # 3. Transformer Blocks
        for _ in range(self.num_layers):
            x = self._transformer_encoder(x)
            
        # 4. Output Head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Binary Classification: 1 (Up) / 0 (Down)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="TimeNet_v1")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return model

    @staticmethod
    def load(path):
        return models.load_model(path)
