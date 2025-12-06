"""
Efficient transformer blocks for PatchFormer.

Includes both global and windowed attention variants optimized for TFLite.
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional, Tuple


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention layer.
    
    Supports both global and windowed attention patterns.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = layers.Dense(embed_dim * 3, use_bias=use_bias, name="qkv")
        self.proj = layers.Dense(embed_dim, use_bias=use_bias, name="proj")
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(
        self, 
        x: tf.Tensor, 
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
            attention_mask: Optional mask (B, N, N) or (B, 1, N, N)
            training: Whether in training mode
            
        Returns:
            Output tensor (B, N, D)
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, head_dim)
        
        # Scaled dot-product attention
        attn = tf.matmul(q, k, transpose_b=True) * self.scale  # (B, H, N, N)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)
        
        # Apply attention to values
        out = tf.matmul(attn, v)  # (B, H, N, head_dim)
        out = tf.transpose(out, [0, 2, 1, 3])  # (B, N, H, head_dim)
        out = tf.reshape(out, [batch_size, seq_len, self.embed_dim])
        
        # Output projection
        out = self.proj(out)
        out = self.dropout(out, training=training)
        
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


class WindowedMultiHeadSelfAttention(layers.Layer):
    """
    Windowed multi-head self-attention with optional shifted windows (Swin-style).
    
    Partitions the 2D grid into non-overlapping windows and applies
    attention within each window. Supports shifted windows for cross-window
    communication.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        grid_size: Tuple[int, int] = (128, 128),
        dropout_rate: float = 0.1,
        shift_size: int = 0,  # 0 = no shift, window_size//2 = shifted
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.shift_size = shift_size
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Number of windows
        self.num_windows_h = grid_size[0] // window_size
        self.num_windows_w = grid_size[1] // window_size
        
        self.qkv = layers.Dense(embed_dim * 3, name="qkv")
        self.proj = layers.Dense(embed_dim, name="proj")
        self.dropout = layers.Dropout(dropout_rate)
        
        # Relative position bias
        self.relative_position_bias_table = None
        
        # Attention mask for shifted windows (to prevent cross-region attention)
        self.attn_mask = None
    
    def build(self, input_shape):
        # Create relative position bias table
        # (2*window_size-1) * (2*window_size-1) possible relative positions
        num_relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_relative_positions, self.num_heads),
            initializer="zeros",
            trainable=True
        )
        
        # Create relative position index
        coords_h = tf.range(self.window_size)
        coords_w = tf.range(self.window_size)
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords = tf.reshape(coords, [2, -1])
        
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = tf.transpose(relative_coords, [1, 2, 0])
        relative_coords = relative_coords + (self.window_size - 1)
        relative_coords = relative_coords[:, :, 0] * (2 * self.window_size - 1) + relative_coords[:, :, 1]
        
        self.relative_position_index = tf.Variable(
            initial_value=relative_coords,
            trainable=False,
            name="relative_position_index"
        )
        
        # Create attention mask for shifted windows
        if self.shift_size > 0:
            self.attn_mask = self._create_shift_mask()
    
    def _create_shift_mask(self):
        """
        Create attention mask for shifted window attention.
        Prevents attention between tokens from different original windows.
        """
        H, W = self.grid_size
        img_mask = tf.zeros((1, H, W, 1))
        
        # Create region labels
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        
        cnt = 0
        mask_values = []
        for h_slice in h_slices:
            for w_slice in w_slices:
                # Create a mask for this region
                h_start = h_slice.start if h_slice.start else 0
                h_stop = h_slice.stop if h_slice.stop else H
                if h_stop < 0:
                    h_stop = H + h_stop
                w_start = w_slice.start if w_slice.start else 0
                w_stop = w_slice.stop if w_slice.stop else W
                if w_stop < 0:
                    w_stop = W + w_stop
                mask_values.append((h_start, h_stop, w_start, w_stop, cnt))
                cnt += 1
        
        # Build the mask using numpy then convert
        import numpy as np
        img_mask_np = np.zeros((1, H, W, 1))
        for h_start, h_stop, w_start, w_stop, region_id in mask_values:
            img_mask_np[0, h_start:h_stop, w_start:w_stop, 0] = region_id
        
        img_mask = tf.constant(img_mask_np, dtype=tf.float32)
        
        # Partition into windows
        mask_windows = self._partition_windows(img_mask)  # (nW, Wh*Ww, 1)
        mask_windows = tf.squeeze(mask_windows, -1)  # (nW, Wh*Ww)
        
        # Create attention mask: (nW, Wh*Ww, Wh*Ww)
        attn_mask = tf.expand_dims(mask_windows, 2) - tf.expand_dims(mask_windows, 1)
        attn_mask = tf.where(attn_mask != 0, -100.0, 0.0)
        
        return attn_mask
    
    def _partition_windows(self, x: tf.Tensor) -> tf.Tensor:
        """
        Partition input into non-overlapping windows.
        
        Args:
            x: (B, H, W, C)
        Returns:
            (B * num_windows, window_size * window_size, C)
        """
        batch_size = tf.shape(x)[0]
        C = tf.shape(x)[-1]
        
        # Reshape to (B, num_windows_h, window_size, num_windows_w, window_size, C)
        x = tf.reshape(x, [
            batch_size,
            self.num_windows_h, self.window_size,
            self.num_windows_w, self.window_size,
            C
        ])
        
        # Permute to (B, num_windows_h, num_windows_w, window_size, window_size, C)
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        
        # Reshape to (B * num_windows, window_size * window_size, C)
        num_windows = self.num_windows_h * self.num_windows_w
        x = tf.reshape(x, [batch_size * num_windows, self.window_size * self.window_size, C])
        
        return x
    
    def _merge_windows(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """
        Merge windows back to original grid.
        
        Args:
            x: (B * num_windows, window_size * window_size, C)
            batch_size: Original batch size
        Returns:
            (B, H * W, C)
        """
        num_windows = self.num_windows_h * self.num_windows_w
        
        # Reshape to (B, num_windows_h, num_windows_w, window_size, window_size, C)
        x = tf.reshape(x, [
            batch_size,
            self.num_windows_h, self.num_windows_w,
            self.window_size, self.window_size,
            self.embed_dim
        ])
        
        # Permute to (B, num_windows_h, window_size, num_windows_w, window_size, C)
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        
        # Reshape to (B, H * W, C)
        x = tf.reshape(x, [batch_size, self.grid_size[0] * self.grid_size[1], self.embed_dim])
        
        return x
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            x: Input tensor (B, N, D) where N = H * W
            
        Returns:
            Output tensor (B, N, D)
        """
        batch_size = tf.shape(x)[0]
        H, W = self.grid_size
        
        # Reshape to 2D grid
        x_2d = tf.reshape(x, [batch_size, H, W, self.embed_dim])
        
        # Apply cyclic shift for shifted window attention
        if self.shift_size > 0:
            shifted_x = tf.roll(x_2d, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x_2d
        
        # Partition into windows
        x_windows = self._partition_windows(shifted_x)  # (B*nW, Wh*Ww, C)
        
        window_size_sq = self.window_size * self.window_size
        
        # Compute Q, K, V
        qkv = self.qkv(x_windows)
        qkv = tf.reshape(qkv, [-1, window_size_sq, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        
        # Add relative position bias
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, [-1])
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            [window_size_sq, window_size_sq, self.num_heads]
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, 0)
        
        # Apply attention mask for shifted windows
        if self.shift_size > 0 and self.attn_mask is not None:
            num_windows = self.num_windows_h * self.num_windows_w
            # Reshape attn to (B, nW, num_heads, Wh*Ww, Wh*Ww)
            attn = tf.reshape(attn, [batch_size, num_windows, self.num_heads, window_size_sq, window_size_sq])
            # Add mask (broadcast over batch and heads)
            attn = attn + tf.reshape(self.attn_mask, [1, num_windows, 1, window_size_sq, window_size_sq])
            attn = tf.reshape(attn, [-1, self.num_heads, window_size_sq, window_size_sq])
        
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)
        
        # Apply attention
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [-1, window_size_sq, self.embed_dim])
        
        # Output projection
        out = self.proj(out)
        out = self.dropout(out, training=training)
        
        # Merge windows
        out = self._merge_windows(out, batch_size)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            out_2d = tf.reshape(out, [batch_size, H, W, self.embed_dim])
            out_2d = tf.roll(out_2d, shift=[self.shift_size, self.shift_size], axis=[1, 2])
            out = tf.reshape(out_2d, [batch_size, H * W, self.embed_dim])
        
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "grid_size": self.grid_size,
            "dropout_rate": self.dropout_rate,
            "shift_size": self.shift_size,
        })
        return config


class SwinTransformerBlock(layers.Layer):
    """
    Swin Transformer block with alternating regular and shifted window attention.
    
    Combines two consecutive transformer blocks:
    1. Regular windowed attention (W-MSA)
    2. Shifted windowed attention (SW-MSA)
    
    This enables cross-window communication while maintaining efficiency.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        grid_size: Tuple[int, int] = (128, 128),
        ffn_ratio: float = 2.0,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.window_size = window_size
        shift_size = window_size // 2
        
        # Block 1: Regular window attention
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attn1 = WindowedMultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            shift_size=0,  # No shift
            name="w_msa"
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        ffn_hidden = int(embed_dim * ffn_ratio)
        self.ffn1 = tf.keras.Sequential([
            layers.Dense(ffn_hidden, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ], name="ffn1")
        
        # Block 2: Shifted window attention
        self.norm3 = layers.LayerNormalization(epsilon=1e-6, name="norm3")
        self.attn2 = WindowedMultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            shift_size=shift_size,  # Shifted
            name="sw_msa"
        )
        self.norm4 = layers.LayerNormalization(epsilon=1e-6, name="norm4")
        self.ffn2 = tf.keras.Sequential([
            layers.Dense(ffn_hidden, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ], name="ffn2")
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Block 1: W-MSA
        residual = x
        x = self.norm1(x)
        x = self.attn1(x, training=training)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn1(x, training=training)
        x = x + residual
        
        # Block 2: SW-MSA
        residual = x
        x = self.norm3(x)
        x = self.attn2(x, training=training)
        x = x + residual
        
        residual = x
        x = self.norm4(x)
        x = self.ffn2(x, training=training)
        x = x + residual
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "window_size": self.window_size})
        return config


class TransformerBlock(layers.Layer):
    """
    Standard transformer block with pre-norm architecture.
    
    LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 2.0,
        dropout_rate: float = 0.1,
        use_windowed_attention: bool = False,
        window_size: int = 8,
        grid_size: Tuple[int, int] = (128, 128),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.use_windowed = use_windowed_attention
        
        # Layer normalization
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        
        # Attention
        if use_windowed_attention:
            self.attention = WindowedMultiHeadSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                grid_size=grid_size,
                dropout_rate=dropout_rate,
                name="windowed_attention"
            )
        else:
            self.attention = MultiHeadSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                name="attention"
            )
        
        # Feed-forward network
        ffn_hidden = int(embed_dim * ffn_ratio)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_hidden, activation="gelu", name="fc1"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim, name="fc2"),
            layers.Dropout(dropout_rate),
        ], name="ffn")
    
    def call(
        self, 
        x: tf.Tensor, 
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
            attention_mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Output tensor (B, N, D)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        if self.use_windowed:
            x = self.attention(x, training=training)
        else:
            x = self.attention(x, attention_mask=attention_mask, training=training)
        x = x + residual
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, training=training)
        x = x + residual
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_ratio": self.ffn_ratio,
            "use_windowed_attention": self.use_windowed,
        })
        return config


class LightweightTransformerBlock(layers.Layer):
    """
    Lightweight transformer block optimized for efficiency.
    
    Uses:
    - Smaller FFN ratio
    - Windowed attention by default
    - Reduced dropout
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 1.5,
        dropout_rate: float = 0.05,
        window_size: int = 8,
        grid_size: Tuple[int, int] = (128, 128),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        
        # Layer normalization (using batch norm would be more TFLite friendly
        # but layer norm is standard for transformers)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        
        # Windowed attention for efficiency
        self.attention = WindowedMultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            name="attention"
        )
        
        # Smaller FFN
        ffn_hidden = int(embed_dim * ffn_ratio)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_hidden, activation="gelu", name="fc1"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim, name="fc2"),
            layers.Dropout(dropout_rate),
        ], name="ffn")
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, training=training)
        x = x + residual
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, training=training)
        x = x + residual
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
        })
        return config


class AdaptiveTransformerBlock(layers.Layer):
    """
    Transformer block that adapts its input dimension.
    
    Useful when the input dimension changes (e.g., after concatenation).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 2.0,
        dropout_rate: float = 0.1,
        use_windowed_attention: bool = True,
        window_size: int = 8,
        grid_size: Tuple[int, int] = (128, 128),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection if dimensions don't match
        if input_dim != output_dim:
            self.input_proj = layers.Dense(output_dim, name="input_proj")
        else:
            self.input_proj = None
        
        # Standard transformer block
        self.transformer = TransformerBlock(
            embed_dim=output_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout_rate=dropout_rate,
            use_windowed_attention=use_windowed_attention,
            window_size=window_size,
            grid_size=grid_size,
            name="transformer"
        )
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if self.input_proj is not None:
            x = self.input_proj(x)
        return self.transformer(x, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config


class TransformerStack(layers.Layer):
    """
    Stack of transformer blocks with optional Swin-style shifted windows.
    
    When use_swin=True, uses SwinTransformerBlock which alternates between
    regular and shifted window attention for cross-window communication.
    """
    
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 2.0,
        dropout_rate: float = 0.1,
        use_windowed_attention: bool = True,
        window_size: int = 8,
        grid_size: Tuple[int, int] = (128, 128),
        lightweight: bool = False,
        use_swin: bool = True,  # Use Swin-style shifted windows
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.use_swin = use_swin
        
        self.blocks = []
        for i in range(num_layers):
            if use_swin and use_windowed_attention:
                # Use Swin blocks for cross-window communication
                block = SwinTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    grid_size=grid_size,
                    ffn_ratio=ffn_ratio,
                    dropout_rate=dropout_rate,
                    name=f"swin_block_{i}"
                )
            elif lightweight:
                block = LightweightTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout_rate=dropout_rate,
                    window_size=window_size,
                    grid_size=grid_size,
                    name=f"block_{i}"
                )
            else:
                block = TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout_rate=dropout_rate,
                    use_windowed_attention=use_windowed_attention,
                    window_size=window_size,
                    grid_size=grid_size,
                    name=f"block_{i}"
                )
            self.blocks.append(block)
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for block in self.blocks:
            x = block(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_layers": self.num_layers, "use_swin": self.use_swin})
        return config
