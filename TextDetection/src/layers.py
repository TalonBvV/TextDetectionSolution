"""
Custom layers for PatchFormer architecture.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class PatchExtractor(layers.Layer):
    """
    Extracts and flattens patches from input images.
    
    Converts (B, H, W, C) -> (B, num_patches, patch_dim)
    where patch_dim = patch_h * patch_w * C
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (8, 8),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    
    def call(self, images: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(images)[0]
        
        # Extract patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        
        # Get dimensions
        patch_h = tf.shape(patches)[1]
        patch_w = tf.shape(patches)[2]
        patch_dim = patches.shape[-1]  # patch_size[0] * patch_size[1] * channels
        
        # Reshape to (batch, num_patches, patch_dim)
        patches = tf.reshape(patches, [batch_size, patch_h * patch_w, patch_dim])
        
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEmbedding(layers.Layer):
    """
    Projects flattened patches to embedding dimension with optional projection.
    
    Converts (B, num_patches, patch_dim) -> (B, num_patches, embed_dim)
    Also provides spatial view: (B, grid_h, grid_w, embed_dim)
    """
    
    def __init__(
        self,
        embed_dim: int,
        patch_dim: int,
        grid_size: Tuple[int, int],
        use_projection: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_dim = patch_dim
        self.grid_size = grid_size
        self.use_projection = use_projection
        
        if use_projection and patch_dim != embed_dim:
            self.projection = layers.Dense(embed_dim, name="patch_projection")
        else:
            self.projection = None
    
    def call(self, patches: tf.Tensor, return_spatial: bool = False) -> tf.Tensor:
        """
        Args:
            patches: (B, num_patches, patch_dim)
            return_spatial: If True, return (B, grid_h, grid_w, dim)
        """
        if self.projection is not None:
            patches = self.projection(patches)
        
        if return_spatial:
            batch_size = tf.shape(patches)[0]
            patches = tf.reshape(
                patches, 
                [batch_size, self.grid_size[0], self.grid_size[1], -1]
            )
        
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "patch_dim": self.patch_dim,
            "grid_size": self.grid_size,
            "use_projection": self.use_projection,
        })
        return config


class DepthwiseSeparableConv(layers.Layer):
    """
    Depthwise separable convolution - efficient for TFLite.
    
    Depthwise conv (spatial) + Pointwise conv (channel mixing)
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        use_bias: bool = False,
        activation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            name="depthwise"
        )
        self.pointwise = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=use_bias,
            name="pointwise"
        )
        self.bn = layers.BatchNormalization(name="bn")
        self.activation = layers.Activation(activation or "gelu")
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config


class SqueezeExcitation(layers.Layer):
    """
    Squeeze-and-Excitation attention block for channel recalibration.
    """
    
    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_ratio, 8)
        
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Dense(reduced_channels, activation="relu", name="fc1")
        self.fc2 = layers.Dense(channels, activation="sigmoid", name="fc2")
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Squeeze
        squeeze = self.global_pool(x)
        
        # Excitation
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        
        # Scale
        return x * excitation
    
    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class ConvBlock(layers.Layer):
    """
    Convolutional block with optional squeeze-excitation and downsampling.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        use_squeeze_excitation: bool = True,
        se_ratio: int = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.use_se = use_squeeze_excitation
        
        self.conv = DepthwiseSeparableConv(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            name="dsconv"
        )
        
        if use_squeeze_excitation:
            self.se = SqueezeExcitation(reduction_ratio=se_ratio, name="se")
        
        # Residual connection if dimensions match
        self.use_residual = strides == 1
        if self.use_residual:
            self.residual_proj = None  # Will be built if needed
    
    def build(self, input_shape):
        if self.use_residual and input_shape[-1] != self.filters:
            self.residual_proj = layers.Conv2D(
                self.filters, 1, padding="same", name="residual_proj"
            )
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = x
        
        out = self.conv(x, training=training)
        
        if self.use_se:
            out = self.se(out)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            out = out + residual
        
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "strides": self.strides,
            "use_squeeze_excitation": self.use_se,
        })
        return config


class GatingMechanism(layers.Layer):
    """
    Gating mechanism for combining auxiliary features with main embeddings.
    
    Instead of simple concatenation, uses learned gates to control
    how auxiliary information flows into the main representation.
    This prevents gradient scale issues and allows the model to learn
    optimal feature integration.
    
    FIX: Prevents gate collapse by:
    1. Normalizing aux features before gating to ensure similar scales
    2. Using separate projections for gate computation to balance main/aux influence
    3. Adding learnable minimum gate to ensure aux is always somewhat used
    4. Initializing gate bias to encourage non-zero initial contribution
    
    gate = sigmoid(W_g_main * main + W_g_aux * aux + bias)
    output = main + (min_gate + (1-min_gate) * gate) * (W_v * aux)
    """
    
    def __init__(
        self,
        main_dim: int,
        aux_dim: int,
        output_dim: Optional[int] = None,
        min_gate: float = 0.1,  # Minimum gate value to prevent collapse
        **kwargs
    ):
        super().__init__(**kwargs)
        self.main_dim = main_dim
        self.aux_dim = aux_dim
        self.output_dim = output_dim or main_dim
        self.min_gate = min_gate
        
        # FIX: Separate projections for main and aux in gate computation
        # This prevents the larger main features from dominating
        self.gate_main_proj = layers.Dense(self.output_dim, use_bias=False, name="gate_main")
        self.gate_aux_proj = layers.Dense(self.output_dim, use_bias=False, name="gate_aux")
        
        # Gate bias initialized to small positive value to encourage initial aux contribution
        self.gate_bias = None  # Will be created in build()
        
        # FIX: Normalize auxiliary features before processing
        self.aux_norm = layers.LayerNormalization(epsilon=1e-6, name="aux_norm")
        
        # Value projection for auxiliary features
        self.value_proj = layers.Dense(self.output_dim, name="value")
        
        # Optional projection for main if dimensions differ
        self.main_proj = None
        if main_dim != self.output_dim:
            self.main_proj = layers.Dense(self.output_dim, name="main_proj")
        
        # Layer normalization for stability
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
    
    def build(self, input_shape):
        # Initialize gate bias to encourage non-zero gating (around 0.3-0.5 initial gate)
        self.gate_bias = self.add_weight(
            name="gate_bias",
            shape=(self.output_dim,),
            initializer=tf.keras.initializers.Constant(0.0),  # sigmoid(0) = 0.5
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, main: tf.Tensor, aux: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            main: Main embedding (B, N, main_dim)
            aux: Auxiliary features (B, N, aux_dim)
            training: Whether in training mode
            
        Returns:
            Gated output (B, N, output_dim)
        """
        # Project main if needed
        if self.main_proj is not None:
            main_out = self.main_proj(main)
        else:
            main_out = main
        
        # FIX: Normalize auxiliary features to similar scale as main
        aux_normalized = self.aux_norm(aux)
        
        # FIX: Compute gate using separate projections for balanced influence
        gate_main = self.gate_main_proj(main)
        gate_aux = self.gate_aux_proj(aux_normalized)
        gate_logits = gate_main + gate_aux + self.gate_bias
        gate = tf.sigmoid(gate_logits)
        
        # FIX: Apply minimum gate to prevent collapse
        # effective_gate = min_gate + (1 - min_gate) * gate
        # This ensures gate is always at least min_gate
        effective_gate = self.min_gate + (1.0 - self.min_gate) * gate
        
        # Project auxiliary to value
        value = self.value_proj(aux_normalized)
        
        # Gated addition with minimum contribution
        output = main_out + effective_gate * value
        output = self.norm(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "main_dim": self.main_dim,
            "aux_dim": self.aux_dim,
            "output_dim": self.output_dim,
            "min_gate": self.min_gate,
        })
        return config


class CrossAttentionFusion(layers.Layer):
    """
    Cross-attention for fusing auxiliary features into main embeddings.
    
    Main embeddings attend to auxiliary features to selectively
    incorporate relevant information.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = layers.Dense(embed_dim, name="q_proj")
        self.k_proj = layers.Dense(embed_dim, name="k_proj")
        self.v_proj = layers.Dense(embed_dim, name="v_proj")
        self.out_proj = layers.Dense(embed_dim, name="out_proj")
        
        self.dropout = layers.Dropout(dropout_rate)
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
    
    def call(
        self, 
        main: tf.Tensor, 
        aux: tf.Tensor, 
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            main: Main embedding (B, N, D) - queries
            aux: Auxiliary features (B, N, D_aux) - keys/values
            
        Returns:
            Fused output (B, N, D)
        """
        batch_size = tf.shape(main)[0]
        seq_len = tf.shape(main)[1]
        
        # Project
        q = self.q_proj(main)
        k = self.k_proj(aux)
        v = self.v_proj(aux)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose to (B, heads, N, head_dim)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Attention
        scale = tf.cast(self.head_dim, tf.float32) ** -0.5
        attn = tf.matmul(q, k, transpose_b=True) * scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)
        
        # Apply attention to values
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch_size, seq_len, self.embed_dim])
        
        # Output projection + residual
        out = self.out_proj(out)
        out = self.dropout(out, training=training)
        out = self.norm(main + out)
        
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config


class Learned2DPositionalEncoding(layers.Layer):
    """
    Learned 2D positional encoding for spatial feature maps.
    
    Creates separate row and column embeddings that are broadcast and combined.
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        embed_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
    
    def build(self, input_shape):
        # Split embedding dimension between row and column
        col_dim = self.embed_dim // 2
        row_dim = self.embed_dim - col_dim
        
        # Learnable position embeddings
        self.row_embed = self.add_weight(
            name="row_embed",
            shape=(self.height, row_dim),
            initializer="truncated_normal",
            trainable=True
        )
        self.col_embed = self.add_weight(
            name="col_embed",
            shape=(self.width, col_dim),
            initializer="truncated_normal",
            trainable=True
        )
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x: Input tensor of shape (B, H, W, C) or (B, N, C)
        """
        # Get input shape
        shape = tf.shape(x)
        is_sequence = len(x.shape) == 3
        
        # Create 2D position encoding
        # row_embed: (H, row_dim) -> (H, 1, row_dim) -> (H, W, row_dim)
        row_pos = tf.expand_dims(self.row_embed, axis=1)
        row_pos = tf.tile(row_pos, [1, self.width, 1])
        
        # col_embed: (W, col_dim) -> (1, W, col_dim) -> (H, W, col_dim)
        col_pos = tf.expand_dims(self.col_embed, axis=0)
        col_pos = tf.tile(col_pos, [self.height, 1, 1])
        
        # Concatenate: (H, W, embed_dim)
        pos_encoding = tf.concat([row_pos, col_pos], axis=-1)
        
        if is_sequence:
            # Flatten to (H*W, embed_dim) and add batch dimension
            pos_encoding = tf.reshape(pos_encoding, [self.height * self.width, self.embed_dim])
            pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        else:
            # Add batch dimension: (1, H, W, embed_dim)
            pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        
        return x + pos_encoding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "height": self.height,
            "width": self.width,
            "embed_dim": self.embed_dim,
        })
        return config


class PatchMasking(layers.Layer):
    """
    Random patch masking for masked autoencoder training.
    
    Masks a percentage of patches and returns masked/unmasked indices.
    """
    
    def __init__(self, mask_ratio: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
    
    def call(
        self, 
        x: tf.Tensor, 
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Args:
            x: Input tensor (B, N, D)
            training: Whether in training mode
            
        Returns:
            masked_x: Input with masked patches zeroed (B, N, D)
            mask: Boolean mask indicating which patches are masked (B, N)
            indices: Random permutation used for masking (B, N)
        """
        if not training:
            # During inference, return input unchanged
            batch_size = tf.shape(x)[0]
            num_patches = tf.shape(x)[1]
            mask = tf.zeros((batch_size, num_patches), dtype=tf.bool)
            indices = tf.tile(
                tf.expand_dims(tf.range(num_patches), 0),
                [batch_size, 1]
            )
            return x, mask, indices
        
        batch_size = tf.shape(x)[0]
        num_patches = tf.shape(x)[1]
        num_mask = tf.cast(
            tf.cast(num_patches, tf.float32) * self.mask_ratio, 
            tf.int32
        )
        
        # Generate random permutation for each batch
        # Using uniform random and argsort for shuffling
        noise = tf.random.uniform((batch_size, num_patches))
        indices = tf.argsort(noise, axis=1)
        
        # Create mask: True for masked positions
        mask_indices = indices[:, :num_mask]
        mask = tf.reduce_any(
            tf.equal(
                tf.expand_dims(tf.range(num_patches), 0),
                tf.expand_dims(mask_indices, -1)
            ),
            axis=1
        )
        
        # Apply mask (zero out masked patches)
        mask_expanded = tf.expand_dims(tf.cast(mask, x.dtype), -1)
        masked_x = x * (1.0 - mask_expanded)
        
        return masked_x, mask, indices
    
    def get_config(self):
        config = super().get_config()
        config.update({"mask_ratio": self.mask_ratio})
        return config


class MLP(layers.Layer):
    """
    Simple MLP block with GELU activation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self.fc1 = layers.Dense(hidden_dim, name="fc1")
        self.fc2 = layers.Dense(output_dim, name="fc2")
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = layers.Activation("gelu")
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class ProjectionHead(layers.Layer):
    """
    Projection head for contrastive learning.
    
    Projects embeddings to a lower-dimensional space and L2 normalizes.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = layers.Dense(hidden_dim, activation="gelu", name="fc1")
        self.fc2 = layers.Dense(output_dim, name="fc2")
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        # L2 normalize
        x = tf.nn.l2_normalize(x, axis=-1)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        })
        return config


class TokenClassificationHead(layers.Layer):
    """
    Head for token-level classification (interior/boundary/background).
    """
    
    def __init__(self, num_classes: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.classifier = layers.Dense(num_classes, name="classifier")
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.classifier(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


class ContextHead(layers.Layer):
    """
    Head for generating context features.
    """
    
    def __init__(self, output_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.fc = layers.Dense(output_dim, activation="gelu", name="fc")
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.fc(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config


class SegmentationHead(layers.Layer):
    """
    Final classification head for per-token segmentation.
    """
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.classifier = layers.Dense(num_classes, name="classifier")
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.classifier(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


# =============================================================================
# v2.1 ADDITIONS: SegFormer + DBNet++ inspired improvements
# =============================================================================

class OverlappingPatchEmbed(layers.Layer):
    """
    Overlapping patch embedding (from SegFormer).
    
    Unlike non-overlapping patches, this uses a strided convolution
    where stride < kernel_size, creating overlapping patches.
    
    Benefits:
    - Eliminates boundary artifacts at patch edges
    - Provides smoother feature transitions
    - Better preserves local continuity
    
    Example: kernel=8, stride=4 means 50% overlap
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        patch_size: int = 8,
        stride: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = stride
        
        # Overlapping patch projection
        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=stride,
            padding='same',
            name="proj"
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
    
    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, Tuple[int, int]]:
        """
        Args:
            x: Input image (B, H, W, C)
            
        Returns:
            patches: (B, num_patches, embed_dim)
            grid_size: (H', W') spatial dimensions after patching
        """
        # Project with overlapping patches
        x = self.proj(x)  # (B, H', W', embed_dim)
        
        B = tf.shape(x)[0]
        H, W = x.shape[1], x.shape[2]
        
        # Flatten spatial dimensions
        x = tf.reshape(x, [B, H * W, self.embed_dim])
        x = self.norm(x)
        
        return x, (H, W)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "stride": self.stride,
        })
        return config


class DeformableConv2D(layers.Layer):
    """
    Deformable Convolution v2 (from DBNet++).
    
    Learns spatial offsets for each sampling position, allowing the
    convolution to adapt its receptive field to the content.
    
    Benefits for text detection:
    - Adapts to curved text
    - Handles rotated text better
    - Focuses on text-relevant regions
    
    Implementation: Predicts 2*k*k offsets (x,y per position) and
    optionally modulation scalars, then samples at offset positions.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        use_modulation: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_modulation = use_modulation
        
        self.k = kernel_size
        self.num_points = kernel_size * kernel_size
        
        # Offset prediction: 2 (x,y) per sampling point
        offset_channels = 2 * self.num_points
        if use_modulation:
            # Also predict modulation scalar per point
            offset_channels += self.num_points
        
        self.offset_conv = layers.Conv2D(
            offset_channels,
            kernel_size=kernel_size,
            padding='same',
            name="offset_conv"
        )
        
        # Main convolution weights
        self.weight_conv = layers.Conv2D(
            filters,
            kernel_size=1,
            padding='same',
            name="weight_conv"
        )
        
        # Learnable per-point weights (replaces standard kernel)
        self.point_weights = None  # Built dynamically
        
    def build(self, input_shape):
        C = input_shape[-1]
        # Weight for each sampling point
        self.point_conv = layers.Conv2D(
            self.filters * self.num_points,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name="point_conv"
        )
        super().build(input_shape)
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            x: Input features (B, H, W, C)
            
        Returns:
            Output features (B, H, W, filters)
        """
        B = tf.shape(x)[0]
        H, W = tf.shape(x)[1], tf.shape(x)[2]
        C = x.shape[-1]
        
        # Predict offsets (and modulation if enabled)
        offset_pred = self.offset_conv(x)
        
        if self.use_modulation:
            # Split into offsets and modulation
            offsets = offset_pred[:, :, :, :2*self.num_points]
            modulation = tf.sigmoid(offset_pred[:, :, :, 2*self.num_points:])
        else:
            offsets = offset_pred
            modulation = tf.ones([B, H, W, self.num_points])
        
        # Reshape offsets to (B, H, W, num_points, 2)
        offsets = tf.reshape(offsets, [B, H, W, self.num_points, 2])
        
        # Generate base sampling grid
        # For a 3x3 kernel centered at (0,0): [(-1,-1), (-1,0), ..., (1,1)]
        base_offsets = self._get_base_offsets()  # (num_points, 2)
        
        # Create coordinate grid
        y_coords = tf.range(H, dtype=tf.float32)
        x_coords = tf.range(W, dtype=tf.float32)
        y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing='ij')
        grid = tf.stack([y_grid, x_grid], axis=-1)  # (H, W, 2)
        grid = tf.expand_dims(grid, 0)  # (1, H, W, 2)
        grid = tf.expand_dims(grid, 3)  # (1, H, W, 1, 2)
        
        # Add base offsets and learned offsets
        base_offsets = tf.reshape(base_offsets, [1, 1, 1, self.num_points, 2])
        sample_coords = grid + base_offsets + offsets  # (B, H, W, num_points, 2)
        
        # Bilinear sampling at each offset position
        sampled = self._bilinear_sample(x, sample_coords)  # (B, H, W, num_points, C)
        
        # Apply modulation
        modulation = tf.expand_dims(modulation, -1)  # (B, H, W, num_points, 1)
        sampled = sampled * modulation
        
        # Aggregate across sampling points
        sampled = tf.reshape(sampled, [B, H, W, self.num_points * C])
        
        # Project to output channels
        output = self.weight_conv(sampled)
        
        return output
    
    def _get_base_offsets(self):
        """Generate base sampling offsets for kernel."""
        k = self.kernel_size
        center = k // 2
        offsets = []
        for i in range(k):
            for j in range(k):
                offsets.append([i - center, j - center])
        return tf.constant(offsets, dtype=tf.float32)
    
    def _bilinear_sample(
        self, 
        x: tf.Tensor, 
        coords: tf.Tensor
    ) -> tf.Tensor:
        """
        Bilinear sampling at arbitrary coordinates.
        
        Args:
            x: Input (B, H, W, C)
            coords: Sample coordinates (B, H, W, num_points, 2) in [y, x] format
            
        Returns:
            Sampled values (B, H, W, num_points, C)
        """
        B = tf.shape(x)[0]
        H, W = tf.shape(x)[1], tf.shape(x)[2]
        C = x.shape[-1]
        
        # Flatten spatial dims for easier indexing
        x_flat = tf.reshape(x, [B, H * W, C])
        
        # Get coordinates
        y = coords[:, :, :, :, 0]
        x_coord = coords[:, :, :, :, 1]
        
        # Clamp to valid range
        y = tf.clip_by_value(y, 0, tf.cast(H - 1, tf.float32))
        x_coord = tf.clip_by_value(x_coord, 0, tf.cast(W - 1, tf.float32))
        
        # Get corner coordinates
        y0 = tf.floor(y)
        x0 = tf.floor(x_coord)
        y1 = y0 + 1
        x1 = x0 + 1
        
        # Clamp corners
        y0 = tf.clip_by_value(y0, 0, tf.cast(H - 1, tf.float32))
        y1 = tf.clip_by_value(y1, 0, tf.cast(H - 1, tf.float32))
        x0 = tf.clip_by_value(x0, 0, tf.cast(W - 1, tf.float32))
        x1 = tf.clip_by_value(x1, 0, tf.cast(W - 1, tf.float32))
        
        # Compute interpolation weights
        wy1 = y - y0
        wy0 = 1.0 - wy1
        wx1 = x_coord - x0
        wx0 = 1.0 - wx1
        
        # Convert to indices
        y0_idx = tf.cast(y0, tf.int32)
        y1_idx = tf.cast(y1, tf.int32)
        x0_idx = tf.cast(x0, tf.int32)
        x1_idx = tf.cast(x1, tf.int32)
        
        # Compute flat indices
        idx_00 = y0_idx * W + x0_idx
        idx_01 = y0_idx * W + x1_idx
        idx_10 = y1_idx * W + x0_idx
        idx_11 = y1_idx * W + x1_idx
        
        # Gather values at corners
        # Shape: (B, H, W, num_points)
        def gather_nd_batch(indices):
            # indices: (B, H, W, num_points)
            B_dim = tf.shape(indices)[0]
            batch_idx = tf.tile(
                tf.reshape(tf.range(B_dim), [B_dim, 1, 1, 1]),
                [1, tf.shape(indices)[1], tf.shape(indices)[2], tf.shape(indices)[3]]
            )
            full_idx = tf.stack([batch_idx, indices], axis=-1)
            return tf.gather_nd(x_flat, full_idx)
        
        v00 = gather_nd_batch(idx_00)  # (B, H, W, num_points, C)
        v01 = gather_nd_batch(idx_01)
        v10 = gather_nd_batch(idx_10)
        v11 = gather_nd_batch(idx_11)
        
        # Bilinear interpolation
        wx0 = tf.expand_dims(wx0, -1)
        wx1 = tf.expand_dims(wx1, -1)
        wy0 = tf.expand_dims(wy0, -1)
        wy1 = tf.expand_dims(wy1, -1)
        
        output = (v00 * wy0 * wx0 + 
                  v01 * wy0 * wx1 + 
                  v10 * wy1 * wx0 + 
                  v11 * wy1 * wx1)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "use_modulation": self.use_modulation,
        })
        return config


class AdaptiveScaleFusion(layers.Layer):
    """
    Adaptive Scale Fusion module (from DBNet++).
    
    Learns to weight features from different scales based on
    spatial attention, rather than using fixed weights.
    
    Benefits:
    - Adapts scale selection per location
    - Small text regions use high-res features
    - Large text regions use low-res features
    - Learned end-to-end
    """
    
    def __init__(
        self,
        out_channels: int,
        num_scales: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.num_scales = num_scales
        
        # Spatial attention for scale selection
        self.attention_conv = layers.Conv2D(
            num_scales,
            kernel_size=3,
            padding='same',
            name="attention_conv"
        )
        
        # Per-scale projection
        self.scale_projs = [
            layers.Conv2D(out_channels, 1, padding='same', name=f"scale_proj_{i}")
            for i in range(num_scales)
        ]
        
        # Final fusion
        self.fusion_conv = DepthwiseSeparableConv(
            out_channels,
            kernel_size=3,
            name="fusion_conv"
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
    
    def call(
        self,
        multi_scale_features: list,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            multi_scale_features: List of (B, H, W, C) tensors at different scales
                                  All should be resized to same spatial size
            
        Returns:
            Fused features (B, H, W, out_channels)
        """
        assert len(multi_scale_features) == self.num_scales
        
        # Stack features: (B, H, W, C, num_scales)
        stacked = tf.stack([
            self.scale_projs[i](f) for i, f in enumerate(multi_scale_features)
        ], axis=-1)
        
        # Compute spatial attention weights
        # Use the finest scale for attention computation
        attn_input = multi_scale_features[0]
        attention = self.attention_conv(attn_input)  # (B, H, W, num_scales)
        attention = tf.nn.softmax(attention, axis=-1)
        attention = tf.expand_dims(attention, -2)  # (B, H, W, 1, num_scales)
        
        # Weighted sum across scales
        fused = tf.reduce_sum(stacked * attention, axis=-1)  # (B, H, W, out_channels)
        
        # Final refinement
        fused = self.fusion_conv(fused, training=training)
        fused = self.norm(fused)
        
        return fused
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "num_scales": self.num_scales,
        })
        return config


class MixFFN(layers.Layer):
    """
    Mix Feed-Forward Network (from SegFormer).
    
    Replaces standard FFN with one that includes a 3×3 depthwise
    convolution, which implicitly encodes positional information.
    
    Benefits:
    - No need for explicit positional encoding
    - Position-awareness through convolution
    - Better local feature mixing
    """
    
    def __init__(
        self,
        embed_dim: int,
        ffn_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ffn_ratio = ffn_ratio
        
        hidden_dim = int(embed_dim * ffn_ratio)
        
        # Expand
        self.fc1 = layers.Dense(hidden_dim, name="fc1")
        
        # 3×3 depthwise conv (provides positional info)
        self.dwconv = layers.DepthwiseConv2D(
            kernel_size=3,
            padding='same',
            name="dwconv"
        )
        
        # Contract
        self.fc2 = layers.Dense(embed_dim, name="fc2")
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(
        self,
        x: tf.Tensor,
        grid_size: Tuple[int, int],
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            x: Input (B, N, D) where N = H*W
            grid_size: (H, W) spatial dimensions
            
        Returns:
            Output (B, N, D)
        """
        B = tf.shape(x)[0]
        H, W = grid_size
        
        # Expand
        x = self.fc1(x)
        x = tf.nn.gelu(x)
        
        # Reshape to spatial for depthwise conv
        hidden_dim = x.shape[-1]
        x = tf.reshape(x, [B, H, W, hidden_dim])
        
        # Depthwise conv (adds position info)
        x = self.dwconv(x)
        x = tf.nn.gelu(x)
        
        # Flatten back
        x = tf.reshape(x, [B, H * W, hidden_dim])
        
        # Contract
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "ffn_ratio": self.ffn_ratio,
        })
        return config
