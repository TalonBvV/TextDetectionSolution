"""
Advanced prediction heads for PatchFormer v2.0.

Contains:
- BoundaryRefinementHead: Differentiable binarization (DBNet-inspired)
- PatchAffinityHead: Patch-to-patch affinity prediction (CRAFT-inspired)
- PolygonVertexHead: Polygon vertex regression for curved text
- AffinityGuidedRefiner: Feature propagation using affinity
- VertexBoundaryRefiner: Boundary sharpening near polygon vertices
- CascadedRefinementModule: Unified pipeline combining all refinements
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, Dict, Optional

from .layers import DepthwiseSeparableConv


class BoundaryRefinementHead(layers.Layer):
    """
    Differentiable Binarization head inspired by DBNet.
    
    Predicts:
    - probability_map: Soft text probability
    - threshold_map: Adaptive threshold per pixel (learned)
    - binary_map: Sharp output via differentiable step function
    
    The key insight is: binary = sigmoid(k * (prob - thresh))
    This creates sharp boundaries while remaining differentiable.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        k: float = 50.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.k = k  # Amplification factor for sharpness
        
        # Shared feature extraction
        self.shared_conv = DepthwiseSeparableConv(
            filters=hidden_dim,
            kernel_size=3,
            name="shared_conv"
        )
        
        # Probability branch
        self.prob_conv1 = DepthwiseSeparableConv(
            filters=hidden_dim // 2,
            kernel_size=3,
            name="prob_conv1"
        )
        self.prob_conv2 = layers.Conv2D(
            filters=1,
            kernel_size=1,
            activation='sigmoid',
            name="prob_head"
        )
        
        # Threshold branch (separate pathway)
        self.thresh_conv1 = DepthwiseSeparableConv(
            filters=hidden_dim // 2,
            kernel_size=3,
            name="thresh_conv1"
        )
        self.thresh_conv2 = DepthwiseSeparableConv(
            filters=hidden_dim // 4,
            kernel_size=3,
            name="thresh_conv2"
        )
        self.thresh_head = layers.Conv2D(
            filters=1,
            kernel_size=1,
            activation='sigmoid',
            name="thresh_head"
        )
    
    def call(
        self, 
        features: tf.Tensor, 
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Args:
            features: Decoder features (B, H, W, C)
            
        Returns:
            Dict with probability, threshold, and binary maps
        """
        # Shared features
        shared = self.shared_conv(features, training=training)
        
        # Probability map
        prob_feat = self.prob_conv1(shared, training=training)
        prob_map = self.prob_conv2(prob_feat)
        
        # Threshold map
        thresh_feat = self.thresh_conv1(shared, training=training)
        thresh_feat = self.thresh_conv2(thresh_feat, training=training)
        thresh_map = self.thresh_head(thresh_feat)
        
        # Differentiable binarization
        # binary = 1 / (1 + exp(-k * (prob - thresh)))
        binary_map = tf.sigmoid(self.k * (prob_map - thresh_map))
        
        return {
            'probability': prob_map,      # (B, H, W, 1)
            'threshold': thresh_map,      # (B, H, W, 1)
            'binary': binary_map,         # (B, H, W, 1)
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "k": self.k,
        })
        return config


class PatchAffinityHead(layers.Layer):
    """
    Predicts affinity between adjacent patches for instance linking.
    
    Inspired by CRAFT's character affinity, but operates on patches.
    Predicts 8-directional affinity: N, NE, E, SE, S, SW, W, NW
    
    High affinity indicates patches belong to same text instance.
    This enables post-processing to link patches into word/line instances.
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 8
        
        # Direction offsets: N, NE, E, SE, S, SW, W, NW
        self.direction_offsets = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        # Feature projection
        self.query_proj = layers.Dense(hidden_dim, name="query_proj")
        self.key_proj = layers.Dense(hidden_dim, name="key_proj")
        
        # Learnable direction embeddings
        self.direction_dense = layers.Dense(hidden_dim, name="direction_dense")
        
        # Output head
        self.output_norm = layers.LayerNormalization(epsilon=1e-6)
        self.output_dense = layers.Dense(1, name="affinity_output")
    
    def build(self, input_shape):
        # Learnable direction embeddings
        self.direction_embeddings = self.add_weight(
            name="direction_embeddings",
            shape=(self.num_directions, self.hidden_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)
    
    def call(
        self, 
        patch_embeddings: tf.Tensor,
        grid_size: Tuple[int, int],
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            patch_embeddings: (B, H*W, D) patch features
            grid_size: (H, W) spatial dimensions
            
        Returns:
            affinity: (B, H, W, 8) affinity scores to 8 neighbors
        """
        B = tf.shape(patch_embeddings)[0]
        H, W = grid_size
        
        # Reshape to spatial grid
        features = tf.reshape(patch_embeddings, [B, H, W, -1])
        
        # Project to query/key space
        queries = self.query_proj(features)  # (B, H, W, hidden_dim)
        keys = self.key_proj(features)       # (B, H, W, hidden_dim)
        
        # Compute affinity for each direction
        affinities = []
        
        for i, (dy, dx) in enumerate(self.direction_offsets):
            # Shift keys to get neighbor features
            shifted_keys = tf.roll(keys, shift=[dy, dx], axis=[1, 2])
            
            # Direction-aware similarity
            # q * (k + direction_embedding)
            dir_embed = self.direction_embeddings[i]  # (hidden_dim,)
            dir_embed = tf.reshape(dir_embed, [1, 1, 1, self.hidden_dim])
            
            # Compute similarity with direction bias
            combined = shifted_keys + dir_embed
            similarity = tf.reduce_sum(queries * combined, axis=-1, keepdims=True)
            
            # Normalize by sqrt(d)
            similarity = similarity / tf.sqrt(tf.cast(self.hidden_dim, tf.float32))
            
            affinities.append(similarity)
        
        # Stack and apply sigmoid
        affinity = tf.concat(affinities, axis=-1)  # (B, H, W, 8)
        affinity = tf.sigmoid(affinity)
        
        return affinity
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
        })
        return config


class PolygonVertexHead(layers.Layer):
    """
    Predicts polygon vertices for arbitrary-shape text detection.
    
    For each patch, predicts:
    - is_vertex: Binary - is this patch near a polygon vertex?
    - vertex_offset: 2D offset from patch center to nearest vertex
    - vertex_confidence: Confidence in the prediction
    
    Post-processing connects high-confidence vertices to form polygons.
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        max_vertices: int = 16,  # Max vertices per polygon
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_vertices = max_vertices
        
        # Feature processing
        self.feature_conv = DepthwiseSeparableConv(
            filters=hidden_dim,
            kernel_size=3,
            name="feature_conv"
        )
        
        # Vertex classification (is this patch near a vertex?)
        self.vertex_cls = layers.Dense(2, name="vertex_cls")  # [not_vertex, is_vertex]
        
        # Vertex offset regression (x, y offset to nearest vertex)
        self.offset_hidden = layers.Dense(hidden_dim, activation='relu', name="offset_hidden")
        self.offset_reg = layers.Dense(2, name="offset_reg")  # (dx, dy)
        
        # Vertex order prediction (position in polygon sequence)
        self.order_hidden = layers.Dense(hidden_dim, activation='relu', name="order_hidden")
        self.order_reg = layers.Dense(1, activation='sigmoid', name="order_reg")  # 0-1 position
    
    def call(
        self, 
        features: tf.Tensor,
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Args:
            features: (B, H, W, C) spatial features
            
        Returns:
            Dict with vertex predictions
        """
        # Process features
        feat = self.feature_conv(features, training=training)
        
        # Flatten for dense layers
        B, H, W, C = tf.shape(feat)[0], tf.shape(feat)[1], tf.shape(feat)[2], feat.shape[-1]
        feat_flat = tf.reshape(feat, [B, H * W, C])
        
        # Vertex classification
        vertex_logits = self.vertex_cls(feat_flat)  # (B, H*W, 2)
        vertex_prob = tf.nn.softmax(vertex_logits, axis=-1)[:, :, 1]  # (B, H*W)
        
        # Offset regression
        offset_feat = self.offset_hidden(feat_flat)
        vertex_offset = self.offset_reg(offset_feat)  # (B, H*W, 2)
        
        # Order prediction
        order_feat = self.order_hidden(feat_flat)
        vertex_order = self.order_reg(order_feat)  # (B, H*W, 1)
        
        # Reshape back to spatial
        vertex_prob = tf.reshape(vertex_prob, [B, H, W, 1])
        vertex_offset = tf.reshape(vertex_offset, [B, H, W, 2])
        vertex_order = tf.reshape(vertex_order, [B, H, W, 1])
        
        return {
            'vertex_logits': tf.reshape(vertex_logits, [B, H, W, 2]),
            'vertex_prob': vertex_prob,
            'vertex_offset': vertex_offset,
            'vertex_order': vertex_order,
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "max_vertices": self.max_vertices,
        })
        return config


class BoundaryAwareCrossAttention(layers.Layer):
    """
    Cross-attention that emphasizes boundary regions.
    
    Uses token classification (boundary probability) to bias attention
    so that boundary patches attend more to other boundary patches.
    This improves boundary precision in the final segmentation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        boundary_weight: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.boundary_weight = boundary_weight
        
        self.q_proj = layers.Dense(embed_dim, name="q_proj")
        self.k_proj = layers.Dense(embed_dim, name="k_proj")
        self.v_proj = layers.Dense(embed_dim, name="v_proj")
        self.out_proj = layers.Dense(embed_dim, name="out_proj")
        
        self.dropout = layers.Dropout(dropout_rate)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(
        self,
        x: tf.Tensor,
        token_logits: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            x: Input embeddings (B, N, D)
            token_logits: Token classification logits (B, N, 3) 
                          where class 1 is boundary
            
        Returns:
            Boundary-aware attended features (B, N, D)
        """
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        
        # Get boundary probabilities
        token_probs = tf.nn.softmax(token_logits, axis=-1)
        boundary_probs = token_probs[:, :, 1]  # (B, N) - boundary class
        
        # Compute boundary attention bias
        # boundary_i * boundary_j creates high bias for boundary-boundary attention
        boundary_bias = tf.expand_dims(boundary_probs, -1) * tf.expand_dims(boundary_probs, -2)
        boundary_bias = boundary_bias * self.boundary_weight  # (B, N, N)
        
        # Standard multi-head attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = tf.reshape(q, [B, N, self.num_heads, self.head_dim])
        k = tf.reshape(k, [B, N, self.num_heads, self.head_dim])
        v = tf.reshape(v, [B, N, self.num_heads, self.head_dim])
        
        # Transpose to (B, heads, N, head_dim)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Attention with boundary bias
        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attn = tf.matmul(q, k, transpose_b=True) / scale  # (B, heads, N, N)
        
        # Add boundary bias (broadcast over heads)
        attn = attn + tf.expand_dims(boundary_bias, 1)
        
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)
        
        # Apply attention
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N, self.embed_dim])
        
        # Output projection + residual
        out = self.out_proj(out)
        out = self.dropout(out, training=training)
        out = self.norm(x + out)
        
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "boundary_weight": self.boundary_weight,
        })
        return config


class MultiScaleTokenAggregator(layers.Layer):
    """
    Aggregates tokens at multiple scales for scale-invariant features.
    
    Creates a feature pyramid within the transformer:
    - Scale 1: 128×128 (original)
    - Scale 2: 64×64 (2x pooled)
    - Scale 3: 32×32 (4x pooled)
    
    Fine-resolution tokens attend to coarser scales for global context.
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
        
        # Token pooling projections
        self.pool_proj_2x = layers.Dense(embed_dim, name="pool_proj_2x")
        self.pool_proj_4x = layers.Dense(embed_dim, name="pool_proj_4x")
        
        # Cross-scale attention (fine attends to coarse)
        self.cross_attn_2x = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            name="cross_attn_2x"
        )
        self.cross_attn_4x = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            name="cross_attn_4x"
        )
        
        # Feature fusion
        self.fusion_proj = layers.Dense(embed_dim, name="fusion_proj")
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.output_norm = layers.LayerNormalization(epsilon=1e-6, name="output_norm")
        
        # FFN after fusion
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ], name="ffn")
    
    def call(
        self,
        x: tf.Tensor,
        grid_size: Tuple[int, int],
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            x: (B, H*W, D) tokens at full resolution
            grid_size: (H, W) spatial dimensions
            
        Returns:
            Multi-scale aggregated features (B, H*W, D)
        """
        B = tf.shape(x)[0]
        H, W = grid_size
        D = self.embed_dim
        
        # Reshape to spatial for pooling
        x_2d = tf.reshape(x, [B, H, W, D])
        
        # Create multi-scale features via average pooling
        # Scale 2: 64×64
        x_2d_2x = tf.nn.avg_pool2d(x_2d, ksize=2, strides=2, padding='VALID')
        feat_2x = self.pool_proj_2x(x_2d_2x)
        feat_2x_flat = tf.reshape(feat_2x, [B, H * W // 4, D])
        
        # Scale 3: 32×32
        x_2d_4x = tf.nn.avg_pool2d(x_2d, ksize=4, strides=4, padding='VALID')
        feat_4x = self.pool_proj_4x(x_2d_4x)
        feat_4x_flat = tf.reshape(feat_4x, [B, H * W // 16, D])
        
        # Cross-scale attention: fine queries attend to coarse keys/values
        # This gives each fine token access to broader context
        
        # Attention to 2x scale
        attn_2x = self.cross_attn_2x(
            query=x,
            key=feat_2x_flat,
            value=feat_2x_flat,
            training=training
        )
        x = self.norm1(x + attn_2x)
        
        # Attention to 4x scale
        attn_4x = self.cross_attn_4x(
            query=x,
            key=feat_4x_flat,
            value=feat_4x_flat,
            training=training
        )
        x = self.norm2(x + attn_4x)
        
        # FFN
        x = self.output_norm(x + self.ffn(x, training=training))
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config


# =============================================================================
# v2.0 UNIFIED REFINEMENT PIPELINE
# All components feed into final segmentation output
# =============================================================================

class AffinityGuidedRefiner(layers.Layer):
    """
    Uses affinity predictions to propagate features between related regions.
    
    High affinity between patches means they should share features during
    refinement, leading to more coherent text regions and sharper boundaries.
    
    This directly improves final segmentation by:
    1. Propagating strong text features to weaker neighboring text patches
    2. Preventing feature bleeding across text/background boundaries
    3. Creating smoother, more coherent segmentation within text instances
    
    FIX: Uses proper padding instead of tf.roll to avoid edge wrap-around artifacts.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        propagation_steps: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.propagation_steps = propagation_steps
        
        # Feature projection
        self.input_proj = layers.Conv2D(hidden_dim, 3, padding='same', name="input_proj")
        
        # Affinity-weighted aggregation
        self.agg_conv = layers.Conv2D(hidden_dim, 3, padding='same', name="agg_conv")
        
        # Gating to control how much aggregated features affect output
        # FIX: Input is 2*hidden_dim (x + aggregated concatenated)
        self.gate_conv = layers.Conv2D(hidden_dim, 1, activation='sigmoid', name="gate")
        
        # Output projection
        self.output_proj = layers.Conv2D(hidden_dim, 1, name="output_proj")
        
        # Direction offsets for 8-neighbor aggregation: (dy, dx)
        # Order: N, NE, E, SE, S, SW, W, NW
        self.direction_offsets = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
    
    def _get_neighbor_features(self, x: tf.Tensor, dy: int, dx: int) -> tf.Tensor:
        """
        Get neighbor features using padding instead of roll.
        
        This avoids edge wrap-around artifacts where pixels at image boundaries
        incorrectly receive features from the opposite side.
        
        Args:
            x: Input features (B, H, W, C)
            dy: Vertical offset (-1, 0, or 1)
            dx: Horizontal offset (-1, 0, or 1)
            
        Returns:
            Shifted features with zero-padding at boundaries (B, H, W, C)
        """
        # Pad with zeros: [[batch], [top, bottom], [left, right], [channels]]
        # We pad 1 on each side, then slice to get the shifted version
        pad_top = max(0, -dy)
        pad_bottom = max(0, dy)
        pad_left = max(0, -dx)
        pad_right = max(0, dx)
        
        # Pad the tensor
        padded = tf.pad(x, [
            [0, 0],  # batch
            [pad_top, pad_bottom],  # height
            [pad_left, pad_right],  # width
            [0, 0]   # channels
        ], mode='CONSTANT', constant_values=0.0)
        
        # Slice to get neighbor values
        # If dy=-1 (looking up), we want padded[0:H, :] which shifts content down
        # If dy=1 (looking down), we want padded[2:H+2, :] which shifts content up
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        
        start_h = pad_bottom  # If we padded bottom, start from 0; if padded top, skip the padding
        start_w = pad_right
        
        neighbor = padded[:, start_h:start_h+H, start_w:start_w+W, :]
        
        return neighbor
    
    def call(
        self,
        features: tf.Tensor,
        affinity: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            features: Decoder features (B, H, W, C)
            affinity: 8-directional affinity (B, H, W, 8), values in [0, 1]
            
        Returns:
            Refined features (B, H, W, C)
        """
        # Project features
        x = self.input_proj(features)
        original_x = x
        
        # Iterative affinity-guided propagation
        for step in range(self.propagation_steps):
            # Aggregate features from neighbors weighted by affinity
            aggregated = tf.zeros_like(x)
            
            for i, (dy, dx) in enumerate(self.direction_offsets):
                # FIX: Use padding-based shift instead of roll
                neighbor_features = self._get_neighbor_features(x, dy, dx)
                
                # Weight by affinity in this direction
                affinity_weight = affinity[:, :, :, i:i+1]  # (B, H, W, 1)
                
                # Accumulate weighted neighbor features
                aggregated = aggregated + neighbor_features * affinity_weight
            
            # Normalize by sum of affinities
            affinity_sum = tf.reduce_sum(affinity, axis=-1, keepdims=True) + 1e-6
            aggregated = aggregated / affinity_sum
            
            # Process aggregated features
            aggregated = self.agg_conv(aggregated)
            
            # Gate: control how much to use aggregated vs original
            gate = self.gate_conv(tf.concat([x, aggregated], axis=-1))
            x = x * (1 - gate) + aggregated * gate
        
        # Output projection with residual
        output = self.output_proj(x) + original_x
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "propagation_steps": self.propagation_steps,
        })
        return config


class VertexBoundaryRefiner(layers.Layer):
    """
    Uses vertex predictions to sharpen boundaries near polygon corners.
    
    Curved text has critical vertices at polygon corners. Extra refinement
    at these points improves polygon boundary accuracy.
    
    This directly improves final segmentation by:
    1. Identifying critical boundary points (polygon vertices)
    2. Applying extra sharpening/refinement near these points
    3. Using vertex offsets to guide boundary direction
    
    FIX: Properly handles input channel dimensions for direction_conv.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        input_dim: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # Input projection to ensure consistent dimensions
        self.input_proj = layers.Conv2D(hidden_dim, 1, padding='same', name="input_proj")
        
        # Vertex-aware feature processing
        self.vertex_conv = layers.Conv2D(hidden_dim, 3, padding='same', name="vertex_conv")
        
        # Boundary sharpening convolutions
        self.sharpen_conv1 = layers.Conv2D(hidden_dim, 3, padding='same', activation='relu', name="sharpen1")
        self.sharpen_conv2 = layers.Conv2D(hidden_dim, 3, padding='same', name="sharpen2")
        
        # FIX: Offset-guided directional refinement - explicitly handle hidden_dim + 2 input channels
        # Takes concatenation of features (hidden_dim) + normalized offset (2)
        self.direction_proj = layers.Conv2D(hidden_dim, 1, padding='same', name="direction_proj")
        self.direction_conv = layers.Conv2D(hidden_dim, 3, padding='same', name="direction_conv")
        
        # Gating - takes features for context
        self.gate = layers.Conv2D(1, 1, activation='sigmoid', name="gate")
        
        # Output projection back to input dimension
        self.output_conv = layers.Conv2D(input_dim, 1, name="output")
    
    def call(
        self,
        features: tf.Tensor,
        vertex_prob: tf.Tensor,
        vertex_offset: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            features: Input features (B, H, W, C)
            vertex_prob: Vertex probability (B, H, W, 1)
            vertex_offset: Offset to nearest vertex (B, H, W, 2)
            
        Returns:
            Refined features (B, H, W, C)
        """
        # Project input to hidden dimension
        x = self.input_proj(features)
        
        # Create vertex-weighted features
        # Higher weight = more refinement near vertices
        vertex_weight = vertex_prob  # (B, H, W, 1)
        
        # Process features with vertex awareness
        vertex_features = self.vertex_conv(x * (1 + vertex_weight))
        
        # Boundary sharpening (applied more strongly near vertices)
        sharpened = self.sharpen_conv1(vertex_features)
        sharpened = self.sharpen_conv2(sharpened)
        
        # Use vertex offset to guide directional refinement
        # Offset tells us which direction the boundary goes
        offset_magnitude = tf.sqrt(
            vertex_offset[:, :, :, 0:1]**2 + vertex_offset[:, :, :, 1:2]**2 + 1e-6
        )
        offset_normalized = vertex_offset / (offset_magnitude + 1e-6)
        
        # FIX: Properly project concatenated features before direction conv
        # Concatenate features + offset: (hidden_dim + 2) channels
        direction_input = tf.concat([x, offset_normalized], axis=-1)
        direction_input = self.direction_proj(direction_input)  # Project to hidden_dim
        direction_features = self.direction_conv(direction_input)
        
        # Combine sharpened and direction-guided features
        combined = sharpened + direction_features * vertex_weight
        
        # Gate: apply refinement proportional to vertex probability
        gate = self.gate(tf.concat([x, vertex_prob], axis=-1))
        refined = x * (1 - gate) + combined * gate
        
        # Output projection back to input dimension
        output = self.output_conv(refined)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim,
        })
        return config


class CascadedRefinementModule(layers.Layer):
    """
    Unified refinement pipeline that combines all auxiliary predictions
    to produce the final segmentation output.
    
    Pipeline:
    1. Initial decoder features
    2. Compute affinity → Affinity-guided feature propagation
    3. Compute vertices → Vertex-aware boundary refinement
    4. DB head → Sharp binary output
    
    ALL components feed into the SINGLE final output.
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        num_classes: int = 2,
        use_affinity: bool = True,
        use_vertex: bool = True,
        use_db: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_affinity = use_affinity
        self.use_vertex = use_vertex
        self.use_db = use_db
        
        # Initial segmentation (for intermediate supervision)
        self.initial_seg_conv = layers.Conv2D(
            num_classes, 1, padding='same', name="initial_seg"
        )
        
        # Affinity computation and refinement
        if use_affinity:
            self.affinity_proj = layers.Dense(128, name="affinity_proj")
            self.affinity_head = self._build_affinity_head()
            self.affinity_refiner = AffinityGuidedRefiner(
                hidden_dim=feature_dim,
                propagation_steps=2,
                name="affinity_refiner"
            )
        
        # Vertex computation and refinement
        if use_vertex:
            self.vertex_conv = DepthwiseSeparableConv(feature_dim, 3, name="vertex_conv")
            self.vertex_cls = layers.Dense(2, name="vertex_cls")
            self.vertex_offset = layers.Dense(2, name="vertex_offset")
            self.vertex_refiner = VertexBoundaryRefiner(
                hidden_dim=feature_dim,
                input_dim=feature_dim,  # FIX: Pass input_dim for proper output dimensions
                name="vertex_refiner"
            )
        
        # DB head for final sharpening
        if use_db:
            self.db_head = BoundaryRefinementHead(
                hidden_dim=feature_dim,
                k=50.0,
                name="db_head"
            )
        
        # Final output (if not using DB)
        self.final_conv = layers.Conv2D(num_classes, 1, padding='same', name="final_seg")
    
    def _build_affinity_head(self):
        """Build lightweight affinity prediction from spatial features."""
        return tf.keras.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(8, 1, activation='sigmoid'),  # 8 directions
        ], name="affinity_head")
    
    def call(
        self,
        features: tf.Tensor,
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Args:
            features: Decoder features (B, H, W, C)
            
        Returns:
            Dict with:
            - 'final': Final refined segmentation (MAIN OUTPUT)
            - 'initial_seg': Initial segmentation (for aux loss)
            - 'affinity': Affinity maps (for aux loss, if enabled)
            - 'vertex_prob': Vertex probability (for aux loss, if enabled)
            - 'db_outputs': DB outputs (for aux loss, if enabled)
        """
        outputs = {}
        x = features
        
        # Stage 1: Initial segmentation (for intermediate supervision)
        initial_seg = self.initial_seg_conv(x)
        outputs['initial_seg'] = initial_seg
        
        # Stage 2: Affinity-guided refinement
        if self.use_affinity:
            # Compute affinity
            affinity = self.affinity_head(x)  # (B, H, W, 8)
            outputs['affinity'] = affinity
            
            # Refine features using affinity
            x = self.affinity_refiner(x, affinity, training=training)
        
        # Stage 3: Vertex-aware boundary refinement
        if self.use_vertex:
            # Compute vertex predictions
            vertex_features = self.vertex_conv(x, training=training)
            B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
            
            vertex_flat = tf.reshape(vertex_features, [B, H * W, -1])
            vertex_logits = self.vertex_cls(vertex_flat)
            vertex_prob = tf.nn.softmax(vertex_logits, axis=-1)[:, :, 1]  # P(vertex)
            vertex_prob = tf.reshape(vertex_prob, [B, H, W, 1])
            
            vertex_off = self.vertex_offset(vertex_flat)
            vertex_off = tf.reshape(vertex_off, [B, H, W, 2])
            
            outputs['vertex_prob'] = vertex_prob
            outputs['vertex_offset'] = vertex_off
            outputs['vertex_logits'] = tf.reshape(vertex_logits, [B, H, W, 2])
            
            # Refine features using vertex information
            x = self.vertex_refiner(x, vertex_prob, vertex_off, training=training)
        
        # Stage 4: DB head for final sharpening
        if self.use_db:
            db_outputs = self.db_head(x, training=training)
            outputs['db_outputs'] = db_outputs
            outputs['probability'] = db_outputs['probability']
            outputs['threshold'] = db_outputs['threshold']
            outputs['binary'] = db_outputs['binary']
            
            # Final output is the sharp binary map
            outputs['final'] = db_outputs['binary']
        else:
            # Without DB, use standard segmentation
            final_seg = self.final_conv(x)
            outputs['final'] = tf.nn.softmax(final_seg, axis=-1)[:, :, :, 1:2]
        
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "use_affinity": self.use_affinity,
            "use_vertex": self.use_vertex,
            "use_db": self.use_db,
        })
        return config
