"""
Dual-stream encoder for PatchFormer.

Combines explicit patch features with learned convolutional features.
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, Optional, Tuple, List

from .config import PatchFormerConfig
from .layers import (
    PatchExtractor,
    PatchEmbedding,
    Learned2DPositionalEncoding,
    DepthwiseSeparableConv,
    ConvBlock,
    TokenClassificationHead,
    ContextHead,
    TokenClassificationHead,
    SegmentationHead,
    GatingMechanism,
    # v2.1 additions
    OverlappingPatchEmbed,
    DeformableConv2D,
    AdaptiveScaleFusion,
    MixFFN,
)
from .transformer import (
    TransformerBlock,
    TransformerStack,
    LightweightTransformerBlock,
    AdaptiveTransformerBlock,
)
from .heads import (
    BoundaryAwareCrossAttention,
    MultiScaleTokenAggregator,
)


class ConvolutionalEncoder(layers.Layer):
    """
    Convolutional stream of the dual-stream encoder.
    
    Progressively downsamples the input image while learning
    multi-scale features.
    
    1024x1024x3 -> 512x512x64 -> 256x256x128 -> 128x128x256 -> 128x128x320
    """
    
    def __init__(
        self,
        output_dim: int = 320,
        use_squeeze_excitation: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        
        # Stem: initial convolution
        self.stem = tf.keras.Sequential([
            layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("gelu"),
        ], name="stem")
        
        # Downsampling blocks
        self.block1 = ConvBlock(
            filters=64, strides=2, 
            use_squeeze_excitation=use_squeeze_excitation,
            name="block1"
        )  # -> 512x512x64
        
        self.block2 = ConvBlock(
            filters=128, strides=2,
            use_squeeze_excitation=use_squeeze_excitation,
            name="block2"
        )  # -> 256x256x128
        
        self.block3 = ConvBlock(
            filters=256, strides=2,
            use_squeeze_excitation=use_squeeze_excitation,
            name="block3"
        )  # -> 128x128x256
        
        self.block4 = ConvBlock(
            filters=output_dim, strides=1,
            use_squeeze_excitation=use_squeeze_excitation,
            name="block4"
        )  # -> 128x128x320
    
    def call(
        self, 
        x: tf.Tensor, 
        training: bool = False,
        return_intermediates: bool = False
    ) -> tf.Tensor:
        """
        Args:
            x: Input image (B, H, W, 3)
            training: Whether in training mode
            return_intermediates: If True, return intermediate features for skip connections
            
        Returns:
            features: (B, H/8, W/8, output_dim) or dict with intermediates
        """
        intermediates = {}
        
        x = self.stem(x, training=training)
        
        x = self.block1(x, training=training)
        intermediates['block1'] = x  # 512x512x64
        
        x = self.block2(x, training=training)
        intermediates['block2'] = x  # 256x256x128
        
        x = self.block3(x, training=training)
        intermediates['block3'] = x  # 128x128x256
        
        x = self.block4(x, training=training)
        intermediates['block4'] = x  # 128x128x320
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config


class ConvolutionalEncoderV2(layers.Layer):
    """
    v2.1 Convolutional encoder with deformable convolutions.
    
    Uses DeformableConv2D in later stages to adapt receptive field
    to curved/rotated text shapes.
    """
    
    def __init__(
        self,
        output_dim: int = 320,
        use_squeeze_excitation: bool = True,
        use_deformable: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.use_deformable = use_deformable
        
        # Stem: initial convolution
        self.stem = tf.keras.Sequential([
            layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("gelu"),
        ], name="stem")
        
        # Downsampling blocks (standard conv for early stages)
        self.block1 = ConvBlock(
            filters=64, strides=2, 
            use_squeeze_excitation=use_squeeze_excitation,
            name="block1"
        )  # -> 512x512x64
        
        self.block2 = ConvBlock(
            filters=128, strides=2,
            use_squeeze_excitation=use_squeeze_excitation,
            name="block2"
        )  # -> 256x256x128
        
        # v2.1: Use deformable convolutions in later stages
        if use_deformable:
            self.block3_deform = DeformableConv2D(
                filters=256,
                kernel_size=3,
                use_modulation=True,
                name="block3_deform"
            )
            self.block3_down = layers.Conv2D(
                256, 3, strides=2, padding='same', name="block3_down"
            )
            self.block3_norm = layers.BatchNormalization(name="block3_norm")
            
            self.block4_deform = DeformableConv2D(
                filters=output_dim,
                kernel_size=3,
                use_modulation=True,
                name="block4_deform"
            )
            self.block4_norm = layers.BatchNormalization(name="block4_norm")
        else:
            self.block3 = ConvBlock(
                filters=256, strides=2,
                use_squeeze_excitation=use_squeeze_excitation,
                name="block3"
            )
            self.block4 = ConvBlock(
                filters=output_dim, strides=1,
                use_squeeze_excitation=use_squeeze_excitation,
                name="block4"
            )
    
    def call(
        self, 
        x: tf.Tensor, 
        training: bool = False,
        return_intermediates: bool = False
    ) -> tf.Tensor:
        """
        Args:
            x: Input image (B, H, W, 3)
            training: Whether in training mode
            return_intermediates: If True, return intermediate features
            
        Returns:
            features: (B, H/8, W/8, output_dim) or dict with intermediates
        """
        intermediates = {}
        
        x = self.stem(x, training=training)
        
        x = self.block1(x, training=training)
        intermediates['block1'] = x  # 512x512x64
        
        x = self.block2(x, training=training)
        intermediates['block2'] = x  # 256x256x128
        
        if self.use_deformable:
            # Deformable conv + downsample
            x = self.block3_down(x)
            x = self.block3_deform(x, training=training)
            x = self.block3_norm(x, training=training)
            x = tf.nn.gelu(x)
            intermediates['block3'] = x  # 128x128x256
            
            x = self.block4_deform(x, training=training)
            x = self.block4_norm(x, training=training)
            x = tf.nn.gelu(x)
            intermediates['block4'] = x  # 128x128x320
        else:
            x = self.block3(x, training=training)
            intermediates['block3'] = x
            
            x = self.block4(x, training=training)
            intermediates['block4'] = x
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "use_deformable": self.use_deformable,
        })
        return config


class DualStreamEncoderV2(layers.Layer):
    """
    v2.1 Dual-stream encoder with:
    - Overlapping patch embeddings (from SegFormer)
    - Deformable convolutions (from DBNet++)
    
    Eliminates boundary artifacts and adapts to text shapes.
    """
    
    def __init__(self, config: PatchFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # v2.1: Overlapping patch embedding (replaces PatchExtractor)
        self.patch_embed = OverlappingPatchEmbed(
            embed_dim=config.patch_dim,
            patch_size=config.patch_size[0],
            stride=config.patch_size[0] // 2,  # 50% overlap
            name="patch_embed"
        )
        
        # v2.1: Deformable convolutional stream
        self.conv_encoder = ConvolutionalEncoderV2(
            output_dim=config.conv_dim,
            use_squeeze_excitation=config.use_squeeze_excitation,
            use_deformable=True,
            name="conv_encoder"
        )
        
        # Projection to align dimensions before fusion
        self.patch_proj = layers.Dense(config.patch_dim, name="patch_proj")
        self.conv_proj = layers.Dense(config.conv_dim, name="conv_proj")
        
        # Positional encoding (can be optional with MixFFN)
        self.positional_encoding = Learned2DPositionalEncoding(
            height=config.grid_size[0],
            width=config.grid_size[1],
            embed_dim=config.embed_dim,
            name="positional_encoding"
        )
        
        # Initial transformer block
        self.initial_transformer = TransformerStack(
            num_layers=config.initial_transformer_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ffn_ratio=config.ffn_ratio,
            dropout_rate=config.dropout_rate,
            use_windowed_attention=config.use_windowed_attention,
            window_size=config.window_size,
            grid_size=config.grid_size,
            name="initial_transformer"
        )
    
    def call(
        self, 
        images: tf.Tensor, 
        training: bool = False,
        return_conv_intermediates: bool = False
    ) -> Tuple[tf.Tensor, Optional[Dict]]:
        """
        Args:
            images: Input images (B, H, W, 3)
            training: Whether in training mode
            return_conv_intermediates: Whether to return conv intermediates
            
        Returns:
            encoded: Encoded features (B, num_patches, embed_dim)
            conv_intermediates: Optional dict of intermediate conv features
        """
        batch_size = tf.shape(images)[0]
        
        # Stream 1: Overlapping patch embedding
        patch_features, patch_grid = self.patch_embed(images)  # (B, N, patch_dim)
        
        # Adjust grid size if different from expected due to overlap
        # Resize patch features to match expected grid
        H, W = self.config.grid_size
        patch_features_2d = tf.reshape(
            patch_features, 
            [batch_size, patch_grid[0], patch_grid[1], self.config.patch_dim]
        )
        if patch_grid[0] != H or patch_grid[1] != W:
            patch_features_2d = tf.image.resize(patch_features_2d, [H, W])
        
        # Stream 2: Deformable convolutional encoding
        if return_conv_intermediates:
            conv_features, conv_intermediates = self.conv_encoder(
                images, training=training, return_intermediates=True
            )
        else:
            conv_features = self.conv_encoder(images, training=training)
            conv_intermediates = None
        
        # Fuse streams: concatenate along channel dimension
        fused = tf.concat([patch_features_2d, conv_features], axis=-1)  # (B, H, W, embed_dim)
        
        # Add positional encoding
        fused = self.positional_encoding(fused)
        
        # Reshape to sequence
        fused = tf.reshape(fused, [batch_size, self.config.num_patches, self.config.embed_dim])
        
        # Initial transformer processing
        encoded = self.initial_transformer(fused, training=training)
        
        return encoded, conv_intermediates
    
    def get_config(self):
        return super().get_config()


class DualStreamEncoder(layers.Layer):
    """
    Dual-stream encoder combining patch and convolutional features.
    
    Stream 1: Explicit patch extraction and flattening
    Stream 2: Learned convolutional features
    
    Both streams are fused and positional encoding is added.
    """
    
    def __init__(self, config: PatchFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Patch stream
        self.patch_extractor = PatchExtractor(
            patch_size=config.patch_size,
            name="patch_extractor"
        )
        
        self.patch_embedding = PatchEmbedding(
            embed_dim=config.patch_dim,
            patch_dim=config.patch_size[0] * config.patch_size[1] * 3,
            grid_size=config.grid_size,
            use_projection=False,  # Keep raw patch features
            name="patch_embedding"
        )
        
        # Convolutional stream
        self.conv_encoder = ConvolutionalEncoder(
            output_dim=config.conv_dim,
            use_squeeze_excitation=config.use_squeeze_excitation,
            name="conv_encoder"
        )
        
        # Positional encoding
        self.positional_encoding = Learned2DPositionalEncoding(
            height=config.grid_size[0],
            width=config.grid_size[1],
            embed_dim=config.embed_dim,
            name="positional_encoding"
        )
        
        # Initial transformer block
        self.initial_transformer = TransformerStack(
            num_layers=config.initial_transformer_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ffn_ratio=config.ffn_ratio,
            dropout_rate=config.dropout_rate,
            use_windowed_attention=config.use_windowed_attention,
            window_size=config.window_size,
            grid_size=config.grid_size,
            name="initial_transformer"
        )
    
    def call(
        self, 
        images: tf.Tensor, 
        training: bool = False,
        return_conv_intermediates: bool = False
    ) -> Tuple[tf.Tensor, Optional[Dict]]:
        """
        Args:
            images: Input images (B, H, W, 3)
            training: Whether in training mode
            return_conv_intermediates: Whether to return conv intermediates for skip connections
            
        Returns:
            encoded: Encoded features (B, num_patches, embed_dim)
            conv_intermediates: Optional dict of intermediate conv features
        """
        # Stream 1: Patch extraction
        patches = self.patch_extractor(images)  # (B, N, patch_dim)
        patch_features = self.patch_embedding(patches, return_spatial=True)  # (B, H, W, patch_dim)
        
        # Stream 2: Convolutional encoding
        if return_conv_intermediates:
            conv_features, conv_intermediates = self.conv_encoder(
                images, training=training, return_intermediates=True
            )
        else:
            conv_features = self.conv_encoder(images, training=training)
            conv_intermediates = None
        
        # Fuse streams: concatenate along channel dimension
        fused = tf.concat([patch_features, conv_features], axis=-1)  # (B, H, W, embed_dim)
        
        # Add positional encoding
        fused = self.positional_encoding(fused)  # (B, H, W, embed_dim)
        
        # Reshape to sequence
        batch_size = tf.shape(fused)[0]
        fused = tf.reshape(fused, [batch_size, self.config.num_patches, self.config.embed_dim])
        
        # Initial transformer processing
        encoded = self.initial_transformer(fused, training=training)
        
        return encoded, conv_intermediates
    
    def get_config(self):
        config = super().get_config()
        return config


class ReconstructionDecoder(layers.Layer):
    """
    Lightweight decoder for masked autoencoder reconstruction.
    
    Reconstructs the original image from encoded patches.
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        
        # Project to reconstruction dimension
        self.projection = layers.Dense(config.reconstruction_dim, name="projection")
        
        # Upsampling decoder
        self.upsample1 = tf.keras.Sequential([
            layers.Conv2DTranspose(192, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("gelu"),
        ], name="upsample1")  # 128 -> 256
        
        self.upsample2 = tf.keras.Sequential([
            layers.Conv2DTranspose(96, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("gelu"),
        ], name="upsample2")  # 256 -> 512
        
        self.upsample3 = tf.keras.Sequential([
            layers.Conv2DTranspose(48, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("gelu"),
        ], name="upsample3")  # 512 -> 1024
        
        # Final output layer
        self.output_conv = layers.Conv2D(3, 3, padding="same", name="output_conv")
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            x: Encoded features (B, N, D)
            
        Returns:
            Reconstructed image (B, H, W, 3)
        """
        batch_size = tf.shape(x)[0]
        
        # Project
        x = self.projection(x)
        
        # Reshape to spatial
        x = tf.reshape(
            x, 
            [batch_size, self.config.grid_size[0], self.config.grid_size[1], self.config.reconstruction_dim]
        )
        
        # Upsample
        x = self.upsample1(x, training=training)  # 256x256
        x = self.upsample2(x, training=training)  # 512x512
        x = self.upsample3(x, training=training)  # 1024x1024
        
        # Output
        x = self.output_conv(x)
        x = tf.sigmoid(x)  # Normalize to [0, 1]
        
        return x


class ProgressiveEncoder(layers.Layer):
    """
    Progressive encoding path with intermediate supervision.
    
    Takes the output of the initial transformer and progressively
    refines it through multiple stages with context generation,
    token classification, and final segmentation.
    
    v2.1 Additions:
    - Adaptive Scale Fusion (replaces static multi-scale aggregation)
    - Boundary-aware cross-attention for boundary precision
    
    FIX: Multi-scale fusion is now applied at MULTIPLE stages:
    1. Before Stage 2 (initial fusion)
    2. After Stage 2 (context-aware fusion)
    3. After Stage 3 (boundary-aware fusion)
    
    This ensures transformer stages benefit from multi-scale context throughout.
    
    Uses gating mechanisms instead of simple concatenation.
    Uses Swin-style attention throughout for cross-window communication.
    """
    
    def __init__(self, config: PatchFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # FIX: Multi-scale fusion at MULTIPLE stages
        # Stage 1 fusion (before context generation)
        self.scale_pool_2x = layers.AveragePooling2D(2, name="scale_pool_2x")
        self.scale_pool_4x = layers.AveragePooling2D(4, name="scale_pool_4x")
        self.scale_fusion_1 = AdaptiveScaleFusion(
            out_channels=config.embed_dim,
            num_scales=3,
            name="scale_fusion_1"
        )
        
        # Stage 2 fusion (after context generation)
        self.scale_fusion_2 = AdaptiveScaleFusion(
            out_channels=config.embed_dim,
            num_scales=3,
            name="scale_fusion_2"
        )
        
        # Stage 3 fusion (after token classification)
        self.scale_fusion_3 = AdaptiveScaleFusion(
            out_channels=config.embed_dim,
            num_scales=3,
            name="scale_fusion_3"
        )
        
        # Stage 2: Context generation with Swin attention
        self.stage2_transformer = TransformerStack(
            num_layers=config.stage2_transformer_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ffn_ratio=config.ffn_ratio,
            dropout_rate=config.dropout_rate,
            use_windowed_attention=True,
            window_size=config.window_size,
            grid_size=config.grid_size,
            use_swin=True,
            name="stage2_transformer"
        )
        self.context_head = ContextHead(
            output_dim=config.context_dim,
            name="context_head"
        )
        
        # Gating for context fusion
        self.context_gate = GatingMechanism(
            main_dim=config.embed_dim,
            aux_dim=config.context_dim,
            output_dim=config.embed_dim,
            name="context_gate"
        )
        
        # Stage 3: Token classification
        self.stage3_transformer = TransformerStack(
            num_layers=config.stage3_transformer_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ffn_ratio=config.ffn_ratio,
            dropout_rate=config.dropout_rate,
            use_windowed_attention=True,
            window_size=config.window_size,
            grid_size=config.grid_size,
            use_swin=True,
            name="stage3_transformer"
        )
        self.token_head = TokenClassificationHead(
            num_classes=config.token_classes,
            name="token_head"
        )
        
        # v2.0: Boundary-aware cross-attention (after token classification)
        self.boundary_attention = BoundaryAwareCrossAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            boundary_weight=2.0,
            name="boundary_attention"
        )
        
        # Gating for token classification fusion
        self.token_gate = GatingMechanism(
            main_dim=config.embed_dim,
            aux_dim=config.token_classes,
            output_dim=config.embed_dim,
            name="token_gate"
        )
        
        # Final stage: Segmentation with Swin attention
        self.final_transformer = TransformerStack(
            num_layers=config.final_transformer_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ffn_ratio=config.ffn_ratio,
            dropout_rate=config.dropout_rate,
            use_windowed_attention=True,
            window_size=config.window_size,
            grid_size=config.grid_size,
            use_swin=True,
            name="final_transformer"
        )
        self.segmentation_head = SegmentationHead(
            num_classes=config.num_classes,
            name="segmentation_head"
        )
        
        # Gating for final class logits fusion
        self.class_gate = GatingMechanism(
            main_dim=config.embed_dim,
            aux_dim=config.num_classes,
            output_dim=config.embed_dim,
            name="class_gate"
        )
    
    def _apply_multi_scale_fusion(
        self,
        x: tf.Tensor,
        fusion_module: AdaptiveScaleFusion,
        training: bool = False
    ) -> tf.Tensor:
        """
        Apply multi-scale fusion to features.
        
        Args:
            x: Input features (B, N, D) or (B, H, W, D)
            fusion_module: The AdaptiveScaleFusion module to use
            training: Whether in training mode
            
        Returns:
            Fused features (B, N, D) in sequence format
        """
        grid_size = self.config.grid_size
        H, W = grid_size
        B = tf.shape(x)[0]
        
        # Reshape to spatial if needed
        if len(x.shape) == 3:
            x_2d = tf.reshape(x, [B, H, W, self.config.embed_dim])
        else:
            x_2d = x
        
        # Create multi-scale features
        x_scale1 = x_2d  # Full resolution: 128x128
        x_scale2 = self.scale_pool_2x(x_2d)  # 64x64
        x_scale3 = self.scale_pool_4x(x_2d)  # 32x32
        
        # Upsample coarse scales to match fine scale
        x_scale2_up = tf.image.resize(x_scale2, [H, W], method='bilinear')
        x_scale3_up = tf.image.resize(x_scale3, [H, W], method='bilinear')
        
        # Adaptive fusion: learn to weight scales per location
        x_fused = fusion_module(
            [x_scale1, x_scale2_up, x_scale3_up],
            training=training
        )
        
        # Flatten back to sequence
        return tf.reshape(x_fused, [B, H * W, self.config.embed_dim])
    
    def call(
        self, 
        x: tf.Tensor, 
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Args:
            x: Encoded features from initial transformer (B, N, embed_dim)
            
        Returns:
            Dictionary with all intermediate and final outputs
        """
        outputs = {}
        grid_size = self.config.grid_size
        H, W = grid_size
        B = tf.shape(x)[0]
        
        # FIX: Multi-scale fusion at Stage 1 (initial)
        x = self._apply_multi_scale_fusion(x, self.scale_fusion_1, training)
        
        # Stage 2: Context generation
        stage2_output = self.stage2_transformer(x, training=training)
        context = self.context_head(stage2_output)
        outputs['stage2_output'] = stage2_output
        outputs['context'] = context
        
        # Gated context fusion
        stage3_input = self.context_gate(stage2_output, context)
        
        # FIX: Multi-scale fusion at Stage 2 (after context)
        stage3_input = self._apply_multi_scale_fusion(stage3_input, self.scale_fusion_2, training)
        
        # Stage 3: Token classification
        stage3_output = self.stage3_transformer(stage3_input, training=training)
        token_logits = self.token_head(stage3_output)
        outputs['stage3_output'] = stage3_output
        outputs['token_logits'] = token_logits
        
        # v2.0: Boundary-aware attention (uses token logits)
        boundary_refined = self.boundary_attention(
            stage3_output, token_logits, training=training
        )
        
        # Gated token classification fusion
        final_input = self.token_gate(boundary_refined, token_logits)
        
        # FIX: Multi-scale fusion at Stage 3 (after boundary attention)
        final_input = self._apply_multi_scale_fusion(final_input, self.scale_fusion_3, training)
        
        # Final stage: Segmentation
        final_output = self.final_transformer(final_input, training=training)
        class_logits = self.segmentation_head(final_output)
        outputs['final_output'] = final_output
        outputs['class_logits'] = class_logits
        
        # Gated class logits fusion for decoder input
        decoder_input = self.class_gate(final_output, class_logits)
        outputs['decoder_input'] = decoder_input
        
        return outputs


class PatchFormerEncoder(layers.Layer):
    """
    Complete encoder module for PatchFormer.
    
    Combines:
    - Dual-stream encoding (patches + conv)
    - Reconstruction path (training only)
    - Contrastive projection (training only)
    - Progressive segmentation encoding
    
    FIX #1: Masking is now applied BEFORE the split, so both paths
    see the same masked representations. This ensures:
    1. Reconstruction learns from masked input → reconstructs full image
    2. Segmentation also processes masked input → learns robust features
    3. The MAE objective properly regularizes the shared encoder
    
    During inference, no masking is applied.
    """
    
    def __init__(self, config: PatchFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Dual-stream encoder
        self.dual_encoder = DualStreamEncoder(config, name="dual_encoder")
        
        # Masking layer (applied before split during training)
        self.masking = PatchMasking(mask_ratio=config.mask_ratio, name="masking")
        
        # Reconstruction path (training only)
        self.reconstruction_decoder = ReconstructionDecoder(config, name="reconstruction_decoder")
        
        # Contrastive projection (training only)
        # Applied to UNMASKED embeddings for proper contrastive learning
        self.contrastive_projection = ProjectionHead(
            hidden_dim=config.embed_dim,
            output_dim=config.projection_dim,
            name="contrastive_projection"
        )
        
        # Progressive encoder
        self.progressive_encoder = ProgressiveEncoder(config, name="progressive_encoder")
    
    def call(
        self,
        images: tf.Tensor,
        training: bool = False,
        return_all: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Args:
            images: Input images (B, H, W, 3)
            training: Whether in training mode
            return_all: Whether to return all intermediate outputs
            
        Returns:
            Dictionary with model outputs
        """
        outputs = {}
        
        # Dual-stream encoding
        encoded, conv_intermediates = self.dual_encoder(
            images, 
            training=training,
            return_conv_intermediates=self.config.use_skip_connections
        )
        outputs['encoded'] = encoded
        
        if conv_intermediates is not None:
            outputs['conv_intermediates'] = conv_intermediates
        
        # FIX: Masking ONLY affects reconstruction path
        # Segmentation and contrastive both use FULL unmasked features
        # This eliminates train/test distribution mismatch for segmentation
        if training:
            # Apply masking ONLY for reconstruction (MAE auxiliary task)
            masked_encoded, mask, indices = self.masking(encoded, training=True)
            outputs['mask'] = mask
            outputs['mask_indices'] = indices
            
            # Reconstruction from MASKED input (learns to reconstruct from partial info)
            reconstruction = self.reconstruction_decoder(masked_encoded, training=True)
            outputs['reconstruction'] = reconstruction
            
            # Contrastive embeddings from UNMASKED input (consistent with segmentation)
            contrastive_embeddings = self.contrastive_projection(encoded)
            outputs['contrastive_embeddings'] = contrastive_embeddings
            
            # Progressive encoding (segmentation) uses FULL UNMASKED features
            # This ensures train/test consistency for the main task
            progressive_outputs = self.progressive_encoder(encoded, training=training)
        else:
            # During inference, no masking
            progressive_outputs = self.progressive_encoder(encoded, training=training)
        
        outputs.update(progressive_outputs)
        
        return outputs
