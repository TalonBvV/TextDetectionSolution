"""
Lightweight decoder for PatchFormer semantic segmentation.

v2.0 additions:
- CascadedRefinementModule: Unified pipeline where all components
  feed into the final segmentation output
- Affinity-guided feature propagation
- Vertex-aware boundary refinement
- Differentiable binarization for sharp outputs
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, Optional, Tuple

from .config import PatchFormerConfig
from .layers import DepthwiseSeparableConv
from .heads import CascadedRefinementModule


class DecoderBlock(layers.Layer):
    """
    Single decoder block with upsampling and convolution.
    
    FIX: Properly handles skip connection fusion with channel reduction
    to avoid information loss from doubled channels.
    """
    
    def __init__(
        self,
        filters: int,
        upsample_factor: int = 2,
        use_skip: bool = False,
        skip_channels: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.upsample_factor = upsample_factor
        self.use_skip = use_skip
        self.skip_channels = skip_channels
        
        # Upsampling (bilinear is TFLite friendly)
        self.upsample = layers.UpSampling2D(
            size=(upsample_factor, upsample_factor),
            interpolation="bilinear"
        )
        
        # Convolution after upsampling
        self.conv1 = DepthwiseSeparableConv(
            filters=filters,
            kernel_size=3,
            name="conv1"
        )
        
        # Additional conv if using skip connections
        if use_skip:
            # FIX: Project skip to half filters to balance with main path
            # This avoids the channel doubling issue while preserving skip info
            self.skip_conv = DepthwiseSeparableConv(
                filters=filters // 2,  # Half channels for skip
                kernel_size=1,
                name="skip_conv"
            )
            # Reduce main features to make room for skip
            self.main_reduce = layers.Conv2D(
                filters // 2,  # Half channels for main
                kernel_size=1,
                padding='same',
                name="main_reduce"
            )
            # FIX: Fusion with attention-weighted combination instead of simple concat
            self.fusion_gate = layers.Conv2D(
                filters // 2,  # Gate for skip features
                kernel_size=1,
                activation='sigmoid',
                padding='same',
                name="fusion_gate"
            )
            self.fusion_conv = DepthwiseSeparableConv(
                filters=filters,
                kernel_size=3,
                name="fusion_conv"
            )
            self.fusion_norm = layers.BatchNormalization(name="fusion_norm")
    
    def call(
        self, 
        x: tf.Tensor, 
        skip: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            x: Input features (B, H, W, C)
            skip: Optional skip connection features
            training: Whether in training mode
        """
        # Upsample
        x = self.upsample(x)
        x = self.conv1(x, training=training)
        
        # FIX: Improved skip connection fusion with gating
        if self.use_skip and skip is not None:
            # Project skip features
            skip_feat = self.skip_conv(skip, training=training)
            
            # Reduce main features
            main_feat = self.main_reduce(x)
            
            # Compute attention gate based on both features
            # Gate determines how much skip information to use at each location
            gate = self.fusion_gate(tf.concat([main_feat, skip_feat], axis=-1))
            
            # Gated skip features
            gated_skip = skip_feat * gate
            
            # Concatenate and fuse (now filters//2 + filters//2 = filters)
            fused = tf.concat([main_feat, gated_skip], axis=-1)
            x = self.fusion_conv(fused, training=training)
            x = self.fusion_norm(x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "upsample_factor": self.upsample_factor,
            "use_skip": self.use_skip,
            "skip_channels": self.skip_channels,
        })
        return config


class SegmentationDecoder(layers.Layer):
    """
    Lightweight segmentation decoder.
    
    Upsamples from 128x128 back to 1024x1024 with optional skip connections.
    
    128x128 -> 256x256 -> 512x512 -> 1024x1024
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        
        # Input projection from decoder_input_dim to manageable size
        self.input_proj = layers.Dense(256, name="input_proj")
        
        # Decoder blocks
        # 128 -> 256
        self.block1 = DecoderBlock(
            filters=256,
            upsample_factor=2,
            use_skip=config.use_skip_connections,
            skip_channels=128,  # From conv block2
            name="block1"
        )
        
        # 256 -> 512
        self.block2 = DecoderBlock(
            filters=128,
            upsample_factor=2,
            use_skip=config.use_skip_connections,
            skip_channels=64,  # From conv block1
            name="block2"
        )
        
        # 512 -> 1024
        self.block3 = DecoderBlock(
            filters=64,
            upsample_factor=2,
            use_skip=False,
            name="block3"
        )
        
        # Final output convolution
        self.output_conv = layers.Conv2D(
            config.num_classes,
            kernel_size=1,
            padding="same",
            name="output_conv"
        )
    
    def call(
        self,
        decoder_input: tf.Tensor,
        conv_intermediates: Optional[Dict[str, tf.Tensor]] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            decoder_input: (B, N, decoder_input_dim) from encoder
            conv_intermediates: Optional dict with 'block1' and 'block2' features
            training: Whether in training mode
            
        Returns:
            Segmentation logits (B, H, W, num_classes)
        """
        batch_size = tf.shape(decoder_input)[0]
        
        # Project and reshape to spatial
        x = self.input_proj(decoder_input)
        x = tf.reshape(
            x,
            [batch_size, self.config.grid_size[0], self.config.grid_size[1], 256]
        )
        
        # Get skip connections
        skip2 = conv_intermediates.get('block2') if conv_intermediates else None  # 256x256
        skip1 = conv_intermediates.get('block1') if conv_intermediates else None  # 512x512
        
        # Decode
        x = self.block1(x, skip=skip2, training=training)  # 256x256
        x = self.block2(x, skip=skip1, training=training)  # 512x512
        x = self.block3(x, training=training)  # 1024x1024
        
        # Output
        x = self.output_conv(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        return config


class MultiScaleDecoder(layers.Layer):
    """
    Decoder with auxiliary outputs at multiple scales for deep supervision.
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        
        # Main decoder
        self.decoder = SegmentationDecoder(config, name="decoder")
        
        # Auxiliary heads at intermediate scales
        self.aux_head_256 = layers.Conv2D(
            config.num_classes, 1, padding="same", name="aux_256"
        )
        self.aux_head_512 = layers.Conv2D(
            config.num_classes, 1, padding="same", name="aux_512"
        )
    
    def call(
        self,
        decoder_input: tf.Tensor,
        conv_intermediates: Optional[Dict[str, tf.Tensor]] = None,
        training: bool = False,
        return_aux: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Args:
            decoder_input: (B, N, decoder_input_dim)
            conv_intermediates: Optional skip connection features
            training: Whether in training mode
            return_aux: Whether to return auxiliary outputs
            
        Returns:
            Dictionary with 'main' and optionally 'aux_256', 'aux_512'
        """
        batch_size = tf.shape(decoder_input)[0]
        
        # Project and reshape
        x = self.decoder.input_proj(decoder_input)
        x = tf.reshape(
            x,
            [batch_size, self.config.grid_size[0], self.config.grid_size[1], 256]
        )
        
        skip2 = conv_intermediates.get('block2') if conv_intermediates else None
        skip1 = conv_intermediates.get('block1') if conv_intermediates else None
        
        outputs = {}
        
        # Stage 1: 128 -> 256
        x = self.decoder.block1(x, skip=skip2, training=training)
        if return_aux and training:
            outputs['aux_256'] = self.aux_head_256(x)
        
        # Stage 2: 256 -> 512
        x = self.decoder.block2(x, skip=skip1, training=training)
        if return_aux and training:
            outputs['aux_512'] = self.aux_head_512(x)
        
        # Stage 3: 512 -> 1024
        x = self.decoder.block3(x, training=training)
        x = self.decoder.output_conv(x)
        outputs['main'] = x
        
        return outputs


class TokenToSegmentation(layers.Layer):
    """
    Direct token-to-segmentation mapping without learned upsampling.
    
    Uses reshape and bilinear upsampling for efficient inference.
    Useful as an auxiliary path or for lightweight deployment.
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        
        # Project to num_classes
        self.classifier = layers.Dense(config.num_classes, name="classifier")
    
    def call(self, class_logits: tf.Tensor) -> tf.Tensor:
        """
        Args:
            class_logits: (B, N, num_classes) from segmentation head
            
        Returns:
            Upsampled segmentation (B, H, W, num_classes)
        """
        batch_size = tf.shape(class_logits)[0]
        
        # Reshape to spatial grid
        x = tf.reshape(
            class_logits,
            [batch_size, self.config.grid_size[0], self.config.grid_size[1], self.config.num_classes]
        )
        
        # Bilinear upsample to full resolution
        x = tf.image.resize(
            x,
            self.config.input_size,
            method='bilinear'
        )
        
        return x


class PatchFormerDecoderV2(layers.Layer):
    """
    v2.0 Decoder with UNIFIED refinement pipeline.
    
    All components feed into a SINGLE final output:
    1. Decoder upsamples features (128→256→512→1024)
    2. CascadedRefinementModule refines features using:
       - Affinity-guided feature propagation
       - Vertex-aware boundary sharpening
       - Differentiable binarization for sharp output
    3. Single 'final' output for inference
    
    Intermediate outputs (affinity, vertex, initial_seg) available for
    auxiliary supervision during training.
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        use_affinity: bool = True,
        use_vertex: bool = True,
        use_db: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.use_affinity = use_affinity
        self.use_vertex = use_vertex
        self.use_db = use_db
        
        # Input projection from decoder_input_dim to manageable size
        self.input_proj = layers.Dense(256, name="input_proj")
        
        # Decoder blocks with skip connections
        # 128 -> 256
        self.block1 = DecoderBlock(
            filters=256,
            upsample_factor=2,
            use_skip=config.use_skip_connections,
            skip_channels=128,
            name="block1"
        )
        
        # 256 -> 512
        self.block2 = DecoderBlock(
            filters=128,
            upsample_factor=2,
            use_skip=config.use_skip_connections,
            skip_channels=64,
            name="block2"
        )
        
        # 512 -> 1024
        self.block3 = DecoderBlock(
            filters=64,
            upsample_factor=2,
            use_skip=False,
            name="block3"
        )
        
        # v2.0: Unified Cascaded Refinement Module
        # ALL components (affinity, vertex, DB) feed into final output
        self.refinement = CascadedRefinementModule(
            feature_dim=64,  # Matches block3 output
            num_classes=config.num_classes,
            use_affinity=use_affinity,
            use_vertex=use_vertex,
            use_db=use_db,
            name="cascaded_refinement"
        )
        
        # Auxiliary heads for deep supervision (training only)
        self.aux_head_256 = layers.Conv2D(
            config.num_classes, 1, padding="same", name="aux_256"
        )
        self.aux_head_512 = layers.Conv2D(
            config.num_classes, 1, padding="same", name="aux_512"
        )
    
    def call(
        self,
        decoder_input: tf.Tensor,
        conv_intermediates: Optional[Dict[str, tf.Tensor]] = None,
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Args:
            decoder_input: (B, N, D) from encoder
            conv_intermediates: Optional skip connection features
            training: Whether in training mode
            
        Returns:
            Dictionary with:
            - 'final': MAIN OUTPUT - refined segmentation (B, 1024, 1024, 1)
            - 'main': Alias for 'final' (backwards compatibility)
            - 'initial_seg': Pre-refinement segmentation (for aux loss)
            - 'affinity': Affinity maps (for aux loss)
            - 'vertex_prob': Vertex probability (for aux loss)
            - 'db_outputs': DB intermediate outputs (for aux loss)
            - 'aux_256', 'aux_512': Multi-scale aux outputs
        """
        batch_size = tf.shape(decoder_input)[0]
        outputs = {}
        
        # Project and reshape to spatial
        x = self.input_proj(decoder_input)
        x = tf.reshape(
            x,
            [batch_size, self.config.grid_size[0], self.config.grid_size[1], 256]
        )
        
        # Get skip connections
        skip2 = conv_intermediates.get('block2') if conv_intermediates else None
        skip1 = conv_intermediates.get('block1') if conv_intermediates else None
        
        # Decoder stage 1: 128 -> 256
        x = self.block1(x, skip=skip2, training=training)
        if training:
            outputs['aux_256'] = self.aux_head_256(x)
        
        # Decoder stage 2: 256 -> 512
        x = self.block2(x, skip=skip1, training=training)
        if training:
            outputs['aux_512'] = self.aux_head_512(x)
        
        # Decoder stage 3: 512 -> 1024
        x = self.block3(x, training=training)
        
        # v2.0: Cascaded Refinement Pipeline
        # Affinity → Vertex → DB → Final Output
        refinement_outputs = self.refinement(x, training=training)
        
        # Merge refinement outputs
        outputs.update(refinement_outputs)
        
        # 'final' is the main output, 'main' is alias for compatibility
        outputs['main'] = outputs['final']
        
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "use_affinity": self.use_affinity,
            "use_vertex": self.use_vertex,
            "use_db": self.use_db,
        })
        return config
