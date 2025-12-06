"""
Configuration classes for PatchFormer model variants.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PatchFormerConfig:
    """
    Configuration for PatchFormer model.
    
    Attributes:
        input_size: Input image resolution (height, width)
        patch_size: Size of each patch (height, width)
        num_classes: Number of segmentation classes
        
        # Encoder dimensions
        patch_dim: Dimension after patch flattening (patch_size^2 * 3)
        conv_dim: Output dimension of convolutional stream
        embed_dim: Fused embedding dimension
        
        # Transformer configuration
        num_heads: Number of attention heads
        ffn_ratio: FFN hidden dimension ratio
        dropout_rate: Dropout rate for regularization
        attention_dropout: Dropout rate for attention weights
        
        # Stage configurations
        initial_transformer_layers: Layers in initial transformer block
        stage2_transformer_layers: Layers in stage 2 (context generation)
        stage3_transformer_layers: Layers in stage 3 (token classification)
        final_transformer_layers: Layers in final stage
        
        # Head dimensions
        context_dim: Dimension of context generation head
        token_classes: Number of token classification classes (interior, boundary, background)
        
        # Window attention
        use_windowed_attention: Whether to use windowed attention
        window_size: Window size for windowed attention
        
        # Reconstruction
        mask_ratio: Ratio of patches to mask during training
        reconstruction_dim: Intermediate dimension for reconstruction decoder
        
        # Contrastive learning
        projection_dim: Dimension of contrastive projection head
        temperature: Temperature for contrastive loss
        
        # Training
        use_skip_connections: Whether to use skip connections in decoder
    """
    
    # Input configuration
    input_size: Tuple[int, int] = (1024, 1024)
    patch_size: Tuple[int, int] = (8, 8)
    num_classes: int = 2  # Default: binary (text/background)
    
    # Encoder dimensions
    patch_dim: int = 192  # 8 * 8 * 3
    conv_dim: int = 320
    embed_dim: int = 512  # patch_dim + conv_dim = 192 + 320
    
    # Transformer configuration
    num_heads: int = 8
    ffn_ratio: float = 2.0
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    
    # Stage configurations
    initial_transformer_layers: int = 1
    stage2_transformer_layers: int = 2
    stage3_transformer_layers: int = 2
    final_transformer_layers: int = 1
    
    # Head dimensions
    context_dim: int = 32
    token_classes: int = 3  # interior, boundary, background
    
    # Window attention
    use_windowed_attention: bool = True
    window_size: int = 8  # 8x8 windows in the 128x128 grid
    
    # Reconstruction
    mask_ratio: float = 0.5
    reconstruction_dim: int = 384
    
    # Contrastive learning
    projection_dim: int = 128
    temperature: float = 0.07
    
    # Architecture options
    use_skip_connections: bool = True
    use_squeeze_excitation: bool = True
    
    # Computed properties
    @property
    def grid_size(self) -> Tuple[int, int]:
        """Size of the patch grid."""
        return (
            self.input_size[0] // self.patch_size[0],
            self.input_size[1] // self.patch_size[1]
        )
    
    @property
    def num_patches(self) -> int:
        """Total number of patches."""
        h, w = self.grid_size
        return h * w
    
    @property
    def stage2_input_dim(self) -> int:
        """Input dimension for stage 2 (after initial transformer)."""
        return self.embed_dim
    
    @property
    def stage3_input_dim(self) -> int:
        """Input dimension for stage 3.
        
        With gating mechanism, dimension stays at embed_dim.
        (Previously used concatenation: embed_dim + context_dim = 544)
        """
        return self.embed_dim
    
    @property
    def final_input_dim(self) -> int:
        """Input dimension for final stage.
        
        With gating mechanism, dimension stays at embed_dim.
        (Previously used concatenation: stage3_input_dim + token_classes = 547)
        """
        return self.embed_dim
    
    @property
    def decoder_input_dim(self) -> int:
        """Input dimension for decoder.
        
        With gating mechanism, dimension stays at embed_dim.
        (Previously used concatenation: final_input_dim + num_classes)
        """
        return self.embed_dim


def get_config(variant: str = "small", num_classes: int = 2) -> PatchFormerConfig:
    """
    Get predefined configuration variants.
    
    Args:
        variant: One of 'tiny', 'small', 'base'
        num_classes: Number of segmentation classes
        
    Returns:
        PatchFormerConfig instance
    """
    configs = {
        "tiny": PatchFormerConfig(
            num_classes=num_classes,
            embed_dim=256,
            conv_dim=160,
            patch_dim=96,  # Will need smaller patches or projection
            num_heads=4,
            ffn_ratio=1.5,
            initial_transformer_layers=1,
            stage2_transformer_layers=1,
            stage3_transformer_layers=1,
            final_transformer_layers=1,
            context_dim=16,
            projection_dim=64,
            reconstruction_dim=192,
            use_squeeze_excitation=False,
        ),
        "small": PatchFormerConfig(
            num_classes=num_classes,
            embed_dim=512,
            conv_dim=320,
            patch_dim=192,
            num_heads=8,
            ffn_ratio=2.0,
            initial_transformer_layers=1,
            stage2_transformer_layers=2,
            stage3_transformer_layers=2,
            final_transformer_layers=1,
            context_dim=32,
            projection_dim=128,
            reconstruction_dim=384,
        ),
        "base": PatchFormerConfig(
            num_classes=num_classes,
            embed_dim=768,
            conv_dim=448,
            patch_dim=320,
            num_heads=12,
            ffn_ratio=4.0,
            initial_transformer_layers=2,
            stage2_transformer_layers=3,
            stage3_transformer_layers=3,
            final_transformer_layers=2,
            context_dim=64,
            projection_dim=256,
            reconstruction_dim=512,
            window_size=16,
        ),
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
    config = configs[variant]
    config.num_classes = num_classes
    
    # Adjust patch_dim based on actual patch size
    config.patch_dim = config.patch_size[0] * config.patch_size[1] * 3
    
    # For tiny, we need to project patch_dim to match
    if variant == "tiny":
        # Will use a projection layer instead of direct flatten
        pass
    
    return config


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.
    """
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Training
    batch_size: int = 4
    epochs: int = 100
    gradient_clip_norm: float = 1.0
    
    # Loss weights
    loss_segmentation: float = 1.0
    loss_token: float = 0.5
    loss_reconstruction: float = 0.3
    loss_contrastive: float = 0.1
    
    # Phase transition
    phase1_epochs: int = 30  # Representation learning phase
    phase2_reconstruction_weight: float = 0.1  # Reduced weight after phase 1
    
    # Masking schedule
    phase1_mask_ratio: float = 0.75
    phase2_mask_ratio: float = 0.5
    
    # Data augmentation
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 5  # Save every N epochs
    
    # Validation
    val_frequency: int = 1  # Validate every N epochs
