"""
Main PatchFormer model class.

Combines encoder, decoder, and provides training/inference interfaces.

v2.0 additions:
- PatchFormerV2 model with advanced heads
- Support for differentiable binarization
- Support for polygon vertex prediction
- Affinity-based instance linking
"""

import tensorflow as tf
from tensorflow.keras import Model
from typing import Dict, Optional, Tuple, Union

from .config import PatchFormerConfig, get_config
from .encoder import PatchFormerEncoder
from .decoder import (
    SegmentationDecoder, 
    MultiScaleDecoder, 
    TokenToSegmentation,
    PatchFormerDecoderV2
)


class PatchFormer(Model):
    """
    PatchFormer: Lightweight Semantic Segmentation Architecture.
    
    A high-resolution semantic segmentation model combining:
    - Dual-stream patch + convolutional encoding
    - Masked autoencoder reconstruction (training)
    - Contrastive embedding learning (training)
    - Progressive transformer encoding with intermediate supervision
    - Lightweight upsampling decoder
    
    Optimized for CPU inference and TensorFlow Lite deployment.
    """
    
    def __init__(
        self,
        config: Optional[PatchFormerConfig] = None,
        variant: str = "small",
        num_classes: int = 2,
        use_multi_scale_decoder: bool = False,
        **kwargs
    ):
        """
        Initialize PatchFormer model.
        
        Args:
            config: PatchFormerConfig instance. If None, uses variant preset.
            variant: Configuration variant ('tiny', 'small', 'base')
            num_classes: Number of segmentation classes
            use_multi_scale_decoder: Whether to use multi-scale decoder with auxiliary outputs
        """
        super().__init__(**kwargs)
        
        if config is None:
            config = get_config(variant, num_classes)
        
        self.config = config
        self.use_multi_scale = use_multi_scale_decoder
        
        # Encoder
        self.encoder = PatchFormerEncoder(config, name="encoder")
        
        # Decoder
        if use_multi_scale_decoder:
            self.decoder = MultiScaleDecoder(config, name="decoder")
        else:
            self.decoder = SegmentationDecoder(config, name="decoder")
        
        # Direct token-to-segmentation for efficient inference
        self.token_to_seg = TokenToSegmentation(config, name="token_to_seg")
    
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        return_all_outputs: bool = False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Forward pass.
        
        Args:
            inputs: Input images (B, H, W, 3), values in [0, 1]
            training: Whether in training mode
            return_all_outputs: Whether to return all intermediate outputs
            
        Returns:
            If return_all_outputs or training:
                Dict with all model outputs
            Else:
                Segmentation logits (B, H, W, num_classes)
        """
        # Encode
        encoder_outputs = self.encoder(inputs, training=training)
        
        # Decode
        if self.use_multi_scale:
            decoder_outputs = self.decoder(
                encoder_outputs['decoder_input'],
                conv_intermediates=encoder_outputs.get('conv_intermediates'),
                training=training,
                return_aux=training
            )
            main_output = decoder_outputs['main']
        else:
            main_output = self.decoder(
                encoder_outputs['decoder_input'],
                conv_intermediates=encoder_outputs.get('conv_intermediates'),
                training=training
            )
            decoder_outputs = {'main': main_output}
        
        if training or return_all_outputs:
            # Combine all outputs
            outputs = {**encoder_outputs, **decoder_outputs}
            return outputs
        
        # Inference: return only segmentation output
        return main_output
    
    def predict_efficient(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Efficient inference using token-to-segmentation shortcut.
        
        Bypasses the learned decoder and uses bilinear upsampling.
        Faster but potentially less accurate.
        
        Args:
            inputs: Input images (B, H, W, 3)
            
        Returns:
            Segmentation logits (B, H, W, num_classes)
        """
        encoder_outputs = self.encoder(inputs, training=False)
        return self.token_to_seg(encoder_outputs['class_logits'])
    
    def get_config(self):
        return {
            "config": self.config.__dict__,
            "use_multi_scale_decoder": self.use_multi_scale,
        }
    
    @classmethod
    def from_config(cls, config_dict):
        model_config = PatchFormerConfig(**config_dict["config"])
        return cls(
            config=model_config,
            use_multi_scale_decoder=config_dict["use_multi_scale_decoder"]
        )


class PatchFormerV2(Model):
    """
    PatchFormer v2.0: Enhanced Architecture for Competitive Text Detection.
    
    UNIFIED REFINEMENT PIPELINE:
    All v2.0 additions feed into a SINGLE final segmentation output:
    
    Encoder:
    - Multi-scale token aggregation (scale invariance)
    - Boundary-aware cross-attention (boundary precision)
    
    Decoder (Cascaded Refinement):
    - Affinity-guided feature propagation (coherent regions)
    - Vertex-aware boundary sharpening (polygon accuracy)
    - Differentiable binarization (sharp output)
    
    Final output: Single refined segmentation mask
    
    Targets: Outperform DBNet, CRAFT, TextFuseNet on standard benchmarks.
    """
    
    def __init__(
        self,
        config: Optional[PatchFormerConfig] = None,
        variant: str = "small",
        num_classes: int = 2,
        use_affinity: bool = True,
        use_vertex: bool = True,
        use_db: bool = True,
        **kwargs
    ):
        """
        Initialize PatchFormer v2.0 model.
        
        Args:
            config: PatchFormerConfig instance
            variant: Configuration variant ('tiny', 'small', 'base')
            num_classes: Number of segmentation classes
            use_affinity: Enable affinity-guided refinement
            use_vertex: Enable vertex-aware boundary sharpening
            use_db: Enable differentiable binarization
        """
        super().__init__(**kwargs)
        
        if config is None:
            config = get_config(variant, num_classes)
        
        self.config = config
        self.use_affinity = use_affinity
        self.use_vertex = use_vertex
        self.use_db = use_db
        
        # Encoder (v2.0: multi-scale aggregation, boundary attention)
        self.encoder = PatchFormerEncoder(config, name="encoder")
        
        # v2.0 Decoder with Cascaded Refinement
        # All components (affinity, vertex, DB) feed into final output
        self.decoder = PatchFormerDecoderV2(
            config,
            use_affinity=use_affinity,
            use_vertex=use_vertex,
            use_db=use_db,
            name="decoder"
        )
        
        # Direct token-to-segmentation for efficient inference
        self.token_to_seg = TokenToSegmentation(config, name="token_to_seg")
    
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        return_all_outputs: bool = False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Forward pass.
        
        Args:
            inputs: Input images (B, H, W, 3), values in [0, 1]
            training: Whether in training mode
            return_all_outputs: Whether to return all intermediate outputs
            
        Returns:
            If return_all_outputs or training:
                Dict with all model outputs including:
                - 'final': MAIN OUTPUT - refined segmentation
                - 'main': Alias for 'final'
                - 'initial_seg': Pre-refinement segmentation
                - 'affinity': Affinity maps (if enabled)
                - 'vertex_prob': Vertex probability (if enabled)
                - 'db_outputs': DB intermediate outputs (if enabled)
                - All encoder intermediate outputs
            Else:
                Final refined segmentation (B, H, W, 1)
        """
        # Encode
        encoder_outputs = self.encoder(inputs, training=training)
        
        # Decode with cascaded refinement
        decoder_outputs = self.decoder(
            encoder_outputs['decoder_input'],
            conv_intermediates=encoder_outputs.get('conv_intermediates'),
            training=training
        )
        
        if training or return_all_outputs:
            # Combine all outputs
            outputs = {**encoder_outputs, **decoder_outputs}
            return outputs
        
        # Inference: return the final refined output
        return decoder_outputs['final']
    
    def predict_with_affinity(
        self, 
        inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict segmentation with affinity maps for instance grouping.
        
        Args:
            inputs: Input images (B, H, W, 3)
            
        Returns:
            segmentation: Final refined segmentation (B, H, W, 1)
            affinity: Affinity maps from refinement (B, 1024, 1024, 8)
        """
        outputs = self(inputs, training=False, return_all_outputs=True)
        
        segmentation = outputs['final']
        affinity = outputs.get('affinity')
        
        return segmentation, affinity
    
    def predict_with_vertices(
        self, 
        inputs: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Predict segmentation with vertex information for polygon extraction.
        
        Args:
            inputs: Input images (B, H, W, 3)
            
        Returns:
            Dict with:
            - 'segmentation': Final refined mask
            - 'vertex_prob': Vertex probability map
            - 'vertex_offset': Offset to nearest vertex
        """
        outputs = self(inputs, training=False, return_all_outputs=True)
        
        return {
            'segmentation': outputs['final'],
            'vertex_prob': outputs.get('vertex_prob'),
            'vertex_offset': outputs.get('vertex_offset'),
        }
    
    def get_config(self):
        return {
            "config": self.config.__dict__,
            "use_affinity": self.use_affinity,
            "use_vertex": self.use_vertex,
            "use_db": self.use_db,
        }
    
    @classmethod
    def from_config(cls, config_dict):
        model_config = PatchFormerConfig(**config_dict["config"])
        return cls(
            config=model_config,
            use_affinity=config_dict.get("use_affinity", True),
            use_vertex=config_dict.get("use_vertex", True),
            use_db=config_dict.get("use_db", True)
        )


class PatchFormerTrainer:
    """
    Training wrapper for PatchFormer with multi-loss handling.
    """
    
    def __init__(
        self,
        model: PatchFormer,
        loss_fn,
        optimizer: tf.keras.optimizers.Optimizer,
        metrics: Optional[Dict] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics or {}
        
        # Track training state
        self.current_epoch = 0
    
    @tf.function
    def train_step(
        self,
        images: tf.Tensor,
        segmentation_masks: tf.Tensor,
        token_labels: Optional[tf.Tensor] = None,
    ) -> Dict[str, tf.Tensor]:
        """
        Single training step.
        
        Args:
            images: Input images (B, H, W, 3)
            segmentation_masks: Ground truth masks (B, H, W)
            token_labels: Optional token classification labels (B, N)
            
        Returns:
            Dictionary with loss values
        """
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model(images, training=True)
            
            # Prepare ground truth
            ground_truth = {
                'segmentation': segmentation_masks,
                'images': images,
            }
            if token_labels is not None:
                ground_truth['token_labels'] = token_labels
            
            # Compute losses
            total_loss, loss_dict = self.loss_fn(
                outputs, ground_truth, self.current_epoch
            )
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        
        return loss_dict
    
    @tf.function
    def val_step(
        self,
        images: tf.Tensor,
        segmentation_masks: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """
        Single validation step.
        """
        outputs = self.model(images, training=False, return_all_outputs=True)
        
        ground_truth = {
            'segmentation': segmentation_masks,
            'images': images,
        }
        
        _, loss_dict = self.loss_fn(outputs, ground_truth, self.current_epoch)
        
        # Compute metrics
        predictions = tf.argmax(outputs['main'], axis=-1)
        
        for name, metric in self.metrics.items():
            metric.update_state(segmentation_masks, predictions)
        
        return loss_dict
    
    def set_epoch(self, epoch: int):
        """Update current epoch for loss scheduling."""
        self.current_epoch = epoch


def create_model(
    variant: str = "small",
    num_classes: int = 2,
    input_size: Tuple[int, int] = (1024, 1024),
    use_multi_scale: bool = False,
) -> PatchFormer:
    """
    Factory function to create PatchFormer model.
    
    Args:
        variant: Model size ('tiny', 'small', 'base')
        num_classes: Number of segmentation classes
        input_size: Input image size
        use_multi_scale: Whether to use multi-scale decoder
        
    Returns:
        PatchFormer model instance
    """
    config = get_config(variant, num_classes)
    config.input_size = input_size
    
    model = PatchFormer(
        config=config,
        use_multi_scale_decoder=use_multi_scale
    )
    
    # Build model
    dummy_input = tf.zeros((1, *input_size, 3))
    _ = model(dummy_input, training=False)
    
    return model


def load_model(
    checkpoint_path: str,
    variant: str = "small",
    num_classes: int = 2,
) -> PatchFormer:
    """
    Load PatchFormer model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        variant: Model variant
        num_classes: Number of classes
        
    Returns:
        Loaded model
    """
    model = create_model(variant=variant, num_classes=num_classes)
    model.load_weights(checkpoint_path)
    return model


class TFLiteExporter:
    """
    Export PatchFormer to TensorFlow Lite format.
    """
    
    @staticmethod
    def export(
        model: PatchFormer,
        output_path: str,
        quantize: bool = True,
        representative_dataset=None,
    ):
        """
        Export model to TFLite.
        
        Args:
            model: PatchFormer model
            output_path: Output .tflite file path
            quantize: Whether to apply INT8 quantization
            representative_dataset: Generator for calibration data
        """
        # Create concrete function for inference only
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, *model.config.input_size, 3], dtype=tf.float32)
        ])
        def inference_fn(images):
            return model(images, training=False)
        
        concrete_func = inference_fn.get_concrete_function()
        
        # Convert
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_dataset is not None:
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model exported to {output_path}")
        print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    @staticmethod
    def export_efficient(
        model: PatchFormer,
        output_path: str,
    ):
        """
        Export efficient inference version (token-to-seg shortcut).
        """
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, *model.config.input_size, 3], dtype=tf.float32)
        ])
        def efficient_fn(images):
            return model.predict_efficient(images)
        
        concrete_func = efficient_fn.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Efficient model exported to {output_path}")
        print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
