"""
Utility functions for PatchFormer.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List
import json
from pathlib import Path


# ============================================================================
# Metrics
# ============================================================================

class IoUMetric(tf.keras.metrics.Metric):
    """
    Intersection over Union metric for segmentation.
    """
    
    def __init__(self, num_classes: int, name: str = "iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        
        # Confusion matrix accumulators
        self.total_cm = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.float32
        )
    
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """
        Update confusion matrix.
        
        Args:
            y_true: Ground truth (B, H, W) with class indices
            y_pred: Predictions (B, H, W) with class indices or (B, H, W, C) logits
        """
        # Handle logits
        if len(y_pred.shape) == 4:
            y_pred = tf.argmax(y_pred, axis=-1)
        
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.int32)
        
        # Compute confusion matrix
        cm = tf.math.confusion_matrix(
            y_true, y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )
        
        self.total_cm.assign_add(cm)
    
    def result(self) -> tf.Tensor:
        """Compute mean IoU from confusion matrix."""
        # IoU = TP / (TP + FP + FN)
        # TP = diagonal
        # FP = column sum - diagonal
        # FN = row sum - diagonal
        
        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)
        diagonal = tf.linalg.diag_part(self.total_cm)
        
        denominator = sum_over_row + sum_over_col - diagonal
        
        # Avoid division by zero
        iou_per_class = tf.math.divide_no_nan(diagonal, denominator)
        
        # Mean IoU (excluding classes with no samples)
        valid_mask = tf.cast(denominator > 0, tf.float32)
        mean_iou = tf.reduce_sum(iou_per_class * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-6)
        
        return mean_iou
    
    def reset_state(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))


class DiceMetric(tf.keras.metrics.Metric):
    """
    Dice coefficient metric.
    """
    
    def __init__(self, name: str = "dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")
    
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        # Convert to binary
        if len(y_pred.shape) == 4:
            y_pred = tf.argmax(y_pred, axis=-1)
        
        y_true = tf.cast(y_true > 0, tf.float32)
        y_pred = tf.cast(y_pred > 0, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
    
    def result(self) -> tf.Tensor:
        return (2.0 * self.intersection + 1e-6) / (self.union + 1e-6)
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class BoundaryF1Metric(tf.keras.metrics.Metric):
    """
    F1 score specifically for boundary detection.
    """
    
    def __init__(self, tolerance: int = 2, name: str = "boundary_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tolerance = tolerance
        
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
    
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        if len(y_pred.shape) == 4:
            y_pred = tf.argmax(y_pred, axis=-1)
        
        # Extract boundaries
        true_boundary = self._extract_boundary(y_true)
        pred_boundary = self._extract_boundary(y_pred)
        
        # Dilate for tolerance matching
        true_dilated = self._dilate(true_boundary, self.tolerance)
        pred_dilated = self._dilate(pred_boundary, self.tolerance)
        
        # Compute TP, FP, FN
        tp = tf.reduce_sum(tf.cast(pred_boundary & true_dilated, tf.float32))
        fp = tf.reduce_sum(tf.cast(pred_boundary & ~true_dilated, tf.float32))
        fn = tf.reduce_sum(tf.cast(true_boundary & ~pred_dilated, tf.float32))
        
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
    
    def _extract_boundary(self, mask: tf.Tensor) -> tf.Tensor:
        """Extract boundary from segmentation mask."""
        mask = tf.cast(mask > 0, tf.float32)
        mask = tf.expand_dims(mask, -1)
        
        # Erosion
        eroded = -tf.nn.max_pool2d(-mask, ksize=3, strides=1, padding='SAME')
        
        # Boundary = mask - eroded
        boundary = mask - eroded
        
        return tf.squeeze(boundary, -1) > 0.5
    
    def _dilate(self, mask: tf.Tensor, iterations: int) -> tf.Tensor:
        """Dilate binary mask."""
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, -1)
        
        for _ in range(iterations):
            mask = tf.nn.max_pool2d(mask, ksize=3, strides=1, padding='SAME')
        
        return tf.squeeze(mask, -1) > 0.5
    
    def result(self) -> tf.Tensor:
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return f1
    
    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


# ============================================================================
# Learning Rate Schedules
# ============================================================================

class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay learning rate with linear warmup.
    """
    
    def __init__(
        self,
        initial_learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        alpha: float = 0.0,
    ):
        super().__init__()
        self.initial_lr = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        
        # Linear warmup
        warmup_lr = self.initial_lr * (step / warmup_steps)
        
        # Cosine decay
        decay_step = tf.minimum(step - warmup_steps, decay_steps)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_step / decay_steps))
        decayed_lr = (self.initial_lr - self.alpha) * cosine_decay + self.alpha
        
        # Choose based on step
        return tf.where(step < warmup_steps, warmup_lr, decayed_lr)
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
        }


# ============================================================================
# Visualization
# ============================================================================

def visualize_predictions(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Create visualization of prediction vs ground truth.
    
    Args:
        image: Original image (H, W, 3)
        ground_truth: GT mask (H, W)
        prediction: Predicted mask (H, W)
        class_colors: Dict mapping class index to RGB color
        
    Returns:
        Visualization image (H, W*3, 3)
    """
    if class_colors is None:
        class_colors = {
            0: (0, 0, 0),      # Background - black
            1: (0, 255, 0),    # Foreground - green
        }
    
    h, w = ground_truth.shape
    
    # Create colored masks
    gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in class_colors.items():
        gt_colored[ground_truth == class_idx] = color
        pred_colored[prediction == class_idx] = color
    
    # Blend with original image
    alpha = 0.5
    image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    gt_overlay = (alpha * gt_colored + (1 - alpha) * image_uint8).astype(np.uint8)
    pred_overlay = (alpha * pred_colored + (1 - alpha) * image_uint8).astype(np.uint8)
    
    # Concatenate
    vis = np.concatenate([image_uint8, gt_overlay, pred_overlay], axis=1)
    
    return vis


def visualize_token_classification(
    image: np.ndarray,
    token_labels: np.ndarray,
    grid_size: Tuple[int, int],
) -> np.ndarray:
    """
    Visualize token classification (interior/boundary/background).
    
    Args:
        image: Original image (H, W, 3)
        token_labels: Token labels (N,) with values {0, 1, 2}
        grid_size: (grid_h, grid_w)
        
    Returns:
        Visualization image
    """
    import cv2
    
    h, w = image.shape[:2]
    grid_h, grid_w = grid_size
    
    # Reshape token labels to grid
    token_grid = token_labels.reshape(grid_h, grid_w)
    
    # Create colored overlay
    colors = {
        0: (0, 255, 0),    # Interior - green
        1: (255, 255, 0),  # Boundary - yellow
        2: (100, 100, 100), # Background - gray
    }
    
    overlay = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for class_idx, color in colors.items():
        overlay[token_grid == class_idx] = color
    
    # Upscale to image size
    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Blend
    image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    vis = (0.5 * overlay + 0.5 * image_uint8).astype(np.uint8)
    
    return vis


# ============================================================================
# Checkpointing
# ============================================================================

class CheckpointManager:
    """
    Manage model checkpoints.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        checkpoint_dir: str,
        max_to_keep: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint = tf.train.Checkpoint(
            model=model,
            optimizer=optimizer
        )
        
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            str(self.checkpoint_dir),
            max_to_keep=max_to_keep
        )
        
        self.best_metric = float('-inf')
        self.best_path = None
    
    def save(self, epoch: int, metric: Optional[float] = None) -> str:
        """Save checkpoint."""
        path = self.manager.save()
        
        # Save best if metric improved
        if metric is not None and metric > self.best_metric:
            self.best_metric = metric
            self.best_path = self.checkpoint_dir / "best"
            self.checkpoint.write(str(self.best_path))
        
        return path
    
    def restore(self, path: Optional[str] = None) -> bool:
        """
        Restore checkpoint.
        
        Args:
            path: Specific checkpoint path or None for latest
            
        Returns:
            Whether restoration was successful
        """
        if path is None:
            path = self.manager.latest_checkpoint
        
        if path is None:
            return False
        
        self.checkpoint.restore(path)
        return True
    
    def restore_best(self) -> bool:
        """Restore best checkpoint."""
        if self.best_path is not None:
            self.checkpoint.restore(str(self.best_path))
            return True
        return False


# ============================================================================
# Config I/O
# ============================================================================

def save_config(config, path: str):
    """Save configuration to JSON."""
    with open(path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)


def load_config(path: str, config_class):
    """Load configuration from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return config_class(**data)


# ============================================================================
# Model Analysis
# ============================================================================

def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dict with total, trainable, and non-trainable counts
    """
    trainable = sum(
        tf.reduce_prod(w.shape).numpy() 
        for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.reduce_prod(w.shape).numpy() 
        for w in model.non_trainable_weights
    )
    
    return {
        "total": trainable + non_trainable,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def estimate_flops(model: tf.keras.Model, input_shape: Tuple[int, ...]) -> int:
    """
    Estimate model FLOPs (rough approximation).
    
    Note: This is a simplified estimate. For accurate FLOPs,
    use TensorFlow profiler or similar tools.
    """
    # Create concrete function
    @tf.function
    def forward(x):
        return model(x, training=False)
    
    concrete_func = forward.get_concrete_function(
        tf.TensorSpec(shape=(1, *input_shape), dtype=tf.float32)
    )
    
    # Use TF profiler if available
    try:
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        
        graph = concrete_func.graph
        profiler_options = ProfileOptionBuilder.float_operation()
        profile_result = profile(graph, options=profiler_options)
        
        return profile_result.total_float_ops
    except:
        # Fallback: rough estimate based on parameters
        params = count_parameters(model)['total']
        # Assume ~2 FLOPs per parameter (multiply-add)
        return params * 2


def print_model_summary(model: tf.keras.Model, input_shape: Tuple[int, ...]):
    """Print detailed model summary."""
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    # Build model
    dummy_input = tf.zeros((1, *input_shape))
    _ = model(dummy_input, training=False)
    
    # Parameter counts
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total:         {params['total']:,}")
    print(f"  Trainable:     {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    
    # Memory estimate (rough)
    memory_mb = params['total'] * 4 / (1024 * 1024)  # Assuming float32
    print(f"\nEstimated memory: {memory_mb:.2f} MB (weights only)")
    
    # Layer breakdown
    print(f"\nLayer breakdown:")
    for layer in model.layers:
        layer_params = sum(
            tf.reduce_prod(w.shape).numpy() 
            for w in layer.trainable_weights
        )
        if layer_params > 0:
            print(f"  {layer.name}: {layer_params:,}")
    
    print("=" * 60)
