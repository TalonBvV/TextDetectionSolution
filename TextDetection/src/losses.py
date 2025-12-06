"""
Loss functions for PatchFormer training.

Includes:
- Segmentation loss (Dice + Focal)
- Token classification loss
- Reconstruction loss (MSE)
- Contrastive cosine loss
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, Optional, Tuple

from .config import PatchFormerConfig, TrainingConfig


class DiceLoss(tf.keras.losses.Loss):
    """
    Dice loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        class_weights: Optional[tf.Tensor] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.smooth = smooth
        self.class_weights = class_weights
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: Ground truth (B, H, W, C) or (B, H, W) with class indices
            y_pred: Predictions (B, H, W, C) logits
        """
        # Apply softmax to predictions
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Convert y_true to one-hot if necessary
        if len(y_true.shape) == 3:
            num_classes = y_pred.shape[-1]
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        
        y_true = tf.cast(y_true, tf.float32)
        
        # Flatten spatial dimensions
        y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        # Compute Dice per class
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)
        
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            dice_per_class = dice_per_class * self.class_weights
            dice = tf.reduce_sum(dice_per_class) / tf.reduce_sum(self.class_weights)
        else:
            dice = tf.reduce_mean(dice_per_class)
        
        return 1.0 - dice


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: Ground truth (B, H, W) with class indices or (B, H, W, C) one-hot
            y_pred: Predictions (B, H, W, C) logits
        """
        # Apply softmax
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Convert y_true to one-hot if necessary
        if len(y_true.shape) == 3 or y_true.shape[-1] == 1:
            if len(y_true.shape) == 4:
                y_true = tf.squeeze(y_true, axis=-1)
            num_classes = y_pred.shape[-1]
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        
        y_true = tf.cast(y_true, tf.float32)
        
        # Compute focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


class SegmentationLoss(tf.keras.losses.Loss):
    """
    Combined segmentation loss: Dice + Focal.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: Optional[tf.Tensor] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(class_weights=class_weights)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        dice = self.dice_loss(y_true, y_pred)
        focal = self.focal_loss(y_true, y_pred)
        
        return self.dice_weight * dice + self.focal_weight * focal


class TokenClassificationLoss(tf.keras.losses.Loss):
    """
    Loss for token-level classification (interior/boundary/background).
    
    Applies higher weight to boundary class.
    """
    
    def __init__(
        self,
        boundary_weight: float = 3.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.boundary_weight = boundary_weight
        
        # Class weights: [interior, boundary, background]
        self.class_weights = tf.constant([1.0, boundary_weight, 1.0])
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: Token labels (B, N) with values {0, 1, 2}
            y_pred: Token logits (B, N, 3)
        """
        # Sparse cross entropy
        y_true = tf.cast(y_true, tf.int32)
        
        # Compute sample weights based on class
        sample_weights = tf.gather(self.class_weights, y_true)
        
        # Compute cross entropy
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        
        # Apply weights
        weighted_loss = loss * sample_weights
        
        return tf.reduce_mean(weighted_loss)


class ReconstructionLoss(tf.keras.losses.Loss):
    """
    Reconstruction loss for masked autoencoder.
    
    MSE loss computed only on masked patches.
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (8, 8),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    
    def call(
        self, 
        y_true: tf.Tensor, 
        y_pred: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Args:
            y_true: Original image (B, H, W, 3)
            y_pred: Reconstructed image (B, H, W, 3)
            mask: Patch mask (B, N) indicating masked patches
        """
        if mask is None:
            # Compute MSE over entire image
            return tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Convert patch mask to pixel mask
        batch_size = tf.shape(y_true)[0]
        h = y_true.shape[1] // self.patch_size[0]
        w = y_true.shape[2] // self.patch_size[1]
        
        # Reshape mask to spatial grid
        mask_spatial = tf.reshape(mask, [batch_size, h, w, 1])
        mask_spatial = tf.cast(mask_spatial, tf.float32)
        
        # Upsample mask to pixel resolution
        mask_pixels = tf.image.resize(
            mask_spatial,
            [y_true.shape[1], y_true.shape[2]],
            method='nearest'
        )
        
        # Compute MSE only on masked regions
        squared_error = tf.square(y_true - y_pred)
        masked_error = squared_error * mask_pixels
        
        # Average over masked pixels
        num_masked = tf.reduce_sum(mask_pixels) + 1e-6
        loss = tf.reduce_sum(masked_error) / num_masked
        
        return loss


class ContrastiveCosineL:
    """
    Improved contrastive cosine loss using spatial proximity + balanced sampling.
    
    FIX #3: Instead of using class labels (which causes class imbalance issues),
    this implementation uses:
    1. Spatial proximity for positive pairs (nearby patches should be similar)
    2. Balanced sampling to ensure both foreground and background are represented
    3. Hard negative mining to focus on difficult cases
    
    This prevents the issue where ~90% background patches all become positives
    for each other, collapsing the embedding space.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        grid_size: Tuple[int, int] = (128, 128),
        proximity_radius: int = 3,  # Patches within this radius are positives
        **kwargs
    ):
        self.temperature = temperature
        self.grid_size = grid_size
        self.proximity_radius = proximity_radius
    
    def __call__(
        self,
        embeddings: tf.Tensor,
        labels: tf.Tensor,
        sample_size: int = 256
    ) -> tf.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings (B, N, D)
            labels: Patch-level class labels (B, N) - used for balanced sampling
            sample_size: Number of patches to sample per batch for efficiency
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = tf.shape(embeddings)[0]
        num_patches = tf.shape(embeddings)[1]
        embed_dim = tf.shape(embeddings)[2]
        H, W = self.grid_size
        
        # Balanced sampling: ensure we have both foreground and background
        # Sample half from foreground, half from background
        foreground_mask = tf.cast(labels > 0, tf.float32)  # (B, N)
        background_mask = 1.0 - foreground_mask
        
        # Sample indices with balanced foreground/background
        half_size = sample_size // 2
        
        # Process each batch item separately for balanced sampling
        sampled_embeddings_list = []
        sampled_positions_list = []
        sampled_labels_list = []
        
        for b in range(tf.shape(embeddings)[0]):
            emb_b = embeddings[b]  # (N, D)
            lab_b = labels[b]  # (N,)
            fg_mask_b = foreground_mask[b]  # (N,)
            bg_mask_b = background_mask[b]  # (N,)
            
            # Get foreground indices
            fg_indices = tf.where(fg_mask_b > 0.5)[:, 0]
            num_fg = tf.shape(fg_indices)[0]
            
            # Get background indices  
            bg_indices = tf.where(bg_mask_b > 0.5)[:, 0]
            num_bg = tf.shape(bg_indices)[0]
            
            # Sample from each (with replacement if needed)
            fg_sample_size = tf.minimum(half_size, num_fg)
            bg_sample_size = sample_size - fg_sample_size
            
            # Random sample from foreground
            if num_fg > 0:
                fg_perm = tf.random.shuffle(fg_indices)
                fg_sampled = fg_perm[:fg_sample_size]
            else:
                fg_sampled = tf.zeros([0], dtype=tf.int64)
            
            # Random sample from background
            bg_perm = tf.random.shuffle(bg_indices)
            bg_sampled = bg_perm[:bg_sample_size]
            
            # Combine
            sampled_indices = tf.concat([fg_sampled, bg_sampled], axis=0)
            sampled_indices = tf.random.shuffle(sampled_indices)[:sample_size]
            
            sampled_emb = tf.gather(emb_b, sampled_indices)
            sampled_lab = tf.gather(lab_b, sampled_indices)
            
            # Convert indices to 2D positions for proximity computation
            positions = tf.stack([
                sampled_indices // W,  # row
                sampled_indices % W,   # col
            ], axis=-1)
            
            sampled_embeddings_list.append(sampled_emb)
            sampled_positions_list.append(positions)
            sampled_labels_list.append(sampled_lab)
        
        # Stack back to batch
        sampled_embeddings = tf.stack(sampled_embeddings_list, axis=0)  # (B, S, D)
        sampled_positions = tf.stack(sampled_positions_list, axis=0)  # (B, S, 2)
        sampled_labels = tf.stack(sampled_labels_list, axis=0)  # (B, S)
        
        # Normalize embeddings
        sampled_embeddings = tf.nn.l2_normalize(sampled_embeddings, axis=-1)
        
        # Compute spatial proximity mask
        # Positive pairs: patches within proximity_radius AND same class
        pos_expand = tf.expand_dims(sampled_positions, 2)  # (B, S, 1, 2)
        pos_compare = tf.expand_dims(sampled_positions, 1)  # (B, 1, S, 2)
        
        # Manhattan distance
        distances = tf.reduce_sum(tf.abs(
            tf.cast(pos_expand, tf.float32) - tf.cast(pos_compare, tf.float32)
        ), axis=-1)  # (B, S, S)
        
        # Spatial proximity mask
        proximity_mask = tf.cast(distances <= self.proximity_radius, tf.float32)
        
        # Same class mask (for refining positives)
        lab_expand = tf.expand_dims(sampled_labels, 2)  # (B, S, 1)
        lab_compare = tf.expand_dims(sampled_labels, 1)  # (B, 1, S)
        same_class_mask = tf.cast(tf.equal(lab_expand, lab_compare), tf.float32)
        
        # Positive pairs: nearby AND same class
        # This ensures text patches are similar to nearby text patches,
        # and background patches are similar to nearby background patches
        positive_mask = proximity_mask * same_class_mask
        
        # Remove self-similarity
        identity_mask = tf.eye(sample_size, batch_shape=[batch_size])
        positive_mask = positive_mask * (1.0 - identity_mask)
        
        # Compute similarity matrix
        similarity = tf.matmul(sampled_embeddings, sampled_embeddings, transpose_b=True)
        similarity = similarity / self.temperature  # (B, S, S)
        
        # InfoNCE loss with hard negative mining
        # For numerical stability, subtract max
        similarity_max = tf.reduce_max(similarity, axis=-1, keepdims=True)
        exp_sim = tf.exp(similarity - similarity_max) * (1.0 - identity_mask)
        log_prob = (similarity - similarity_max) - tf.math.log(
            tf.reduce_sum(exp_sim, axis=-1, keepdims=True) + 1e-6
        )
        
        # Average log probability over positive pairs
        num_positives = tf.reduce_sum(positive_mask, axis=-1)  # (B, S)
        
        # Avoid division by zero
        valid_mask = tf.cast(num_positives > 0, tf.float32)
        num_positives = tf.maximum(num_positives, 1.0)
        
        # Compute loss
        positive_log_prob = tf.reduce_sum(log_prob * positive_mask, axis=-1)
        loss_per_sample = -positive_log_prob / num_positives
        
        # Only average over samples with positives
        loss = tf.reduce_sum(loss_per_sample * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-6)
        
        return loss


class SpatialContrastiveLoss:
    """
    Alternative contrastive loss using purely spatial relationships.
    
    Positive pairs are defined by spatial proximity only, regardless of class.
    This encourages local smoothness in the embedding space.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        grid_size: Tuple[int, int] = (128, 128),
        positive_radius: int = 2,
        negative_radius: int = 8,
        **kwargs
    ):
        self.temperature = temperature
        self.grid_size = grid_size
        self.positive_radius = positive_radius
        self.negative_radius = negative_radius
    
    def __call__(
        self,
        embeddings: tf.Tensor,
        sample_size: int = 256
    ) -> tf.Tensor:
        """
        Args:
            embeddings: Embeddings (B, N, D)
            sample_size: Number of anchor patches to sample
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = tf.shape(embeddings)[0]
        H, W = self.grid_size
        
        # Sample anchor patches
        indices = tf.random.shuffle(tf.range(H * W))[:sample_size]
        
        # Get anchor embeddings
        anchors = tf.gather(embeddings, indices, axis=1)  # (B, S, D)
        anchors = tf.nn.l2_normalize(anchors, axis=-1)
        
        # Convert to 2D positions
        anchor_rows = indices // W
        anchor_cols = indices % W
        
        # For each anchor, find positive (nearby) and negative (far) patches
        losses = []
        
        for i in range(sample_size):
            row, col = anchor_rows[i], anchor_cols[i]
            
            # Positive: patches within positive_radius
            row_range = tf.range(
                tf.maximum(0, row - self.positive_radius),
                tf.minimum(H, row + self.positive_radius + 1)
            )
            col_range = tf.range(
                tf.maximum(0, col - self.positive_radius),
                tf.minimum(W, col + self.positive_radius + 1)
            )
            
            # Create positive indices grid
            rr, cc = tf.meshgrid(row_range, col_range, indexing='ij')
            pos_indices = tf.reshape(rr * W + cc, [-1])
            pos_indices = tf.boolean_mask(
                pos_indices, 
                pos_indices != indices[i]  # Exclude self
            )
            
            if tf.shape(pos_indices)[0] == 0:
                continue
            
            # Get positive embeddings
            pos_emb = tf.gather(embeddings, pos_indices, axis=1)  # (B, num_pos, D)
            pos_emb = tf.nn.l2_normalize(pos_emb, axis=-1)
            
            # Anchor embedding
            anchor = anchors[:, i:i+1, :]  # (B, 1, D)
            
            # Positive similarity
            pos_sim = tf.reduce_sum(anchor * pos_emb, axis=-1)  # (B, num_pos)
            
            # Negative: sample from patches beyond negative_radius
            all_indices = tf.range(H * W)
            all_rows = all_indices // W
            all_cols = all_indices % W
            
            dist = tf.abs(all_rows - row) + tf.abs(all_cols - col)
            neg_mask = dist > self.negative_radius
            neg_indices = tf.boolean_mask(all_indices, neg_mask)
            
            # Sample negatives
            num_neg = tf.minimum(32, tf.shape(neg_indices)[0])
            neg_sample = tf.random.shuffle(neg_indices)[:num_neg]
            
            neg_emb = tf.gather(embeddings, neg_sample, axis=1)  # (B, num_neg, D)
            neg_emb = tf.nn.l2_normalize(neg_emb, axis=-1)
            
            # Negative similarity
            neg_sim = tf.reduce_sum(anchor * neg_emb, axis=-1)  # (B, num_neg)
            
            # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
            pos_exp = tf.exp(pos_sim / self.temperature)
            neg_exp = tf.exp(neg_sim / self.temperature)
            
            # Average over positives
            loss_i = -tf.math.log(
                tf.reduce_mean(pos_exp, axis=-1) / 
                (tf.reduce_mean(pos_exp, axis=-1) + tf.reduce_sum(neg_exp, axis=-1) + 1e-6)
            )
            losses.append(loss_i)
        
        if len(losses) == 0:
            return tf.constant(0.0)
        
        return tf.reduce_mean(tf.stack(losses))


class DifferentiableBinarizationLoss:
    """
    DBNet-style loss for probability, threshold, and binary maps.
    
    Combines:
    - Binary cross-entropy on probability map
    - Binary cross-entropy on binary map (main output)
    - L1 loss on threshold map with distance-transform-based GT
    - Online Hard Example Mining (OHEM) for class imbalance
    
    FIX: Uses distance transform to generate proper threshold ground truth
    instead of self-supervised threshold that just pushes toward 0.5.
    """
    
    def __init__(
        self,
        ohem_ratio: float = 3.0,
        thresh_scale: float = 10.0,
        thresh_min: float = 0.3,
        thresh_max: float = 0.7,
        shrink_ratio: float = 0.4,
        **kwargs
    ):
        self.ohem_ratio = ohem_ratio
        self.thresh_scale = thresh_scale
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.shrink_ratio = shrink_ratio
    
    def _compute_distance_thresh_map(self, gt_mask: tf.Tensor) -> tf.Tensor:
        """
        Compute threshold map based on distance transform.
        
        The threshold should be higher near boundaries and lower in the interior.
        This creates adaptive thresholds that help with boundary sharpening.
        
        Args:
            gt_mask: Binary mask (B, H, W, 1)
            
        Returns:
            thresh_map: Threshold values (B, H, W, 1) in [thresh_min, thresh_max]
        """
        # Compute boundary distance using morphological operations
        # Distance from boundary approximation using iterative erosion
        
        # Dilate to get outer boundary region
        dilated = tf.nn.max_pool2d(gt_mask, ksize=3, strides=1, padding='SAME')
        
        # Erode to get inner boundary region  
        eroded = -tf.nn.max_pool2d(-gt_mask, ksize=3, strides=1, padding='SAME')
        
        # Boundary region = dilated - eroded
        boundary = dilated - eroded
        
        # Compute approximate distance from boundary using iterative erosion
        # More erosion iterations = further from boundary
        distance = tf.zeros_like(gt_mask)
        current = gt_mask
        
        for i in range(1, 6):  # 5 erosion levels
            eroded_i = -tf.nn.max_pool2d(-current, ksize=3, strides=1, padding='SAME')
            # Add distance contribution from this erosion level
            distance = distance + tf.cast(eroded_i > 0.5, tf.float32)
            current = eroded_i
        
        # Normalize distance to [0, 1] (0 = boundary, 1 = far interior)
        max_dist = 5.0
        distance_normalized = tf.clip_by_value(distance / max_dist, 0.0, 1.0)
        
        # Threshold map: higher near boundaries (low distance), lower in interior (high distance)
        # Near boundary: thresh_max, Far from boundary: thresh_min
        thresh_map = self.thresh_max - (self.thresh_max - self.thresh_min) * distance_normalized
        
        # Only apply in text regions and boundary regions
        text_and_boundary = tf.cast(dilated > 0.5, tf.float32)
        thresh_map = thresh_map * text_and_boundary + 0.5 * (1.0 - text_and_boundary)
        
        return thresh_map
    
    def __call__(
        self,
        predictions: Dict[str, tf.Tensor],
        gt_mask: tf.Tensor,
        gt_thresh: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Args:
            predictions: Dict with 'probability', 'threshold', 'binary' maps
            gt_mask: Ground truth binary mask (B, H, W, 1)
            gt_thresh: Optional ground truth threshold (B, H, W, 1)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        prob_map = predictions['probability']
        thresh_map = predictions['threshold']
        binary_map = predictions['binary']
        
        loss_dict = {}
        
        # Ensure proper shape
        if len(gt_mask.shape) == 3:
            gt_mask = tf.expand_dims(gt_mask, -1)
        gt_mask = tf.cast(gt_mask, tf.float32)
        
        # Probability loss with OHEM
        prob_loss = tf.keras.losses.binary_crossentropy(gt_mask, prob_map)
        prob_loss = self._ohem(prob_loss, gt_mask)
        loss_dict['prob_loss'] = prob_loss
        
        # Binary loss (main supervision) with OHEM
        binary_loss = tf.keras.losses.binary_crossentropy(gt_mask, binary_map)
        binary_loss = self._ohem(binary_loss, gt_mask)
        loss_dict['binary_loss'] = binary_loss
        
        # FIX: Threshold loss with distance-transform-based ground truth
        if gt_thresh is not None:
            if len(gt_thresh.shape) == 3:
                gt_thresh = tf.expand_dims(gt_thresh, -1)
        else:
            # Generate threshold GT from distance transform
            gt_thresh = self._compute_distance_thresh_map(gt_mask)
        
        # L1 loss on threshold in text regions and boundary regions
        dilated = tf.nn.max_pool2d(gt_mask, ksize=5, strides=1, padding='SAME')
        supervision_mask = tf.cast(dilated > 0.5, tf.float32)
        thresh_diff = tf.abs(gt_thresh - thresh_map) * supervision_mask
        thresh_loss = tf.reduce_sum(thresh_diff) / (tf.reduce_sum(supervision_mask) + 1e-6)
        loss_dict['thresh_loss'] = thresh_loss
        
        total_loss = prob_loss + binary_loss + self.thresh_scale * thresh_loss
        
        return total_loss, loss_dict
    
    def _ohem(
        self, 
        loss: tf.Tensor, 
        gt_mask: tf.Tensor
    ) -> tf.Tensor:
        """
        Online Hard Example Mining.
        
        Selects top-k hardest negative samples to balance with positives.
        """
        # Flatten for processing
        loss_flat = tf.reshape(loss, [-1])
        gt_flat = tf.reshape(gt_mask, [-1])
        
        pos_mask = tf.cast(gt_flat > 0.5, tf.float32)
        neg_mask = 1.0 - pos_mask
        
        num_pos = tf.reduce_sum(pos_mask)
        num_neg = tf.minimum(num_pos * self.ohem_ratio, tf.reduce_sum(neg_mask))
        num_neg = tf.maximum(num_neg, 1.0)
        
        # Positive loss
        pos_loss = tf.reduce_sum(loss_flat * pos_mask) / (num_pos + 1e-6)
        
        # Top-k hardest negatives
        neg_loss_vals = loss_flat * neg_mask
        neg_loss_vals = tf.where(neg_mask > 0.5, neg_loss_vals, tf.zeros_like(neg_loss_vals) - 1e6)
        
        k = tf.cast(num_neg, tf.int32)
        top_k_neg, _ = tf.math.top_k(neg_loss_vals, k)
        neg_loss = tf.reduce_mean(tf.maximum(top_k_neg, 0.0))
        
        return pos_loss + neg_loss


class AffinityLoss:
    """
    Loss for patch affinity prediction.
    
    Supervises the 8-directional affinity maps:
    - Patches belonging to same text instance should have high affinity
    - Patches from different instances or background should have low affinity
    
    FIX: When instance_map is binary (all text = 1), generates pseudo-instances
    using connected components approximation. Also uses padding instead of roll
    to avoid edge wrap-around artifacts.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (128, 128),
        **kwargs
    ):
        self.grid_size = grid_size
        self.direction_offsets = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
    
    def _generate_pseudo_instances(self, binary_mask: tf.Tensor) -> tf.Tensor:
        """
        Generate pseudo-instance IDs from binary mask using erosion-based separation.
        
        This approximates connected components by:
        1. Eroding the mask to separate touching instances
        2. Using the eroded mask as seeds
        3. Pixels connected to same seed get same ID
        
        Args:
            binary_mask: Binary text mask (B, H, W), 0=background, 1=text
            
        Returns:
            pseudo_instances: (B, H, W) with unique IDs per separated region
        """
        # Expand dims for pooling operations
        mask = tf.expand_dims(tf.cast(binary_mask, tf.float32), -1)
        
        # Strong erosion to separate touching text regions
        eroded = mask
        for _ in range(3):  # 3 erosion iterations
            eroded = -tf.nn.max_pool2d(-eroded, ksize=3, strides=1, padding='SAME')
        
        # Create unique IDs based on eroded seed positions
        # Use spatial coordinates as proxy for instance ID
        B = tf.shape(binary_mask)[0]
        H, W = self.grid_size
        
        # Create coordinate grids
        y_coords = tf.range(H, dtype=tf.float32)
        x_coords = tf.range(W, dtype=tf.float32)
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Encode position as unique ID: y * W + x + 1 (offset by 1 to keep 0 as background)
        position_ids = tf.cast(yy * tf.cast(W, tf.float32) + xx + 1, tf.int32)
        position_ids = tf.expand_dims(position_ids, 0)  # (1, H, W)
        position_ids = tf.tile(position_ids, [B, 1, 1])  # (B, H, W)
        
        # Seeds get their position ID, non-seeds get 0
        eroded_int = tf.cast(eroded[:, :, :, 0] > 0.5, tf.int32)
        seed_ids = position_ids * eroded_int
        
        # Propagate seed IDs to nearby text pixels using dilation
        # This gives each text region the ID of its seed
        seed_ids_float = tf.cast(seed_ids, tf.float32)
        seed_ids_4d = tf.expand_dims(seed_ids_float, -1)
        
        # Iterative dilation to propagate IDs (max pooling propagates IDs)
        propagated = seed_ids_4d
        text_mask_4d = tf.expand_dims(tf.cast(binary_mask, tf.float32), -1)
        
        for _ in range(5):  # Propagate up to 5 pixels
            dilated = tf.nn.max_pool2d(propagated, ksize=3, strides=1, padding='SAME')
            # Only update text pixels that don't have an ID yet
            propagated = tf.where(
                (propagated > 0) | (text_mask_4d < 0.5),
                propagated,
                dilated
            )
        
        pseudo_instances = tf.cast(propagated[:, :, :, 0], tf.int32)
        
        # Ensure background stays 0
        pseudo_instances = pseudo_instances * tf.cast(binary_mask > 0, tf.int32)
        
        return pseudo_instances
    
    def _shift_with_padding(self, tensor: tf.Tensor, dy: int, dx: int) -> tf.Tensor:
        """
        Shift tensor using padding instead of roll to avoid wrap-around.
        
        Args:
            tensor: Input tensor (B, H, W) or (B, H, W, C)
            dy: Vertical shift
            dx: Horizontal shift
            
        Returns:
            Shifted tensor with zeros at boundaries
        """
        # Determine padding
        pad_top = max(0, -dy)
        pad_bottom = max(0, dy)
        pad_left = max(0, -dx)
        pad_right = max(0, dx)
        
        if len(tensor.shape) == 3:
            paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right]]
        else:
            paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
        
        padded = tf.pad(tensor, paddings, mode='CONSTANT', constant_values=0)
        
        # Slice to get shifted version
        H = tf.shape(tensor)[1]
        W = tf.shape(tensor)[2]
        start_h = pad_bottom
        start_w = pad_right
        
        if len(tensor.shape) == 3:
            return padded[:, start_h:start_h+H, start_w:start_w+W]
        else:
            return padded[:, start_h:start_h+H, start_w:start_w+W, :]
    
    def __call__(
        self,
        pred_affinity: tf.Tensor,
        instance_map: tf.Tensor,
        text_mask: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Args:
            pred_affinity: Predicted affinity (B, H, W, 8)
            instance_map: Instance IDs per pixel (B, H, W), 0=background
                         If binary (only 0 and 1), pseudo-instances are generated
            text_mask: Optional binary text mask for weighting
            
        Returns:
            total_loss: Affinity loss
            loss_dict: Per-direction losses
        """
        H, W = self.grid_size
        loss_dict = {}
        
        # Ensure proper shapes
        if len(instance_map.shape) == 4:
            instance_map = instance_map[:, :, :, 0]
        instance_map = tf.cast(instance_map, tf.int32)
        
        # FIX: Check if instance_map is binary (only 0 and 1 values)
        # If so, generate pseudo-instances to provide meaningful supervision
        max_instance = tf.reduce_max(instance_map)
        is_binary = tf.cast(max_instance <= 1, tf.bool)
        
        instance_map = tf.cond(
            is_binary,
            lambda: self._generate_pseudo_instances(tf.cast(instance_map > 0, tf.int32)),
            lambda: instance_map
        )
        
        total_loss = 0.0
        
        for i, (dy, dx) in enumerate(self.direction_offsets):
            # FIX: Use padding-based shift instead of roll
            shifted = self._shift_with_padding(instance_map, dy, dx)
            
            # Ground truth: same instance AND both non-zero (not background)
            same_instance = tf.cast(
                (instance_map == shifted) & (instance_map > 0) & (shifted > 0),
                tf.float32
            )
            
            # Predicted affinity for this direction
            pred = pred_affinity[:, :, :, i]
            
            # Binary cross-entropy
            bce = tf.keras.losses.binary_crossentropy(
                tf.expand_dims(same_instance, -1),
                tf.expand_dims(pred, -1)
            )
            
            # Weight by text mask if provided (focus on text regions)
            if text_mask is not None:
                if len(text_mask.shape) == 4:
                    text_mask_2d = text_mask[:, :, :, 0]
                else:
                    text_mask_2d = text_mask
                text_mask_2d = tf.cast(text_mask_2d, tf.float32)
                
                # Higher weight for text regions
                weight = 1.0 + text_mask_2d * 2.0
                bce = bce * weight
            
            dir_loss = tf.reduce_mean(bce)
            loss_dict[f'affinity_dir_{i}'] = dir_loss
            total_loss = total_loss + dir_loss
        
        # Average over directions
        total_loss = total_loss / 8.0
        
        return total_loss, loss_dict


class PolygonVertexLoss:
    """
    Loss for polygon vertex prediction.
    
    Combines:
    - Vertex classification (is this a vertex location?)
    - Vertex offset regression (offset to exact vertex position)
    - Vertex order regression (position in polygon sequence)
    
    FIX: Adds automatic vertex ground truth generation from segmentation masks
    using corner detection when explicit vertex annotations are not provided.
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        offset_weight: float = 1.0,
        order_weight: float = 0.5,
        corner_threshold: float = 0.3,
        **kwargs
    ):
        self.cls_weight = cls_weight
        self.offset_weight = offset_weight
        self.order_weight = order_weight
        self.corner_threshold = corner_threshold
    
    def generate_vertex_gt_from_segmentation(
        self,
        segmentation: tf.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate vertex ground truth from segmentation mask using corner detection.
        
        Uses Harris corner detection approximation:
        1. Compute boundary from segmentation
        2. Detect corners using gradient-based method
        3. Generate offset maps pointing to nearest corner
        
        Args:
            segmentation: Binary segmentation mask (B, H, W) or (B, H, W, 1)
            target_size: Optional target size for output
            
        Returns:
            vertex_mask: Binary corner locations (B, H, W, 1)
            vertex_offset: Offset to nearest corner (B, H, W, 2)
        """
        # Ensure proper shape
        if len(segmentation.shape) == 3:
            seg = tf.expand_dims(tf.cast(segmentation, tf.float32), -1)
        else:
            seg = tf.cast(segmentation, tf.float32)
        
        B = tf.shape(seg)[0]
        H = tf.shape(seg)[1]
        W = tf.shape(seg)[2]
        
        # Compute boundary using morphological gradient
        dilated = tf.nn.max_pool2d(seg, ksize=3, strides=1, padding='SAME')
        eroded = -tf.nn.max_pool2d(-seg, ksize=3, strides=1, padding='SAME')
        boundary = dilated - eroded
        
        # Compute image gradients on boundary for corner detection
        # Sobel-like gradient computation
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Compute gradients
        Ix = tf.nn.conv2d(boundary, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        Iy = tf.nn.conv2d(boundary, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        
        # Harris corner response: det(M) - k * trace(M)^2
        # M = [[Ix^2, Ix*Iy], [Ix*Iy, Iy^2]]
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # Apply Gaussian smoothing to the products
        gaussian_kernel = tf.constant([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=tf.float32) / 16.0
        gaussian_kernel = tf.reshape(gaussian_kernel, [3, 3, 1, 1])
        
        Sxx = tf.nn.conv2d(Ixx, gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')
        Syy = tf.nn.conv2d(Iyy, gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')
        Sxy = tf.nn.conv2d(Ixy, gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # Harris response
        det = Sxx * Syy - Sxy * Sxy
        trace = Sxx + Syy
        k = 0.04
        harris = det - k * trace * trace
        
        # Normalize and threshold
        harris_max = tf.reduce_max(harris, axis=[1, 2, 3], keepdims=True)
        harris_normalized = harris / (harris_max + 1e-6)
        
        # Non-maximum suppression using max pooling
        harris_max_pool = tf.nn.max_pool2d(harris_normalized, ksize=5, strides=1, padding='SAME')
        is_local_max = tf.cast(tf.abs(harris_normalized - harris_max_pool) < 1e-6, tf.float32)
        
        # Threshold to get corner mask
        vertex_mask = tf.cast(harris_normalized > self.corner_threshold, tf.float32) * is_local_max
        vertex_mask = vertex_mask * boundary  # Only corners on boundaries
        
        # Generate offset maps (simplified: point toward boundary center)
        # For each pixel, compute offset to nearest vertex
        # This is expensive, so we use an approximation
        
        # Create coordinate grids
        y_coords = tf.cast(tf.range(H), tf.float32)
        x_coords = tf.cast(tf.range(W), tf.float32)
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')
        yy = tf.reshape(yy, [1, H, W, 1])
        xx = tf.reshape(xx, [1, H, W, 1])
        
        # Compute weighted centroid of nearby vertices for each pixel
        # Use the vertex mask weighted by distance
        vertex_y = yy * vertex_mask
        vertex_x = xx * vertex_mask
        
        # Dilate to propagate vertex positions
        for _ in range(10):
            vertex_y_dilated = tf.nn.max_pool2d(vertex_y, ksize=3, strides=1, padding='SAME')
            vertex_x_dilated = tf.nn.max_pool2d(vertex_x, ksize=3, strides=1, padding='SAME')
            vertex_y = tf.where(vertex_y > 0, vertex_y, vertex_y_dilated)
            vertex_x = tf.where(vertex_x > 0, vertex_x, vertex_x_dilated)
        
        # Compute offset (target - current position)
        offset_y = vertex_y - yy
        offset_x = vertex_x - xx
        
        # Normalize offsets
        offset_magnitude = tf.sqrt(offset_y**2 + offset_x**2 + 1e-6)
        max_offset = 10.0  # Clip to reasonable range
        offset_y = tf.clip_by_value(offset_y / max_offset, -1.0, 1.0)
        offset_x = tf.clip_by_value(offset_x / max_offset, -1.0, 1.0)
        
        vertex_offset = tf.concat([offset_y, offset_x], axis=-1)
        
        # Only provide offset supervision on boundary regions
        vertex_offset = vertex_offset * boundary
        
        # Resize to target size if specified
        if target_size is not None:
            vertex_mask = tf.image.resize(vertex_mask, target_size, method='nearest')
            vertex_offset = tf.image.resize(vertex_offset, target_size, method='bilinear')
        
        return vertex_mask, vertex_offset
    
    def __call__(
        self,
        predictions: Dict[str, tf.Tensor],
        gt_vertex_mask: Optional[tf.Tensor] = None,
        gt_vertex_offset: Optional[tf.Tensor] = None,
        gt_vertex_order: Optional[tf.Tensor] = None,
        segmentation: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Args:
            predictions: Dict with vertex predictions
            gt_vertex_mask: Binary mask of vertex locations (B, H, W) - optional
            gt_vertex_offset: Offset to vertex (B, H, W, 2) - optional
            gt_vertex_order: Vertex order in polygon (B, H, W) - optional
            segmentation: Segmentation mask for auto GT generation if gt_vertex_* not provided
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual components
        """
        loss_dict = {}
        
        vertex_logits = predictions['vertex_logits']  # (B, H, W, 2)
        vertex_offset = predictions['vertex_offset']  # (B, H, W, 2)
        vertex_order = predictions.get('vertex_order')  # (B, H, W, 1) - optional
        
        # FIX: Auto-generate vertex GT from segmentation if not provided
        if gt_vertex_mask is None or gt_vertex_offset is None:
            if segmentation is not None:
                target_size = (tf.shape(vertex_logits)[1], tf.shape(vertex_logits)[2])
                gt_vertex_mask, gt_vertex_offset = self.generate_vertex_gt_from_segmentation(
                    segmentation, target_size=None  # Keep original size
                )
                # Resize to match prediction size
                gt_vertex_mask = tf.image.resize(
                    gt_vertex_mask, 
                    [tf.shape(vertex_logits)[1], tf.shape(vertex_logits)[2]],
                    method='nearest'
                )
                gt_vertex_offset = tf.image.resize(
                    gt_vertex_offset,
                    [tf.shape(vertex_logits)[1], tf.shape(vertex_logits)[2]],
                    method='bilinear'
                )
            else:
                # Fallback: use boundary as vertex proxy (better than nothing)
                raise ValueError("Either gt_vertex_mask/gt_vertex_offset or segmentation must be provided")
        
        # Ensure proper shapes
        if len(gt_vertex_mask.shape) == 3:
            gt_vertex_mask = tf.expand_dims(gt_vertex_mask, -1)
        gt_vertex_mask = tf.cast(gt_vertex_mask, tf.float32)
        
        # Classification loss (focal loss for imbalance)
        gt_cls = tf.concat([1.0 - gt_vertex_mask, gt_vertex_mask], axis=-1)
        cls_loss = tf.keras.losses.categorical_crossentropy(
            gt_cls, tf.nn.softmax(vertex_logits, axis=-1)
        )
        cls_loss = tf.reduce_mean(cls_loss)
        loss_dict['vertex_cls_loss'] = cls_loss
        
        # Offset regression (only on vertex regions and nearby boundary)
        # Dilate vertex mask to supervise offset in a small neighborhood
        vertex_region = tf.nn.max_pool2d(gt_vertex_mask, ksize=5, strides=1, padding='SAME')
        offset_diff = tf.abs(gt_vertex_offset - vertex_offset)
        offset_loss = tf.reduce_sum(offset_diff * vertex_region) / (
            tf.reduce_sum(vertex_region) * 2.0 + 1e-6
        )
        loss_dict['vertex_offset_loss'] = offset_loss
        
        # Order regression (optional)
        if gt_vertex_order is not None and vertex_order is not None:
            if len(gt_vertex_order.shape) == 3:
                gt_vertex_order = tf.expand_dims(gt_vertex_order, -1)
            gt_vertex_order = tf.cast(gt_vertex_order, tf.float32)
            
            order_diff = tf.abs(gt_vertex_order - vertex_order)
            order_loss = tf.reduce_sum(order_diff * gt_vertex_mask) / (
                tf.reduce_sum(gt_vertex_mask) + 1e-6
            )
            loss_dict['vertex_order_loss'] = order_loss
        else:
            order_loss = 0.0
        
        total_loss = (
            self.cls_weight * cls_loss +
            self.offset_weight * offset_loss +
            self.order_weight * order_loss
        )
        
        return total_loss, loss_dict


class PatchFormerLoss:
    """
    Combined loss function for PatchFormer training.
    
    Combines:
    - Segmentation loss (main)
    - Token classification loss
    - Reconstruction loss
    - Contrastive loss
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        training_config: TrainingConfig,
    ):
        self.config = config
        self.training_config = training_config
        
        # Initialize loss functions
        self.segmentation_loss = SegmentationLoss()
        self.token_loss = TokenClassificationLoss(boundary_weight=3.0)
        self.reconstruction_loss = ReconstructionLoss(patch_size=config.patch_size)
        
        # Updated contrastive loss with spatial proximity + balanced sampling
        self.contrastive_loss = ContrastiveCosineL(
            temperature=config.temperature,
            grid_size=config.grid_size,
            proximity_radius=3,  # Patches within 3 cells are positives
        )
    
    def __call__(
        self,
        model_outputs: Dict[str, tf.Tensor],
        ground_truth: Dict[str, tf.Tensor],
        current_epoch: int = 0,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute total loss and individual loss components.
        
        Args:
            model_outputs: Dict with model predictions
            ground_truth: Dict with:
                - 'segmentation': (B, H, W) segmentation mask
                - 'token_labels': (B, N) token classification labels
                - 'images': (B, H, W, 3) original images for reconstruction
            current_epoch: Current training epoch (for loss scheduling)
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with individual losses
        """
        loss_dict = {}
        
        # Get loss weights (potentially adjusted by epoch)
        seg_weight = self.training_config.loss_segmentation
        token_weight = self.training_config.loss_token
        
        # Decay reconstruction weight after phase 1
        if current_epoch >= self.training_config.phase1_epochs:
            recon_weight = self.training_config.phase2_reconstruction_weight
        else:
            recon_weight = self.training_config.loss_reconstruction
        
        contrast_weight = self.training_config.loss_contrastive
        
        # Segmentation loss (main output)
        seg_loss = self.segmentation_loss(
            ground_truth['segmentation'],
            model_outputs['main']  # Decoder output
        )
        loss_dict['segmentation'] = seg_loss
        
        # Token classification loss
        if 'token_logits' in model_outputs and 'token_labels' in ground_truth:
            token_loss = self.token_loss(
                ground_truth['token_labels'],
                model_outputs['token_logits']
            )
            loss_dict['token'] = token_loss
        else:
            token_loss = 0.0
            token_weight = 0.0
        
        # Reconstruction loss (training only)
        if 'reconstruction' in model_outputs:
            recon_loss = self.reconstruction_loss(
                ground_truth['images'],
                model_outputs['reconstruction'],
                mask=model_outputs.get('mask')
            )
            loss_dict['reconstruction'] = recon_loss
        else:
            recon_loss = 0.0
            recon_weight = 0.0
        
        # Contrastive loss (training only)
        if 'contrastive_embeddings' in model_outputs:
            # Use segmentation mask to generate patch labels
            patch_labels = self._get_patch_labels(ground_truth['segmentation'])
            contrast_loss = self.contrastive_loss(
                model_outputs['contrastive_embeddings'],
                patch_labels
            )
            loss_dict['contrastive'] = contrast_loss
        else:
            contrast_loss = 0.0
            contrast_weight = 0.0
        
        # Auxiliary losses (if using multi-scale decoder)
        aux_weight = 0.3
        if 'aux_256' in model_outputs:
            # Downsample ground truth
            gt_256 = tf.image.resize(
                tf.expand_dims(tf.cast(ground_truth['segmentation'], tf.float32), -1),
                [256, 256],
                method='nearest'
            )
            gt_256 = tf.squeeze(gt_256, -1)
            aux_loss_256 = self.segmentation_loss(gt_256, model_outputs['aux_256'])
            loss_dict['aux_256'] = aux_loss_256
            seg_loss = seg_loss + aux_weight * aux_loss_256
        
        if 'aux_512' in model_outputs:
            gt_512 = tf.image.resize(
                tf.expand_dims(tf.cast(ground_truth['segmentation'], tf.float32), -1),
                [512, 512],
                method='nearest'
            )
            gt_512 = tf.squeeze(gt_512, -1)
            aux_loss_512 = self.segmentation_loss(gt_512, model_outputs['aux_512'])
            loss_dict['aux_512'] = aux_loss_512
            seg_loss = seg_loss + aux_weight * 0.5 * aux_loss_512
        
        # Combine losses
        total_loss = (
            seg_weight * seg_loss +
            token_weight * token_loss +
            recon_weight * recon_loss +
            contrast_weight * contrast_loss
        )
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def _get_patch_labels(self, segmentation_mask: tf.Tensor) -> tf.Tensor:
        """
        Convert pixel-level segmentation mask to patch-level labels.
        
        Args:
            segmentation_mask: (B, H, W) with class indices
            
        Returns:
            patch_labels: (B, num_patches)
        """
        batch_size = tf.shape(segmentation_mask)[0]
        
        # Resize to patch grid using mode (most common class)
        # For simplicity, use nearest neighbor
        patch_labels = tf.image.resize(
            tf.expand_dims(tf.cast(segmentation_mask, tf.float32), -1),
            self.config.grid_size,
            method='nearest'
        )
        patch_labels = tf.squeeze(patch_labels, -1)
        patch_labels = tf.cast(patch_labels, tf.int32)
        
        # Flatten
        patch_labels = tf.reshape(patch_labels, [batch_size, -1])
        
        return patch_labels


def generate_token_labels(
    segmentation_mask: tf.Tensor,
    grid_size: Tuple[int, int] = (128, 128)
) -> tf.Tensor:
    """
    Generate token classification labels from segmentation mask.
    
    Labels:
        0: Interior (fully inside object)
        1: Boundary (adjacent to object edge)
        2: Background (not in or near object)
    
    Args:
        segmentation_mask: (B, H, W) with class indices (0 = background, >0 = foreground)
        grid_size: Size of the patch grid
        
    Returns:
        token_labels: (B, num_patches) with values {0, 1, 2}
    """
    batch_size = tf.shape(segmentation_mask)[0]
    
    # Resize to grid size
    mask_resized = tf.image.resize(
        tf.expand_dims(tf.cast(segmentation_mask, tf.float32), -1),
        grid_size,
        method='nearest'
    )
    mask_resized = tf.squeeze(mask_resized, -1)
    
    # Create binary foreground mask
    foreground = tf.cast(mask_resized > 0, tf.float32)
    
    # Compute boundary using morphological operations
    # Dilation - Erosion = Boundary
    kernel = tf.ones((3, 3, 1), dtype=tf.float32)
    
    foreground_4d = tf.expand_dims(foreground, -1)
    
    # Dilation (max pool)
    dilated = tf.nn.max_pool2d(foreground_4d, ksize=3, strides=1, padding='SAME')
    
    # Erosion (negative dilation of negative)
    eroded = -tf.nn.max_pool2d(-foreground_4d, ksize=3, strides=1, padding='SAME')
    
    # Boundary = dilated AND NOT eroded (for foreground pixels)
    boundary = tf.cast(dilated > 0.5, tf.float32) * (1.0 - tf.cast(eroded > 0.5, tf.float32))
    boundary = tf.squeeze(boundary, -1)
    
    # Assign labels
    # 0: Interior = foreground AND NOT boundary
    # 1: Boundary = boundary region
    # 2: Background = NOT foreground
    
    interior = foreground * (1.0 - boundary)
    
    # Create label tensor: 2 (background) - 2*interior (makes interior 0) + boundary (makes boundary 1)
    token_labels = tf.cast(
        2.0 * (1.0 - foreground) +  # Background = 2
        0.0 * interior +  # Interior = 0
        1.0 * boundary * foreground,  # Boundary (foreground part) = 1
        tf.int32
    )
    
    # More precise: start with background (2), override with interior (0), then boundary (1)
    token_labels = tf.where(foreground > 0.5, 0, 2)  # Interior or background
    token_labels = tf.where(boundary > 0.5, 1, token_labels)  # Boundary overrides
    
    # Flatten
    token_labels = tf.reshape(token_labels, [batch_size, -1])
    
    return token_labels


class PatchFormerV2Loss:
    """
    Combined loss function for PatchFormer v2.0 training.
    
    UNIFIED PIPELINE SUPERVISION:
    All intermediate outputs (affinity, vertex, DB) are supervised,
    but they all feed into the single final output.
    
    Loss components:
    - Final segmentation loss (on refined output)
    - Initial segmentation loss (intermediate supervision)
    - Affinity loss (guides feature propagation)
    - Vertex loss (guides boundary sharpening)
    - DB loss (guides final sharpening)
    - Token classification (encoder supervision)
    - Reconstruction (encoder supervision)
    - Contrastive (embedding structure)
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        training_config: TrainingConfig,
        use_db_loss: bool = True,
        use_affinity_loss: bool = True,
        use_vertex_loss: bool = True,
    ):
        self.config = config
        self.training_config = training_config
        self.use_db_loss = use_db_loss
        self.use_affinity_loss = use_affinity_loss
        self.use_vertex_loss = use_vertex_loss
        
        # Main segmentation loss (for final and initial outputs)
        self.segmentation_loss = SegmentationLoss()
        
        # Encoder supervision
        self.token_loss = TokenClassificationLoss(boundary_weight=3.0)
        self.reconstruction_loss = ReconstructionLoss(patch_size=config.patch_size)
        self.contrastive_loss = ContrastiveCosineL(
            temperature=config.temperature,
            grid_size=config.grid_size,
            proximity_radius=3,
        )
        
        # v2.0: Refinement pipeline losses
        if use_db_loss:
            self.db_loss = DifferentiableBinarizationLoss(
                ohem_ratio=3.0,
                thresh_scale=10.0
            )
        
        if use_affinity_loss:
            # Affinity loss operates at decoder resolution (1024x1024)
            self.affinity_loss_fn = AffinityLoss(grid_size=(1024, 1024))
        
        if use_vertex_loss:
            self.vertex_loss_fn = PolygonVertexLoss()
    
    def __call__(
        self,
        model_outputs: Dict[str, tf.Tensor],
        ground_truth: Dict[str, tf.Tensor],
        current_epoch: int = 0,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute total loss with unified pipeline supervision.
        
        Args:
            model_outputs: Dict with model predictions including:
                - 'final': Main refined output
                - 'initial_seg': Pre-refinement segmentation
                - 'affinity': Affinity maps
                - 'vertex_prob', 'vertex_logits': Vertex predictions
                - 'db_outputs': DB intermediate outputs
            ground_truth: Dict with:
                - 'segmentation': (B, H, W) segmentation mask
                - 'token_labels': (B, N) token classification labels (optional)
                - 'images': (B, H, W, 3) original images
                - 'instance_map': (B, H, W) instance IDs (optional)
            current_epoch: Current training epoch
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with individual losses
        """
        loss_dict = {}
        
        # === Weight configuration ===
        final_weight = 1.0  # Main final output
        initial_weight = 0.5  # Intermediate supervision
        db_weight = 1.0 if self.use_db_loss else 0.0
        affinity_weight = 0.5 if self.use_affinity_loss else 0.0
        vertex_weight = 0.3 if self.use_vertex_loss else 0.0
        token_weight = self.training_config.loss_token
        
        # Decay reconstruction weight after phase 1
        if current_epoch >= self.training_config.phase1_epochs:
            recon_weight = self.training_config.phase2_reconstruction_weight
        else:
            recon_weight = self.training_config.loss_reconstruction
        
        contrast_weight = self.training_config.loss_contrastive
        
        # === MAIN: Final segmentation loss ===
        # This is the output that matters for inference
        final_loss = self._compute_binary_seg_loss(
            ground_truth['segmentation'],
            model_outputs['final']
        )
        loss_dict['final'] = final_loss
        
        # === Initial segmentation loss (intermediate supervision) ===
        initial_loss = 0.0
        if 'initial_seg' in model_outputs:
            initial_loss = self.segmentation_loss(
                ground_truth['segmentation'],
                model_outputs['initial_seg']
            )
            loss_dict['initial'] = initial_loss
        
        # === v2.0: Differentiable Binarization loss ===
        db_loss_total = 0.0
        if self.use_db_loss and 'db_outputs' in model_outputs:
            db_loss, db_loss_dict = self.db_loss(
                model_outputs['db_outputs'],
                ground_truth['segmentation']
            )
            db_loss_total = db_loss
            loss_dict['db_prob'] = db_loss_dict.get('prob_loss', 0.0)
            loss_dict['db_binary'] = db_loss_dict.get('binary_loss', 0.0)
            loss_dict['db_thresh'] = db_loss_dict.get('thresh_loss', 0.0)
        
        # === v2.0: Affinity loss ===
        affinity_loss_total = 0.0
        if self.use_affinity_loss and 'affinity' in model_outputs:
            # Get instance map (or fall back to segmentation)
            if 'instance_map' in ground_truth:
                instance_map = ground_truth['instance_map']
            else:
                instance_map = ground_truth['segmentation']
            
            # Affinity is at full resolution (1024x1024)
            instance_map_full = tf.cast(instance_map, tf.int32)
            
            affinity_loss, _ = self.affinity_loss_fn(
                model_outputs['affinity'],
                instance_map_full,
                text_mask=tf.cast(instance_map_full > 0, tf.float32)
            )
            affinity_loss_total = affinity_loss
            loss_dict['affinity'] = affinity_loss
        
        # === v2.0: Vertex loss ===
        vertex_loss_total = 0.0
        if self.use_vertex_loss and 'vertex_logits' in model_outputs:
            # Generate vertex ground truth from segmentation boundary
            gt_vertex_mask = self._generate_vertex_mask(ground_truth['segmentation'])
            gt_vertex_offset = tf.zeros_like(model_outputs['vertex_offset'])  # Simplified
            
            vertex_preds = {
                'vertex_logits': model_outputs['vertex_logits'],
                'vertex_offset': model_outputs['vertex_offset'],
                'vertex_order': model_outputs.get('vertex_order', 
                    tf.zeros_like(model_outputs['vertex_prob']))
            }
            
            vertex_loss, _ = self.vertex_loss_fn(
                vertex_preds,
                gt_vertex_mask,
                gt_vertex_offset
            )
            vertex_loss_total = vertex_loss
            loss_dict['vertex'] = vertex_loss
        
        # === Token classification loss (encoder supervision) ===
        token_loss = 0.0
        if 'token_logits' in model_outputs and 'token_labels' in ground_truth:
            token_loss = self.token_loss(
                ground_truth['token_labels'],
                model_outputs['token_logits']
            )
            loss_dict['token'] = token_loss
        else:
            token_weight = 0.0
        
        # === Reconstruction loss (encoder supervision) ===
        recon_loss = 0.0
        if 'reconstruction' in model_outputs:
            recon_loss = self.reconstruction_loss(
                ground_truth['images'],
                model_outputs['reconstruction'],
                mask=model_outputs.get('mask')
            )
            loss_dict['reconstruction'] = recon_loss
        else:
            recon_weight = 0.0
        
        # === Contrastive loss (encoder supervision) ===
        contrast_loss = 0.0
        if 'contrastive_embeddings' in model_outputs:
            patch_labels = self._get_patch_labels(ground_truth['segmentation'])
            contrast_loss = self.contrastive_loss(
                model_outputs['contrastive_embeddings'],
                patch_labels
            )
            loss_dict['contrastive'] = contrast_loss
        else:
            contrast_weight = 0.0
        
        # === Auxiliary losses (multi-scale deep supervision) ===
        aux_weight = 0.3
        aux_loss = 0.0
        if 'aux_256' in model_outputs:
            gt_256 = tf.image.resize(
                tf.expand_dims(tf.cast(ground_truth['segmentation'], tf.float32), -1),
                [256, 256], method='nearest'
            )
            gt_256 = tf.squeeze(gt_256, -1)
            aux_256 = self.segmentation_loss(gt_256, model_outputs['aux_256'])
            loss_dict['aux_256'] = aux_256
            aux_loss += aux_256
        
        if 'aux_512' in model_outputs:
            gt_512 = tf.image.resize(
                tf.expand_dims(tf.cast(ground_truth['segmentation'], tf.float32), -1),
                [512, 512], method='nearest'
            )
            gt_512 = tf.squeeze(gt_512, -1)
            aux_512 = self.segmentation_loss(gt_512, model_outputs['aux_512'])
            loss_dict['aux_512'] = aux_512
            aux_loss += 0.5 * aux_512
        
        # === Combine all losses ===
        total_loss = (
            final_weight * final_loss +
            initial_weight * initial_loss +
            db_weight * db_loss_total +
            affinity_weight * affinity_loss_total +
            vertex_weight * vertex_loss_total +
            token_weight * token_loss +
            recon_weight * recon_loss +
            contrast_weight * contrast_loss +
            aux_weight * aux_loss
        )
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def _compute_binary_seg_loss(
        self, 
        gt_mask: tf.Tensor, 
        pred: tf.Tensor
    ) -> tf.Tensor:
        """Compute segmentation loss for binary output."""
        # Handle shape
        if len(gt_mask.shape) == 3:
            gt_mask = tf.expand_dims(gt_mask, -1)
        gt_mask = tf.cast(gt_mask, tf.float32)
        
        # Binary cross-entropy + Dice
        bce = tf.keras.losses.binary_crossentropy(gt_mask, pred)
        bce = tf.reduce_mean(bce)
        
        # Dice loss
        intersection = tf.reduce_sum(gt_mask * pred)
        union = tf.reduce_sum(gt_mask) + tf.reduce_sum(pred)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        
        return bce + dice
    
    def _generate_vertex_mask(self, segmentation: tf.Tensor) -> tf.Tensor:
        """Generate approximate vertex mask from segmentation boundaries."""
        # Convert to float and add channel dim
        if len(segmentation.shape) == 3:
            seg = tf.expand_dims(tf.cast(segmentation, tf.float32), -1)
        else:
            seg = tf.cast(segmentation, tf.float32)
        
        # Find boundaries using morphological operations
        dilated = tf.nn.max_pool2d(seg, ksize=3, strides=1, padding='SAME')
        eroded = -tf.nn.max_pool2d(-seg, ksize=3, strides=1, padding='SAME')
        boundary = tf.cast(dilated > 0.5, tf.float32) * (1.0 - tf.cast(eroded > 0.5, tf.float32))
        
        # Find corners (high curvature points) using Laplacian-like filter
        # Simplified: use boundary as vertex mask
        vertex_mask = boundary
        
        return tf.squeeze(vertex_mask, -1)
    
    def _get_patch_labels(self, segmentation_mask: tf.Tensor) -> tf.Tensor:
        """Convert pixel-level segmentation to patch-level labels."""
        batch_size = tf.shape(segmentation_mask)[0]
        
        patch_labels = tf.image.resize(
            tf.expand_dims(tf.cast(segmentation_mask, tf.float32), -1),
            self.config.grid_size,
            method='nearest'
        )
        patch_labels = tf.squeeze(patch_labels, -1)
        patch_labels = tf.cast(patch_labels, tf.int32)
        patch_labels = tf.reshape(patch_labels, [batch_size, -1])
        
        return patch_labels
