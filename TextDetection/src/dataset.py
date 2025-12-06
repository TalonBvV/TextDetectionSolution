"""
Data pipeline for PatchFormer training.

Supports generic semantic segmentation datasets with augmentation.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Callable, List, Dict
from pathlib import Path

from .config import PatchFormerConfig
from .losses import generate_token_labels


class SegmentationDataset:
    """
    Generic segmentation dataset loader.
    
    Expects data in format:
    - images: (H, W, 3) uint8 or float32
    - masks: (H, W) int32 with class indices
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        augment: bool = True,
        include_token_labels: bool = True,
    ):
        self.config = config
        self.augment = augment
        self.include_token_labels = include_token_labels
        
        # Augmentation functions
        self.augmentation_fn = self._build_augmentation_pipeline() if augment else None
    
    def _build_augmentation_pipeline(self) -> Callable:
        """Build augmentation pipeline."""
        
        def augment_fn(image, mask):
            # Random horizontal flip
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(tf.expand_dims(mask, -1))
                mask = tf.squeeze(mask, -1)
            
            # Random rotation (0, 90, 180, 270 degrees)
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k)
            mask = tf.image.rot90(tf.expand_dims(mask, -1), k)
            mask = tf.squeeze(mask, -1)
            
            # Color jitter (only on image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            # Random scale and crop
            scale = tf.random.uniform((), minval=0.8, maxval=1.2)
            new_size = tf.cast(
                tf.cast(tf.shape(image)[:2], tf.float32) * scale,
                tf.int32
            )
            
            image = tf.image.resize(image, new_size)
            mask = tf.image.resize(
                tf.expand_dims(tf.cast(mask, tf.float32), -1),
                new_size,
                method='nearest'
            )
            mask = tf.squeeze(tf.cast(mask, tf.int32), -1)
            
            # Random crop to target size
            combined = tf.concat([
                image,
                tf.cast(tf.expand_dims(mask, -1), tf.float32)
            ], axis=-1)
            
            target_h, target_w = self.config.input_size
            combined = tf.image.random_crop(combined, [target_h, target_w, 4])
            
            image = combined[:, :, :3]
            mask = tf.cast(combined[:, :, 3], tf.int32)
            
            return image, mask
        
        return augment_fn
    
    def preprocess(
        self,
        image: tf.Tensor,
        mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Preprocess image and mask pair.
        
        Args:
            image: Input image (H, W, 3)
            mask: Segmentation mask (H, W)
            
        Returns:
            image: Preprocessed image
            targets: Dictionary with mask and optional token labels
        """
        # Normalize image to [0, 1]
        if image.dtype == tf.uint8:
            image = tf.cast(image, tf.float32) / 255.0
        
        # Ensure correct data types
        mask = tf.cast(mask, tf.int32)
        
        # Resize to target size
        target_h, target_w = self.config.input_size
        
        if self.augment and self.augmentation_fn is not None:
            image, mask = self.augmentation_fn(image, mask)
        else:
            image = tf.image.resize(image, [target_h, target_w])
            mask = tf.image.resize(
                tf.expand_dims(tf.cast(mask, tf.float32), -1),
                [target_h, target_w],
                method='nearest'
            )
            mask = tf.squeeze(tf.cast(mask, tf.int32), -1)
        
        # Prepare targets
        targets = {'segmentation': mask}
        
        if self.include_token_labels:
            # Generate token classification labels
            token_labels = generate_token_labels(
                tf.expand_dims(mask, 0),
                self.config.grid_size
            )
            targets['token_labels'] = tf.squeeze(token_labels, 0)
        
        return image, targets
    
    def create_dataset(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        batch_size: int = 4,
        shuffle: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE,
    ) -> tf.data.Dataset:
        """
        Create tf.data.Dataset from file paths.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_parallel_calls: Parallelism for preprocessing
            
        Returns:
            tf.data.Dataset
        """
        def load_and_preprocess(image_path, mask_path):
            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
            
            # Load mask (assuming PNG with class indices)
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.squeeze(mask, -1)
            
            return self.preprocess(image, mask)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        
        dataset = dataset.map(
            load_and_preprocess,
            num_parallel_calls=num_parallel_calls
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_dataset_from_arrays(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        batch_size: int = 4,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Create dataset from numpy arrays.
        
        Args:
            images: (N, H, W, 3) images
            masks: (N, H, W) segmentation masks
            batch_size: Batch size
            shuffle: Whether to shuffle
            
        Returns:
            tf.data.Dataset
        """
        def preprocess_wrapper(image, mask):
            return self.preprocess(image, mask)
        
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        
        dataset = dataset.map(
            preprocess_wrapper,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class TextDetectionDataset(SegmentationDataset):
    """
    Specialized dataset for text detection.
    
    Handles text detection specific preprocessing:
    - Binary masks (text/background)
    - Word-level annotations
    - Polygon-based ground truth
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        augment: bool = True,
        polygon_shrink_ratio: float = 0.4,
    ):
        # Ensure binary classification for text detection
        config.num_classes = 2
        super().__init__(config, augment, include_token_labels=True)
        
        self.polygon_shrink_ratio = polygon_shrink_ratio
    
    def create_mask_from_polygons(
        self,
        polygons: List[np.ndarray],
        image_size: Tuple[int, int],
        shrink: bool = True,
    ) -> np.ndarray:
        """
        Create segmentation mask from polygon annotations.
        
        Args:
            polygons: List of (N, 2) polygon arrays
            image_size: (height, width)
            shrink: Whether to shrink polygons for better boundary detection
            
        Returns:
            Binary mask (H, W)
        """
        import cv2
        
        mask = np.zeros(image_size, dtype=np.uint8)
        
        for poly in polygons:
            poly = np.array(poly, dtype=np.int32)
            
            if shrink and self.polygon_shrink_ratio < 1.0:
                # Shrink polygon for text kernel
                poly = self._shrink_polygon(poly, self.polygon_shrink_ratio)
            
            cv2.fillPoly(mask, [poly], 1)
        
        return mask
    
    def _shrink_polygon(
        self,
        polygon: np.ndarray,
        ratio: float
    ) -> np.ndarray:
        """Shrink polygon towards centroid."""
        centroid = polygon.mean(axis=0)
        shrunk = centroid + (polygon - centroid) * ratio
        return shrunk.astype(np.int32)


def create_dummy_dataset(
    config: PatchFormerConfig,
    num_samples: int = 16,
    batch_size: int = 4,
    include_instances: bool = True,
) -> tf.data.Dataset:
    """
    Create dummy dataset for testing.
    
    Args:
        config: Model configuration
        num_samples: Number of dummy samples
        batch_size: Batch size
        include_instances: Whether to include instance maps
        
    Returns:
        tf.data.Dataset with random data
    """
    h, w = config.input_size
    
    # Generate random images and masks
    images = np.random.rand(num_samples, h, w, 3).astype(np.float32)
    masks = np.random.randint(0, config.num_classes, (num_samples, h, w)).astype(np.int32)
    
    if include_instances:
        # Generate instance maps (each connected region gets unique ID)
        # For dummy data, just use random labels
        instance_maps = np.zeros((num_samples, h, w), dtype=np.int32)
        for i in range(num_samples):
            # Create random instances where mask > 0
            instance_id = 1
            for y in range(0, h, 64):
                for x in range(0, w, 64):
                    region = masks[i, y:y+64, x:x+64]
                    if region.any():
                        instance_maps[i, y:y+64, x:x+64] = np.where(
                            region > 0, instance_id, 0
                        )
                        instance_id += 1
        
        def dummy_generator():
            for img, mask, inst in zip(images, masks, instance_maps):
                targets = {
                    'segmentation': mask,
                    'instance_map': inst,
                    'token_labels': generate_token_labels(
                        tf.expand_dims(tf.constant(mask), 0),
                        config.grid_size
                    )[0].numpy()
                }
                yield img, targets
        
        dataset = tf.data.Dataset.from_generator(
            dummy_generator,
            output_signature=(
                tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
                {
                    'segmentation': tf.TensorSpec(shape=(h, w), dtype=tf.int32),
                    'instance_map': tf.TensorSpec(shape=(h, w), dtype=tf.int32),
                    'token_labels': tf.TensorSpec(shape=(config.num_patches,), dtype=tf.int32),
                }
            )
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    else:
        dataset_loader = SegmentationDataset(config, augment=False)
        return dataset_loader.create_dataset_from_arrays(images, masks, batch_size)


# Standard dataset loaders for common text detection datasets

def load_icdar_dataset(
    root_path: str,
    config: PatchFormerConfig,
    split: str = "train",
) -> tf.data.Dataset:
    """
    Load ICDAR text detection dataset.
    
    Expected structure:
    root_path/
        train/
            images/
            gt/
        test/
            images/
            gt/
    """
    root = Path(root_path)
    split_path = root / split
    
    image_dir = split_path / "images"
    gt_dir = split_path / "gt"
    
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    # Parse ground truth files and create masks
    # This is a placeholder - actual implementation depends on ICDAR format version
    
    raise NotImplementedError("ICDAR loader requires format-specific implementation")


def load_synthtext_dataset(
    root_path: str,
    config: PatchFormerConfig,
    max_samples: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Load SynthText dataset.
    
    Expected to have gt.mat with annotations.
    """
    raise NotImplementedError("SynthText loader requires scipy for .mat files")


# ============================================================================
# Unified Dataset Loader (for generated datasets)
# ============================================================================

class UnifiedTextDetectionDataset(SegmentationDataset):
    """
    Dataset loader for unified text detection dataset format.
    
    Loads the complete dataset structure:
    - images/: RGB images
    - masks/: Binary segmentation masks
    - instances/: Instance segmentation masks
    - polygons/: JSON polygon annotations
    
    This matches the output format from DatasetGeneration pipeline.
    """
    
    def __init__(
        self,
        config: PatchFormerConfig,
        augment: bool = True,
        load_instances: bool = True,
        load_polygons: bool = False,
    ):
        # Ensure binary classification
        config.num_classes = 2
        super().__init__(config, augment, include_token_labels=True)
        
        self.load_instances = load_instances
        self.load_polygons = load_polygons
    
    def preprocess_with_instances(
        self,
        image: tf.Tensor,
        mask: tf.Tensor,
        instance_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Preprocess image, mask, and instance mask together.
        
        Args:
            image: Input image (H, W, 3)
            mask: Binary segmentation mask (H, W)
            instance_mask: Instance segmentation mask (H, W)
            
        Returns:
            image: Preprocessed image
            targets: Dictionary with mask, instance_map, and token labels
        """
        # Normalize image to [0, 1]
        if image.dtype == tf.uint8:
            image = tf.cast(image, tf.float32) / 255.0
        
        mask = tf.cast(mask, tf.int32)
        instance_mask = tf.cast(instance_mask, tf.int32)
        
        target_h, target_w = self.config.input_size
        
        if self.augment and self.augmentation_fn is not None:
            # Apply augmentation to image, mask, and instance mask together
            image, mask, instance_mask = self._augment_with_instances(
                image, mask, instance_mask
            )
        else:
            image = tf.image.resize(image, [target_h, target_w])
            mask = tf.image.resize(
                tf.expand_dims(tf.cast(mask, tf.float32), -1),
                [target_h, target_w],
                method='nearest'
            )
            mask = tf.squeeze(tf.cast(mask, tf.int32), -1)
            
            instance_mask = tf.image.resize(
                tf.expand_dims(tf.cast(instance_mask, tf.float32), -1),
                [target_h, target_w],
                method='nearest'
            )
            instance_mask = tf.squeeze(tf.cast(instance_mask, tf.int32), -1)
        
        # Prepare targets
        targets = {
            'segmentation': mask,
            'instance_map': instance_mask,
        }
        
        if self.include_token_labels:
            token_labels = generate_token_labels(
                tf.expand_dims(mask, 0),
                self.config.grid_size
            )
            targets['token_labels'] = tf.squeeze(token_labels, 0)
        
        return image, targets
    
    def _augment_with_instances(
        self,
        image: tf.Tensor,
        mask: tf.Tensor,
        instance_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply augmentations to image, mask, and instance mask."""
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(tf.expand_dims(mask, -1))
            mask = tf.squeeze(mask, -1)
            instance_mask = tf.image.flip_left_right(tf.expand_dims(instance_mask, -1))
            instance_mask = tf.squeeze(instance_mask, -1)
        
        # Random rotation (0, 90, 180, 270 degrees)
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        mask = tf.image.rot90(tf.expand_dims(mask, -1), k)
        mask = tf.squeeze(mask, -1)
        instance_mask = tf.image.rot90(tf.expand_dims(instance_mask, -1), k)
        instance_mask = tf.squeeze(instance_mask, -1)
        
        # Color jitter (only on image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        # Random scale and crop
        scale = tf.random.uniform((), minval=0.8, maxval=1.2)
        new_size = tf.cast(
            tf.cast(tf.shape(image)[:2], tf.float32) * scale,
            tf.int32
        )
        
        image = tf.image.resize(image, new_size)
        mask = tf.image.resize(
            tf.expand_dims(tf.cast(mask, tf.float32), -1),
            new_size,
            method='nearest'
        )
        mask = tf.squeeze(tf.cast(mask, tf.int32), -1)
        instance_mask = tf.image.resize(
            tf.expand_dims(tf.cast(instance_mask, tf.float32), -1),
            new_size,
            method='nearest'
        )
        instance_mask = tf.squeeze(tf.cast(instance_mask, tf.int32), -1)
        
        # Random crop to target size
        combined = tf.concat([
            image,
            tf.cast(tf.expand_dims(mask, -1), tf.float32),
            tf.cast(tf.expand_dims(instance_mask, -1), tf.float32)
        ], axis=-1)
        
        target_h, target_w = self.config.input_size
        combined = tf.image.random_crop(combined, [target_h, target_w, 5])
        
        image = combined[:, :, :3]
        mask = tf.cast(combined[:, :, 3], tf.int32)
        instance_mask = tf.cast(combined[:, :, 4], tf.int32)
        
        return image, mask, instance_mask
    
    def create_dataset_from_paths(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        instance_paths: List[str],
        batch_size: int = 4,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Create dataset from file paths including instance masks.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of binary mask paths
            instance_paths: List of instance mask paths
            batch_size: Batch size
            shuffle: Whether to shuffle
            
        Returns:
            tf.data.Dataset
        """
        def load_and_preprocess(image_path, mask_path, instance_path):
            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
            
            # Load binary mask
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.squeeze(mask, -1)
            
            # Load instance mask (16-bit)
            instance = tf.io.read_file(instance_path)
            instance = tf.image.decode_png(instance, channels=1, dtype=tf.uint16)
            instance = tf.squeeze(instance, -1)
            instance = tf.cast(instance, tf.int32)
            
            return self.preprocess_with_instances(image, mask, instance)
        
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, mask_paths, instance_paths)
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        
        dataset = dataset.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def load_unified_dataset_paths(
    dataset_dir: str,
    split: str = "train",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load file paths from unified dataset structure.
    
    Args:
        dataset_dir: Path to dataset root (containing train/, val/, test/)
        split: One of 'train', 'val', 'test'
        
    Returns:
        (image_paths, mask_paths, instance_paths)
    """
    root = Path(dataset_dir)
    split_dir = root / split
    
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    instances_dir = split_dir / "instances"
    
    image_paths = []
    mask_paths = []
    instance_paths = []
    
    # Find all masks (they're the canonical list)
    for mask_path in sorted(masks_dir.glob("*.png")):
        sample_id = mask_path.stem
        
        # Find corresponding image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = images_dir / f"{sample_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        # Find instance mask
        inst_path = instances_dir / f"{sample_id}.png"
        
        if img_path and inst_path.exists():
            image_paths.append(str(img_path))
            mask_paths.append(str(mask_path))
            instance_paths.append(str(inst_path))
    
    return image_paths, mask_paths, instance_paths


def create_unified_dataset(
    dataset_dir: str,
    config: PatchFormerConfig,
    split: str = "train",
    batch_size: int = 4,
    augment: bool = True,
) -> tf.data.Dataset:
    """
    Convenience function to create dataset from unified structure.
    
    Args:
        dataset_dir: Path to dataset root
        config: Model configuration
        split: Data split to load
        batch_size: Batch size
        augment: Whether to apply augmentation
        
    Returns:
        tf.data.Dataset ready for training
    """
    image_paths, mask_paths, instance_paths = load_unified_dataset_paths(
        dataset_dir, split
    )
    
    print(f"Loaded {len(image_paths)} {split} samples from {dataset_dir}")
    
    dataset_loader = UnifiedTextDetectionDataset(
        config, 
        augment=augment,
        load_instances=True
    )
    
    return dataset_loader.create_dataset_from_paths(
        image_paths, mask_paths, instance_paths,
        batch_size=batch_size,
        shuffle=(split == "train")
    )
