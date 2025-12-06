"""
Dataset Augmentation Pipeline for Text Detection Training.

Applies augmentations to the merged dataset:
- Horizontal/Vertical Flip
- Rotation (90°, 180°, 270° and arbitrary angles)
- Mosaic (combine 4 images into 1)

Generates 2x the number of original samples.

Usage:
    python dataset_augment.py --input_dir ./merged_dataset --output_dir ./augmented_dataset
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cv2


# ============================================================================
# Augmentation Classes
# ============================================================================

class BaseAugmentation:
    """Base class for augmentations."""
    
    def __init__(self, probability: float = 1.0):
        self.probability = probability
    
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance_mask: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Apply augmentation."""
        if random.random() > self.probability:
            return image, mask, instance_mask, polygons
        return self._apply(image, mask, instance_mask, polygons)
    
    def _apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance_mask: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Override in subclasses."""
        raise NotImplementedError
    
    def _transform_polygons(
        self,
        polygons: List[Dict[str, Any]],
        transform_fn
    ) -> List[Dict[str, Any]]:
        """Transform polygon coordinates."""
        transformed = []
        for poly in polygons:
            new_poly = poly.copy()
            if "points" in new_poly:
                new_points = [transform_fn(p) for p in new_poly["points"]]
                new_poly["points"] = new_points
            transformed.append(new_poly)
        return transformed


class HorizontalFlip(BaseAugmentation):
    """Horizontal flip augmentation."""
    
    def _apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance_mask: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        h, w = image.shape[:2]
        
        # Flip images
        image_flipped = cv2.flip(image, 1)
        mask_flipped = cv2.flip(mask, 1)
        instance_flipped = cv2.flip(instance_mask, 1)
        
        # Transform polygons
        def transform(point):
            return [w - point[0], point[1]]
        
        polygons_flipped = self._transform_polygons(polygons, transform)
        
        return image_flipped, mask_flipped, instance_flipped, polygons_flipped


class VerticalFlip(BaseAugmentation):
    """Vertical flip augmentation."""
    
    def _apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance_mask: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        h, w = image.shape[:2]
        
        # Flip images
        image_flipped = cv2.flip(image, 0)
        mask_flipped = cv2.flip(mask, 0)
        instance_flipped = cv2.flip(instance_mask, 0)
        
        # Transform polygons
        def transform(point):
            return [point[0], h - point[1]]
        
        polygons_flipped = self._transform_polygons(polygons, transform)
        
        return image_flipped, mask_flipped, instance_flipped, polygons_flipped


class Rotation90(BaseAugmentation):
    """Rotate by 90, 180, or 270 degrees."""
    
    def __init__(self, angle: int = 90, probability: float = 1.0):
        super().__init__(probability)
        assert angle in [90, 180, 270], "Angle must be 90, 180, or 270"
        self.angle = angle
        self.k = angle // 90  # Number of 90-degree rotations
    
    def _apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance_mask: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        h, w = image.shape[:2]
        
        # Rotate images
        image_rot = np.rot90(image, self.k)
        mask_rot = np.rot90(mask, self.k)
        instance_rot = np.rot90(instance_mask, self.k)
        
        # Transform polygons based on rotation
        def transform(point):
            x, y = point
            if self.k == 1:  # 90 degrees CCW
                return [y, w - x]
            elif self.k == 2:  # 180 degrees
                return [w - x, h - y]
            elif self.k == 3:  # 270 degrees CCW (90 CW)
                return [h - y, x]
            return point
        
        polygons_rot = self._transform_polygons(polygons, transform)
        
        return image_rot, mask_rot, instance_rot, polygons_rot


class RandomRotation(BaseAugmentation):
    """Random rotation with arbitrary angle."""
    
    def __init__(self, max_angle: float = 15.0, probability: float = 1.0):
        super().__init__(probability)
        self.max_angle = max_angle
    
    def _apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance_mask: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        h, w = image.shape[:2]
        angle = random.uniform(-self.max_angle, self.max_angle)
        
        # Rotation center
        center = (w / 2, h / 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box size
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for new size
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # Rotate images
        image_rot = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))
        mask_rot = cv2.warpAffine(mask, M, (new_w, new_h), borderValue=0,
                                   flags=cv2.INTER_NEAREST)
        instance_rot = cv2.warpAffine(instance_mask.astype(np.float32), M, (new_w, new_h),
                                       borderValue=0, flags=cv2.INTER_NEAREST)
        instance_rot = instance_rot.astype(np.int32)
        
        # Transform polygons
        def transform(point):
            pt = np.array([[point[0], point[1], 1.0]])
            new_pt = M @ pt.T
            return [float(new_pt[0, 0]), float(new_pt[1, 0])]
        
        polygons_rot = self._transform_polygons(polygons, transform)
        
        # Crop back to original size (center crop)
        start_x = max(0, (new_w - w) // 2)
        start_y = max(0, (new_h - h) // 2)
        
        image_rot = image_rot[start_y:start_y+h, start_x:start_x+w]
        mask_rot = mask_rot[start_y:start_y+h, start_x:start_x+w]
        instance_rot = instance_rot[start_y:start_y+h, start_x:start_x+w]
        
        # Adjust polygon coordinates for crop
        def adjust_crop(point):
            return [point[0] - start_x, point[1] - start_y]
        
        polygons_rot = self._transform_polygons(polygons_rot, adjust_crop)
        
        # Filter out polygons that are now outside the image
        valid_polygons = []
        for poly in polygons_rot:
            if "points" in poly:
                points = np.array(poly["points"])
                # Check if any point is inside the image
                inside = (points[:, 0] >= 0) & (points[:, 0] < w) & \
                         (points[:, 1] >= 0) & (points[:, 1] < h)
                if np.any(inside):
                    # Clip points to image bounds
                    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                    poly["points"] = points.tolist()
                    valid_polygons.append(poly)
        
        return image_rot, mask_rot, instance_rot, valid_polygons


class Mosaic(BaseAugmentation):
    """
    Mosaic augmentation - combines 4 images into one.
    Creates diverse training samples with multiple text instances.
    """
    
    def __init__(self, all_samples: List[Path], probability: float = 1.0):
        super().__init__(probability)
        self.all_samples = all_samples
    
    def _load_sample(self, sample_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """Load a sample's images and annotations."""
        images_dir = sample_path.parent.parent / "images"
        masks_dir = sample_path.parent.parent / "masks"
        instances_dir = sample_path.parent.parent / "instances"
        polygons_dir = sample_path.parent.parent / "polygons"
        
        stem = sample_path.stem
        
        # Find image
        image = None
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = images_dir / f"{stem}{ext}"
            if img_path.exists():
                image = cv2.imread(str(img_path))
                break
        
        if image is None:
            return None, None, None, []
        
        # Load mask
        mask_path = masks_dir / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Load instance mask
        inst_path = instances_dir / f"{stem}.png"
        instance = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED) if inst_path.exists() else np.zeros(image.shape[:2], dtype=np.int32)
        
        # Load polygons
        poly_path = polygons_dir / f"{stem}.json"
        polygons = []
        if poly_path.exists():
            with open(poly_path, 'r') as f:
                data = json.load(f)
                polygons = data.get("polygons", [])
        
        return image, mask, instance, polygons
    
    def create_mosaic(
        self,
        samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List]],
        output_size: Tuple[int, int] = (1024, 1024)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Create mosaic from 4 samples.
        
        Args:
            samples: List of 4 (image, mask, instance, polygons) tuples
            output_size: Output image size (h, w)
            
        Returns:
            Mosaic image, mask, instance mask, and combined polygons
        """
        h, w = output_size
        
        # Create output arrays
        mosaic_image = np.zeros((h, w, 3), dtype=np.uint8)
        mosaic_mask = np.zeros((h, w), dtype=np.uint8)
        mosaic_instance = np.zeros((h, w), dtype=np.int32)
        mosaic_polygons = []
        
        # Random center point for the mosaic
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        
        # Quadrant positions
        positions = [
            (0, 0, cx, cy),           # Top-left
            (cx, 0, w - cx, cy),      # Top-right
            (0, cy, cx, h - cy),      # Bottom-left
            (cx, cy, w - cx, h - cy)  # Bottom-right
        ]
        
        instance_offset = 0
        
        for idx, (sample, (x, y, qw, qh)) in enumerate(zip(samples, positions)):
            image, mask, instance, polygons = sample
            
            if image is None:
                continue
            
            # Resize sample to fit quadrant
            img_h, img_w = image.shape[:2]
            scale = min(qw / img_w, qh / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            # Resize
            image_resized = cv2.resize(image, (new_w, new_h))
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            instance_resized = cv2.resize(instance.astype(np.float32), (new_w, new_h),
                                          interpolation=cv2.INTER_NEAREST).astype(np.int32)
            
            # Place in mosaic (center in quadrant)
            offset_x = x + (qw - new_w) // 2
            offset_y = y + (qh - new_h) // 2
            
            mosaic_image[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = image_resized
            mosaic_mask[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = mask_resized
            
            # Offset instance IDs to avoid conflicts
            instance_resized_offset = np.where(
                instance_resized > 0,
                instance_resized + instance_offset,
                0
            )
            mosaic_instance[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = instance_resized_offset
            
            # Update instance offset
            if instance_resized.max() > 0:
                instance_offset = mosaic_instance.max()
            
            # Transform polygons
            for poly in polygons:
                new_poly = poly.copy()
                if "points" in new_poly:
                    new_points = []
                    for pt in new_poly["points"]:
                        new_x = pt[0] * scale + offset_x
                        new_y = pt[1] * scale + offset_y
                        new_points.append([new_x, new_y])
                    new_poly["points"] = new_points
                mosaic_polygons.append(new_poly)
        
        return mosaic_image, mosaic_mask, mosaic_instance, mosaic_polygons
    
    def _apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance_mask: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Apply mosaic with current sample + 3 random samples."""
        # Current sample
        current = (image, mask, instance_mask, polygons)
        
        # Select 3 random other samples
        other_samples = random.sample(self.all_samples, min(3, len(self.all_samples)))
        
        samples = [current]
        for sample_path in other_samples:
            loaded = self._load_sample(sample_path)
            if loaded[0] is not None:
                samples.append(loaded)
        
        # Pad with current if we don't have enough
        while len(samples) < 4:
            samples.append(current)
        
        # Create mosaic
        h, w = image.shape[:2]
        return self.create_mosaic(samples[:4], (h, w))


# ============================================================================
# Augmentation Pipeline
# ============================================================================

class AugmentationPipeline:
    """Pipeline to apply multiple augmentations."""
    
    def __init__(self, augmentations: List[BaseAugmentation]):
        self.augmentations = augmentations
    
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance_mask: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Apply all augmentations in sequence."""
        for aug in self.augmentations:
            image, mask, instance_mask, polygons = aug(image, mask, instance_mask, polygons)
        return image, mask, instance_mask, polygons


# ============================================================================
# Dataset Augmenter
# ============================================================================

class DatasetAugmenter:
    """Apply augmentations to entire dataset."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def augment(
        self,
        target_multiplier: float = 2.0,
        num_workers: int = 4
    ) -> Dict[str, int]:
        """
        Augment the dataset.
        
        Args:
            target_multiplier: Target ratio of augmented to original samples
            num_workers: Number of parallel workers
            
        Returns:
            Statistics dictionary
        """
        print(f"\n{'#'*60}")
        print(f"# Dataset Augmentation")
        print(f"# Input: {self.input_dir}")
        print(f"# Output: {self.output_dir}")
        print(f"# Target: {target_multiplier}x augmented samples")
        print(f"{'#'*60}")
        
        stats = {"original": 0, "augmented": 0}
        
        # Process each split
        for split in ["train", "val"]:
            split_input = self.input_dir / split
            split_output = self.output_dir / split
            
            if not split_input.exists():
                continue
            
            # Create output directories
            for subdir in ["images", "masks", "instances", "polygons"]:
                (split_output / subdir).mkdir(parents=True, exist_ok=True)
            
            # Get all samples
            polygons_dir = split_input / "polygons"
            if not polygons_dir.exists():
                continue
            
            sample_files = list(polygons_dir.glob("*.json"))
            num_samples = len(sample_files)
            
            if num_samples == 0:
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing {split} split ({num_samples} samples)...")
            
            # Calculate number of augmentations per sample
            target_augmented = int(num_samples * target_multiplier)
            augs_per_sample = max(1, target_augmented // num_samples)
            
            # Create augmentation strategies
            strategies = self._create_augmentation_strategies(sample_files)
            
            augmented_count = 0
            
            for sample_file in tqdm(sample_files, desc=f"Augmenting {split}"):
                # Load sample
                sample = self._load_sample(split_input, sample_file.stem)
                if sample is None:
                    continue
                
                image, mask, instance, polygons_data = sample
                polygons = polygons_data.get("polygons", [])
                
                # Apply different augmentation strategies
                for aug_idx in range(augs_per_sample):
                    strategy = strategies[aug_idx % len(strategies)]
                    
                    aug_image, aug_mask, aug_instance, aug_polygons = strategy(
                        image.copy(), mask.copy(), instance.copy(), polygons
                    )
                    
                    # Save augmented sample
                    aug_id = f"{sample_file.stem}_aug{aug_idx}"
                    self._save_sample(
                        split_output, aug_id,
                        aug_image, aug_mask, aug_instance,
                        aug_polygons, polygons_data
                    )
                    augmented_count += 1
            
            stats[f"{split}_augmented"] = augmented_count
            stats["augmented"] += augmented_count
            stats["original"] += num_samples
            
            print(f"  ✓ Generated {augmented_count} augmented samples")
        
        # Save stats
        stats_path = self.output_dir / "augmentation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Augmentation Summary:")
        print(f"{'='*60}")
        print(f"  Original samples: {stats['original']}")
        print(f"  Augmented samples: {stats['augmented']}")
        print(f"  Total: {stats['original'] + stats['augmented']}")
        print(f"\n  Output: {self.output_dir}")
        
        return stats
    
    def _create_augmentation_strategies(
        self, 
        sample_files: List[Path]
    ) -> List[AugmentationPipeline]:
        """Create different augmentation strategy pipelines."""
        strategies = [
            # Strategy 1: Horizontal flip
            AugmentationPipeline([HorizontalFlip()]),
            
            # Strategy 2: Vertical flip
            AugmentationPipeline([VerticalFlip()]),
            
            # Strategy 3: 90° rotation
            AugmentationPipeline([Rotation90(90)]),
            
            # Strategy 4: 180° rotation
            AugmentationPipeline([Rotation90(180)]),
            
            # Strategy 5: 270° rotation
            AugmentationPipeline([Rotation90(270)]),
            
            # Strategy 6: Random small rotation
            AugmentationPipeline([RandomRotation(15)]),
            
            # Strategy 7: Horizontal flip + rotation
            AugmentationPipeline([HorizontalFlip(), Rotation90(90)]),
            
            # Strategy 8: Mosaic
            AugmentationPipeline([Mosaic(sample_files)]),
        ]
        
        return strategies
    
    def _load_sample(
        self, 
        split_dir: Path, 
        sample_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
        """Load a sample."""
        # Find image
        image = None
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = split_dir / "images" / f"{sample_id}{ext}"
            if img_path.exists():
                image = cv2.imread(str(img_path))
                break
        
        if image is None:
            return None
        
        # Load mask
        mask_path = split_dir / "masks" / f"{sample_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Load instance mask
        inst_path = split_dir / "instances" / f"{sample_id}.png"
        instance = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED) if inst_path.exists() else np.zeros(image.shape[:2], dtype=np.int32)
        if instance is None:
            instance = np.zeros(image.shape[:2], dtype=np.int32)
        
        # Load polygons
        poly_path = split_dir / "polygons" / f"{sample_id}.json"
        polygons_data = {}
        if poly_path.exists():
            with open(poly_path, 'r') as f:
                polygons_data = json.load(f)
        
        return image, mask, instance, polygons_data
    
    def _save_sample(
        self,
        split_dir: Path,
        sample_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        instance: np.ndarray,
        polygons: List[Dict],
        original_data: Dict
    ):
        """Save an augmented sample."""
        # Save image
        cv2.imwrite(str(split_dir / "images" / f"{sample_id}.jpg"), image)
        
        # Save mask
        cv2.imwrite(str(split_dir / "masks" / f"{sample_id}.png"), mask)
        
        # Save instance mask
        cv2.imwrite(str(split_dir / "instances" / f"{sample_id}.png"), 
                   instance.astype(np.uint16))
        
        # Save polygons
        poly_data = original_data.copy()
        poly_data["image_id"] = sample_id
        poly_data["polygons"] = polygons
        poly_data["augmented"] = True
        poly_data["width"] = image.shape[1]
        poly_data["height"] = image.shape[0]
        
        with open(split_dir / "polygons" / f"{sample_id}.json", 'w') as f:
            json.dump(poly_data, f, indent=2)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Augment text detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Augment to 2x samples
    python dataset_augment.py --input_dir ./merged_dataset --output_dir ./augmented_dataset
    
    # Augment to 3x samples
    python dataset_augment.py --input_dir ./merged --output_dir ./augmented --multiplier 3.0
"""
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        default="./merged_dataset",
        help="Input directory with merged dataset"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./augmented_dataset",
        help="Output directory for augmented samples"
    )
    
    parser.add_argument(
        "--multiplier", "-m",
        type=float,
        default=2.0,
        help="Target multiplier for augmented samples (default: 2.0)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    augmenter = DatasetAugmenter(args.input_dir, args.output_dir)
    augmenter.augment(args.multiplier, args.workers)


if __name__ == "__main__":
    main()
