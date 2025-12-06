"""
Dataset Finalization for Text Detection Training.

Combines merged and augmented datasets into final training-ready structure:
- Merges original and augmented samples
- Resizes all images to target resolution (≥1024px)
- Validates all samples
- Creates final train/val/test splits
- Generates comprehensive manifest

Output structure:
    final_dataset/
    ├── train/
    │   ├── images/      # RGB images (PNG/JPG)
    │   ├── masks/       # Binary masks (0=bg, 1=text)
    │   ├── instances/   # Instance masks (unique ID per region)
    │   └── polygons/    # JSON with vertex coordinates
    ├── val/
    │   └── ...
    ├── test/            # Optional test split
    │   └── ...
    └── manifest.json    # Dataset metadata

Usage:
    python dataset_finalize.py --merged_dir ./merged --augmented_dir ./augmented --output_dir ./final
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import cv2


# ============================================================================
# Sample Processor
# ============================================================================

class SampleProcessor:
    """Process individual samples for finalization."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        min_size: int = 1024,
        maintain_aspect: bool = True
    ):
        self.target_size = target_size
        self.min_size = min_size
        self.maintain_aspect = maintain_aspect
    
    def process(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Process a sample: resize and validate.
        
        Args:
            image: RGB image
            mask: Binary mask
            instance: Instance mask
            polygons: Polygon annotations
            
        Returns:
            Processed (image, mask, instance, polygons)
        """
        h, w = image.shape[:2]
        
        # Calculate resize factor
        if self.maintain_aspect:
            # Resize so smaller dimension is at least min_size
            scale = max(self.min_size / min(h, w), 1.0)
            new_h = int(h * scale)
            new_w = int(w * scale)
        else:
            new_h, new_w = self.target_size
            scale = new_w / w  # Approximate scale for polygon transformation
        
        # Resize image
        if (new_h, new_w) != (h, w):
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            instance = cv2.resize(
                instance.astype(np.float32), (new_w, new_h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
            
            # Transform polygons
            scale_x = new_w / w
            scale_y = new_h / h
            
            transformed_polygons = []
            for poly in polygons:
                new_poly = poly.copy()
                if "points" in new_poly:
                    new_points = [
                        [p[0] * scale_x, p[1] * scale_y] 
                        for p in new_poly["points"]
                    ]
                    new_poly["points"] = new_points
                transformed_polygons.append(new_poly)
            polygons = transformed_polygons
        
        return image, mask, instance, polygons
    
    def validate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance: np.ndarray,
        polygons: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Validate a sample.
        
        Returns:
            (is_valid, error_message)
        """
        # Check image
        if image is None or image.size == 0:
            return False, "Empty image"
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            return False, f"Invalid image shape: {image.shape}"
        
        h, w = image.shape[:2]
        
        # Check mask
        if mask is None or mask.shape != (h, w):
            return False, f"Mask shape mismatch: {mask.shape if mask is not None else None} vs ({h}, {w})"
        
        # Check instance mask
        if instance is None or instance.shape != (h, w):
            return False, f"Instance mask shape mismatch"
        
        # Check for content (at least some text)
        if mask.max() == 0:
            return False, "No text content in mask"
        
        # Check polygons
        if not polygons:
            return False, "No polygon annotations"
        
        return True, ""


# ============================================================================
# Dataset Finalizer
# ============================================================================

class DatasetFinalizer:
    """Combine and finalize datasets for training."""
    
    def __init__(
        self,
        merged_dir: str,
        augmented_dir: str,
        output_dir: str,
        target_size: Tuple[int, int] = (1024, 1024)
    ):
        self.merged_dir = Path(merged_dir) if merged_dir else None
        self.augmented_dir = Path(augmented_dir) if augmented_dir else None
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        self.processor = SampleProcessor(target_size=target_size)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def finalize(
        self,
        test_split: float = 0.0,
        validate_samples: bool = True,
        num_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Finalize the dataset.
        
        Args:
            test_split: Fraction for test set (taken from validation)
            validate_samples: Whether to validate each sample
            num_workers: Number of parallel workers
            
        Returns:
            Statistics dictionary
        """
        print(f"\n{'#'*60}")
        print(f"# Dataset Finalization")
        print(f"# Merged: {self.merged_dir}")
        print(f"# Augmented: {self.augmented_dir}")
        print(f"# Output: {self.output_dir}")
        print(f"# Target size: {self.target_size}")
        print(f"{'#'*60}")
        
        stats = {
            "total": 0,
            "train": 0,
            "val": 0,
            "test": 0,
            "skipped": 0,
            "errors": []
        }
        
        # Process each split
        for split in ["train", "val"]:
            print(f"\n{'='*60}")
            print(f"Processing {split} split...")
            
            # Collect samples from both sources
            samples = []
            
            # From merged dataset
            if self.merged_dir:
                merged_split = self.merged_dir / split
                if merged_split.exists():
                    samples.extend(self._collect_samples(merged_split, "merged"))
            
            # From augmented dataset
            if self.augmented_dir:
                aug_split = self.augmented_dir / split
                if aug_split.exists():
                    samples.extend(self._collect_samples(aug_split, "augmented"))
            
            print(f"  Found {len(samples)} samples")
            
            if not samples:
                continue
            
            # Process and copy samples
            processed = self._process_samples(
                samples, split, validate_samples, num_workers
            )
            
            stats[split] = processed["success"]
            stats["skipped"] += processed["skipped"]
            stats["errors"].extend(processed["errors"])
        
        # Create test split from validation if requested
        if test_split > 0:
            self._create_test_split(test_split)
            # Update stats
            val_count = len(list((self.output_dir / "val" / "images").glob("*")))
            test_count = len(list((self.output_dir / "test" / "images").glob("*")))
            stats["val"] = val_count
            stats["test"] = test_count
        
        stats["total"] = stats["train"] + stats["val"] + stats["test"]
        
        # Create manifest
        manifest = self._create_manifest(stats)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Finalization Summary:")
        print(f"{'='*60}")
        print(f"  Train samples: {stats['train']}")
        print(f"  Val samples: {stats['val']}")
        print(f"  Test samples: {stats['test']}")
        print(f"  Total: {stats['total']}")
        print(f"  Skipped: {stats['skipped']}")
        if stats['errors']:
            print(f"  Errors: {len(stats['errors'])}")
        print(f"\n  Output: {self.output_dir}")
        
        return stats
    
    def _collect_samples(
        self, 
        split_dir: Path, 
        source: str
    ) -> List[Dict[str, Any]]:
        """Collect sample paths from a directory."""
        samples = []
        
        polygons_dir = split_dir / "polygons"
        if not polygons_dir.exists():
            return samples
        
        for poly_file in polygons_dir.glob("*.json"):
            sample_id = poly_file.stem
            
            # Find corresponding image
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = split_dir / "images" / f"{sample_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path is None:
                continue
            
            samples.append({
                "id": sample_id,
                "source": source,
                "image_path": image_path,
                "mask_path": split_dir / "masks" / f"{sample_id}.png",
                "instance_path": split_dir / "instances" / f"{sample_id}.png",
                "polygon_path": poly_file,
            })
        
        return samples
    
    def _process_samples(
        self,
        samples: List[Dict[str, Any]],
        split: str,
        validate: bool,
        num_workers: int
    ) -> Dict[str, Any]:
        """Process and save samples."""
        # Create output directories
        split_dir = self.output_dir / split
        for subdir in ["images", "masks", "instances", "polygons"]:
            (split_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        result = {"success": 0, "skipped": 0, "errors": []}
        
        for sample in tqdm(samples, desc=f"Finalizing {split}"):
            try:
                # Load sample
                image = cv2.imread(str(sample["image_path"]))
                if image is None:
                    result["skipped"] += 1
                    continue
                
                mask_path = sample["mask_path"]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else np.zeros(image.shape[:2], dtype=np.uint8)
                
                inst_path = sample["instance_path"]
                instance = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED) if inst_path.exists() else np.zeros(image.shape[:2], dtype=np.int32)
                if instance is None:
                    instance = np.zeros(image.shape[:2], dtype=np.int32)
                
                with open(sample["polygon_path"], 'r') as f:
                    poly_data = json.load(f)
                polygons = poly_data.get("polygons", [])
                
                # Process sample
                image, mask, instance, polygons = self.processor.process(
                    image, mask, instance, polygons
                )
                
                # Validate if requested
                if validate:
                    is_valid, error = self.processor.validate(
                        image, mask, instance, polygons
                    )
                    if not is_valid:
                        result["skipped"] += 1
                        result["errors"].append(f"{sample['id']}: {error}")
                        continue
                
                # Generate unique ID
                sample_id = f"{sample['source']}_{sample['id']}"
                
                # Save to output
                cv2.imwrite(str(split_dir / "images" / f"{sample_id}.jpg"), image)
                cv2.imwrite(str(split_dir / "masks" / f"{sample_id}.png"), mask)
                cv2.imwrite(str(split_dir / "instances" / f"{sample_id}.png"), 
                           instance.astype(np.uint16))
                
                # Update polygon data
                poly_data["image_id"] = sample_id
                poly_data["polygons"] = polygons
                poly_data["width"] = image.shape[1]
                poly_data["height"] = image.shape[0]
                
                with open(split_dir / "polygons" / f"{sample_id}.json", 'w') as f:
                    json.dump(poly_data, f, indent=2)
                
                result["success"] += 1
                
            except Exception as e:
                result["skipped"] += 1
                result["errors"].append(f"{sample['id']}: {str(e)}")
        
        return result
    
    def _create_test_split(self, test_fraction: float):
        """Create test split from validation samples."""
        import random
        
        val_dir = self.output_dir / "val"
        test_dir = self.output_dir / "test"
        
        if not val_dir.exists():
            return
        
        # Create test directories
        for subdir in ["images", "masks", "instances", "polygons"]:
            (test_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Get all validation samples
        val_samples = list((val_dir / "polygons").glob("*.json"))
        
        # Select samples for test
        num_test = int(len(val_samples) * test_fraction)
        random.shuffle(val_samples)
        test_samples = val_samples[:num_test]
        
        print(f"\n  Moving {num_test} samples to test split...")
        
        for poly_path in tqdm(test_samples, desc="Creating test split"):
            sample_id = poly_path.stem
            
            # Move files
            for subdir in ["images", "masks", "instances", "polygons"]:
                # Find source file
                src_dir = val_dir / subdir
                dst_dir = test_dir / subdir
                
                if subdir == "images":
                    for ext in ['.jpg', '.jpeg', '.png']:
                        src = src_dir / f"{sample_id}{ext}"
                        if src.exists():
                            shutil.move(str(src), str(dst_dir / f"{sample_id}{ext}"))
                            break
                else:
                    src = src_dir / f"{sample_id}.png" if subdir != "polygons" else src_dir / f"{sample_id}.json"
                    if src.exists():
                        dst = dst_dir / src.name
                        shutil.move(str(src), str(dst))
    
    def _create_manifest(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Create dataset manifest."""
        manifest = {
            "name": "Text Detection Dataset",
            "version": "1.0.0",
            "created": str(np.datetime64('now')),
            "statistics": {
                "total_samples": stats["total"],
                "train_samples": stats["train"],
                "val_samples": stats["val"],
                "test_samples": stats["test"],
                "skipped_samples": stats["skipped"],
            },
            "structure": {
                "images": {
                    "description": "RGB images",
                    "format": "JPEG/PNG",
                    "min_size": f"{self.target_size[0]}px",
                },
                "masks": {
                    "description": "Binary segmentation masks",
                    "format": "PNG (8-bit)",
                    "values": {"0": "background", "1": "text"},
                },
                "instances": {
                    "description": "Instance segmentation masks",
                    "format": "PNG (16-bit)",
                    "values": "0=background, 1,2,3...=instance IDs",
                },
                "polygons": {
                    "description": "Polygon annotations",
                    "format": "JSON",
                    "fields": [
                        "image_id", "width", "height", 
                        "polygons[].points", "polygons[].text"
                    ],
                },
            },
            "usage": {
                "training": "Use train/ and val/ splits for model training",
                "evaluation": "Use test/ split for final evaluation",
                "loading": "Each sample has corresponding files in images/, masks/, instances/, polygons/",
            },
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Also create a simple splits file for easy loading
        splits = {
            "train": sorted([p.stem for p in (self.output_dir / "train" / "polygons").glob("*.json")]) if (self.output_dir / "train" / "polygons").exists() else [],
            "val": sorted([p.stem for p in (self.output_dir / "val" / "polygons").glob("*.json")]) if (self.output_dir / "val" / "polygons").exists() else [],
            "test": sorted([p.stem for p in (self.output_dir / "test" / "polygons").glob("*.json")]) if (self.output_dir / "test" / "polygons").exists() else [],
        }
        
        splits_path = self.output_dir / "splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        return manifest


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Finalize text detection dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Combine merged and augmented datasets
    python dataset_finalize.py --merged_dir ./merged --augmented_dir ./augmented --output_dir ./final
    
    # Create with test split
    python dataset_finalize.py --merged_dir ./merged --output_dir ./final --test-split 0.1
    
    # From merged only (no augmentation)
    python dataset_finalize.py --merged_dir ./merged --output_dir ./final
"""
    )
    
    parser.add_argument(
        "--merged_dir", "-m",
        type=str,
        default="./merged_dataset",
        help="Directory with merged dataset"
    )
    
    parser.add_argument(
        "--augmented_dir", "-a",
        type=str,
        default="./augmented_dataset",
        help="Directory with augmented samples (optional)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./final_dataset",
        help="Output directory for final dataset"
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        default=1024,
        help="Target minimum image size (default: 1024)"
    )
    
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.0,
        help="Fraction of validation to use as test set (default: 0.0)"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip sample validation"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Handle optional augmented dir
    augmented_dir = args.augmented_dir
    if augmented_dir and not Path(augmented_dir).exists():
        print(f"Note: Augmented directory not found: {augmented_dir}")
        augmented_dir = None
    
    finalizer = DatasetFinalizer(
        merged_dir=args.merged_dir,
        augmented_dir=augmented_dir,
        output_dir=args.output_dir,
        target_size=(args.target_size, args.target_size)
    )
    
    finalizer.finalize(
        test_split=args.test_split,
        validate_samples=not args.no_validate,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
