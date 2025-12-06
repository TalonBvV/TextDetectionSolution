"""
Dataset Merger and Converter for Text Detection Training.

Converts multiple text detection datasets to a unified format:
- Images: RGB PNG/JPG
- Masks: Binary PNG (0=background, 1=text)
- Instances: Integer PNG (unique ID per text region)
- Polygons: JSON with vertex coordinates

Supported datasets:
- HierText
- COCO-Text v2
- TextOCR
- ClapperText
- CORD-v2

Usage:
    python dataset_merge.py --input_dir ./raw_datasets --output_dir ./merged_dataset
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cv2

# Try importing PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")


# ============================================================================
# Unified Data Structures
# ============================================================================

@dataclass
class TextPolygon:
    """Represents a single text region polygon."""
    points: List[List[float]]  # List of [x, y] coordinates
    text: str = ""
    language: str = "en"
    legibility: str = "legible"  # legible, illegible
    word_type: str = "word"  # word, line, paragraph
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box (x_min, y_min, x_max, y_max)."""
        points = np.array(self.points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        return int(x_min), int(y_min), int(x_max), int(y_max)


@dataclass
class UnifiedSample:
    """Represents a single sample in unified format."""
    image_id: str
    source_dataset: str
    original_path: str
    width: int
    height: int
    polygons: List[TextPolygon] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_dataset": self.source_dataset,
            "original_path": self.original_path,
            "width": self.width,
            "height": self.height,
            "polygons": [p.to_dict() for p in self.polygons]
        }


# ============================================================================
# Dataset Converters
# ============================================================================

class BaseConverter:
    """Base class for dataset converters."""
    
    def __init__(self, dataset_dir: Path, output_dir: Path):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.masks_dir = output_dir / "masks"
        self.instances_dir = output_dir / "instances"
        self.polygons_dir = output_dir / "polygons"
        
        # Create output directories
        for d in [self.images_dir, self.masks_dir, self.instances_dir, self.polygons_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def convert(self) -> List[UnifiedSample]:
        """Convert dataset to unified format. Override in subclasses."""
        raise NotImplementedError
    
    def _create_masks(
        self, 
        sample: UnifiedSample, 
        output_prefix: str
    ) -> Tuple[Path, Path]:
        """
        Create binary mask and instance mask from polygons.
        
        Args:
            sample: Unified sample with polygons
            output_prefix: Prefix for output files
            
        Returns:
            (mask_path, instance_path)
        """
        height, width = sample.height, sample.width
        
        # Binary mask (0=background, 1=text)
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Instance mask (0=background, 1,2,3...=instance IDs)
        instance_mask = np.zeros((height, width), dtype=np.int32)
        
        for idx, polygon in enumerate(sample.polygons, start=1):
            points = np.array(polygon.points, dtype=np.int32)
            
            # Ensure points form a valid polygon
            if len(points) < 3:
                continue
            
            # Fill polygon on binary mask
            cv2.fillPoly(binary_mask, [points], 1)
            
            # Fill polygon on instance mask with unique ID
            cv2.fillPoly(instance_mask, [points], idx)
        
        # Save masks
        mask_path = self.masks_dir / f"{output_prefix}.png"
        instance_path = self.instances_dir / f"{output_prefix}.png"
        
        cv2.imwrite(str(mask_path), binary_mask)
        
        # Save instance mask as 16-bit to support more instances
        cv2.imwrite(str(instance_path), instance_mask.astype(np.uint16))
        
        return mask_path, instance_path
    
    def _save_polygons(self, sample: UnifiedSample, output_prefix: str) -> Path:
        """Save polygon annotations as JSON."""
        polygon_path = self.polygons_dir / f"{output_prefix}.json"
        
        with open(polygon_path, 'w', encoding='utf-8') as f:
            json.dump(sample.to_dict(), f, indent=2, ensure_ascii=False)
        
        return polygon_path
    
    def _copy_image(self, src_path: Path, output_prefix: str) -> Path:
        """Copy and optionally convert image to output directory."""
        # Determine output format (keep original or convert to PNG)
        suffix = src_path.suffix.lower()
        if suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
            out_suffix = suffix
        else:
            out_suffix = '.png'
        
        dst_path = self.images_dir / f"{output_prefix}{out_suffix}"
        
        if src_path.exists():
            # Read and write to ensure consistent format
            img = cv2.imread(str(src_path))
            if img is not None:
                cv2.imwrite(str(dst_path), img)
        
        return dst_path


class HierTextConverter(BaseConverter):
    """Convert HierText dataset to unified format."""
    
    def convert(self) -> List[UnifiedSample]:
        print(f"\n{'='*60}")
        print("Converting HierText Dataset")
        print(f"{'='*60}")
        
        samples = []
        gt_dir = self.dataset_dir / "ground_truth"
        images_dir = self.dataset_dir / "images"
        
        # Process both train and val splits
        for split in ["train", "val"]:
            jsonl_path = gt_dir / f"{split}.jsonl"
            
            if not jsonl_path.exists():
                print(f"  ⚠ {split}.jsonl not found, skipping...")
                continue
            
            split_images_dir = images_dir / split
            
            print(f"  Processing {split} split...")
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in tqdm(lines, desc=f"HierText {split}"):
                try:
                    data = json.loads(line.strip())
                    sample = self._parse_hiertext_sample(data, split_images_dir, split)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        print(f"  ✓ Converted {len(samples)} samples")
        return samples
    
    def _parse_hiertext_sample(
        self, 
        data: Dict[str, Any], 
        images_dir: Path,
        split: str
    ) -> Optional[UnifiedSample]:
        """Parse a single HierText sample."""
        image_id = data.get("image_id", "")
        
        # Find image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            return None
        
        # Get image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        height, width = img.shape[:2]
        
        # Extract polygons from hierarchical structure
        polygons = []
        annotations = data.get("annotations", [])
        
        for para in annotations:
            # Paragraph level
            for line in para.get("lines", []):
                # Line level
                for word in line.get("words", []):
                    # Word level - this is what we want for text detection
                    vertices = word.get("vertices", [])
                    if len(vertices) >= 3:
                        polygon = TextPolygon(
                            points=vertices,
                            text=word.get("text", ""),
                            legibility="legible" if word.get("legible", True) else "illegible",
                            word_type="word"
                        )
                        polygons.append(polygon)
        
        if not polygons:
            return None
        
        output_prefix = f"hiertext_{split}_{image_id}"
        
        sample = UnifiedSample(
            image_id=output_prefix,
            source_dataset="hiertext",
            original_path=str(image_path),
            width=width,
            height=height,
            polygons=polygons
        )
        
        # Create outputs
        self._copy_image(image_path, output_prefix)
        self._create_masks(sample, output_prefix)
        self._save_polygons(sample, output_prefix)
        
        return sample


class COCOTextConverter(BaseConverter):
    """Convert COCO-Text v2 dataset to unified format."""
    
    def convert(self) -> List[UnifiedSample]:
        print(f"\n{'='*60}")
        print("Converting COCO-Text v2 Dataset")
        print(f"{'='*60}")
        
        samples = []
        
        # Find annotation file
        annotations_dir = self.dataset_dir / "annotations"
        images_dir = self.dataset_dir / "images"
        
        # Look for the main annotation file
        ann_file = None
        for candidate in [
            annotations_dir / "cocotext.v2.json",
            self.dataset_dir / "cocotext.v2.json",
            annotations_dir / "COCO_Text.json",
        ]:
            if candidate.exists():
                ann_file = candidate
                break
        
        # Also check for extracted structure
        if ann_file is None:
            for f in annotations_dir.rglob("*.json"):
                if "cocotext" in f.name.lower():
                    ann_file = f
                    break
        
        if ann_file is None:
            print("  ⚠ COCO-Text annotation file not found")
            return samples
        
        print(f"  Loading annotations from {ann_file.name}...")
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # COCO-Text format has 'imgs' and 'anns' keys
        imgs = data.get("imgs", data.get("images", {}))
        anns = data.get("anns", data.get("annotations", {}))
        
        # Group annotations by image
        img_to_anns = {}
        for ann_id, ann in anns.items():
            img_id = str(ann.get("image_id", ""))
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        print(f"  Processing {len(imgs)} images...")
        
        for img_id, img_info in tqdm(imgs.items(), desc="COCO-Text"):
            if img_id not in img_to_anns:
                continue
            
            sample = self._parse_cocotext_sample(
                img_id, img_info, img_to_anns[img_id], images_dir
            )
            if sample:
                samples.append(sample)
        
        print(f"  ✓ Converted {len(samples)} samples")
        return samples
    
    def _parse_cocotext_sample(
        self,
        img_id: str,
        img_info: Dict[str, Any],
        annotations: List[Dict[str, Any]],
        images_dir: Path
    ) -> Optional[UnifiedSample]:
        """Parse a single COCO-Text sample."""
        # Get image path
        file_name = img_info.get("file_name", f"COCO_train2014_{int(img_id):012d}.jpg")
        
        # Try different image locations
        image_path = None
        for subdir in ["train2014", "val2014", ""]:
            candidate = images_dir / subdir / file_name
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            return None
        
        # Get dimensions
        width = img_info.get("width", 0)
        height = img_info.get("height", 0)
        
        if width == 0 or height == 0:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            height, width = img.shape[:2]
        
        # Parse annotations
        polygons = []
        for ann in annotations:
            # Skip non-text or illegible annotations if needed
            if ann.get("class", "machine printed") == "not text":
                continue
            
            # Get bounding box or polygon
            bbox = ann.get("bbox", ann.get("mask", []))
            
            if isinstance(bbox, list) and len(bbox) == 4:
                # Convert bbox [x, y, w, h] to polygon
                x, y, w, h = bbox
                points = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]
            elif isinstance(bbox, list) and len(bbox) >= 6:
                # Already a polygon (list of coordinates)
                points = [[bbox[i], bbox[i+1]] for i in range(0, len(bbox), 2)]
            else:
                continue
            
            polygon = TextPolygon(
                points=points,
                text=ann.get("utf8_string", ""),
                legibility="legible" if ann.get("legibility", "legible") == "legible" else "illegible",
                language=ann.get("language", "english"),
                word_type="word"
            )
            polygons.append(polygon)
        
        if not polygons:
            return None
        
        output_prefix = f"cocotext_{img_id}"
        
        sample = UnifiedSample(
            image_id=output_prefix,
            source_dataset="cocotext",
            original_path=str(image_path),
            width=width,
            height=height,
            polygons=polygons
        )
        
        # Create outputs
        self._copy_image(image_path, output_prefix)
        self._create_masks(sample, output_prefix)
        self._save_polygons(sample, output_prefix)
        
        return sample


class TextOCRConverter(BaseConverter):
    """Convert TextOCR dataset to unified format."""
    
    def convert(self) -> List[UnifiedSample]:
        print(f"\n{'='*60}")
        print("Converting TextOCR Dataset")
        print(f"{'='*60}")
        
        samples = []
        annotations_dir = self.dataset_dir / "annotations"
        images_dir = self.dataset_dir / "images"
        
        for split in ["train", "val"]:
            ann_file = annotations_dir / f"TextOCR_0.1_{split}.json"
            
            if not ann_file.exists():
                print(f"  ⚠ {split} annotations not found, skipping...")
                continue
            
            print(f"  Loading {split} annotations...")
            
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # TextOCR format
            imgs = data.get("imgs", {})
            anns = data.get("anns", {})
            img_to_anns = data.get("imgToAnns", {})
            
            print(f"  Processing {len(imgs)} {split} images...")
            
            for img_id, img_info in tqdm(imgs.items(), desc=f"TextOCR {split}"):
                ann_ids = img_to_anns.get(img_id, [])
                img_anns = [anns[str(aid)] for aid in ann_ids if str(aid) in anns]
                
                sample = self._parse_textocr_sample(
                    img_id, img_info, img_anns, images_dir, split
                )
                if sample:
                    samples.append(sample)
        
        print(f"  ✓ Converted {len(samples)} samples")
        return samples
    
    def _parse_textocr_sample(
        self,
        img_id: str,
        img_info: Dict[str, Any],
        annotations: List[Dict[str, Any]],
        images_dir: Path,
        split: str
    ) -> Optional[UnifiedSample]:
        """Parse a single TextOCR sample."""
        # TextOCR uses Open Images IDs
        file_name = img_info.get("file_name", f"{img_id}.jpg")
        
        # Find image
        image_path = None
        for subdir in ["", split, "train", "validation"]:
            candidate = images_dir / subdir / file_name
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            return None
        
        width = img_info.get("width", 0)
        height = img_info.get("height", 0)
        
        if width == 0 or height == 0:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            height, width = img.shape[:2]
        
        # Parse annotations
        polygons = []
        for ann in annotations:
            # Get polygon points
            points = ann.get("points", [])
            
            if isinstance(points, list) and len(points) >= 6:
                # Flat list [x1,y1,x2,y2,...]
                polygon_points = [[points[i], points[i+1]] for i in range(0, len(points), 2)]
            elif "bbox" in ann:
                # Bounding box format
                bbox = ann["bbox"]
                x, y, w, h = bbox
                polygon_points = [
                    [x, y], [x + w, y], [x + w, y + h], [x, y + h]
                ]
            else:
                continue
            
            polygon = TextPolygon(
                points=polygon_points,
                text=ann.get("utf8_string", ann.get("text", "")),
                word_type="word"
            )
            polygons.append(polygon)
        
        if not polygons:
            return None
        
        output_prefix = f"textocr_{split}_{img_id}"
        
        sample = UnifiedSample(
            image_id=output_prefix,
            source_dataset="textocr",
            original_path=str(image_path),
            width=width,
            height=height,
            polygons=polygons
        )
        
        self._copy_image(image_path, output_prefix)
        self._create_masks(sample, output_prefix)
        self._save_polygons(sample, output_prefix)
        
        return sample


class ClapperTextConverter(BaseConverter):
    """Convert ClapperText dataset to unified format."""
    
    def convert(self) -> List[UnifiedSample]:
        print(f"\n{'='*60}")
        print("Converting ClapperText Dataset")
        print(f"{'='*60}")
        
        samples = []
        
        # ClapperText structure varies - try common patterns
        # Pattern 1: images/ + annotations/
        images_dir = self.dataset_dir / "images"
        annotations_dir = self.dataset_dir / "annotations"
        
        # Pattern 2: Nested from GitHub extraction
        if not images_dir.exists():
            for subdir in self.dataset_dir.iterdir():
                if subdir.is_dir():
                    candidate_images = subdir / "images"
                    if candidate_images.exists():
                        images_dir = candidate_images
                        annotations_dir = subdir / "annotations"
                        break
        
        if not images_dir.exists():
            print("  ⚠ ClapperText images directory not found")
            return samples
        
        # Find all images and corresponding annotations
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        print(f"  Found {len(image_files)} images...")
        
        for image_path in tqdm(image_files, desc="ClapperText"):
            # Look for annotation file
            ann_path = None
            for ext in [".json", ".txt", ".xml"]:
                candidate = annotations_dir / f"{image_path.stem}{ext}"
                if candidate.exists():
                    ann_path = candidate
                    break
            
            sample = self._parse_clappertext_sample(image_path, ann_path)
            if sample:
                samples.append(sample)
        
        print(f"  ✓ Converted {len(samples)} samples")
        return samples
    
    def _parse_clappertext_sample(
        self,
        image_path: Path,
        ann_path: Optional[Path]
    ) -> Optional[UnifiedSample]:
        """Parse a single ClapperText sample."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        height, width = img.shape[:2]
        polygons = []
        
        if ann_path and ann_path.exists():
            if ann_path.suffix == ".json":
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                
                # Parse JSON annotations
                for ann in data.get("annotations", data.get("words", [])):
                    points = ann.get("points", ann.get("polygon", []))
                    if points:
                        if isinstance(points[0], (int, float)):
                            # Flat list
                            points = [[points[i], points[i+1]] for i in range(0, len(points), 2)]
                        polygon = TextPolygon(
                            points=points,
                            text=ann.get("text", ""),
                            word_type="word"
                        )
                        polygons.append(polygon)
            
            elif ann_path.suffix == ".txt":
                # Common format: x1,y1,x2,y2,x3,y3,x4,y4,text
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 8:
                            try:
                                coords = [float(p) for p in parts[:8]]
                                points = [[coords[i], coords[i+1]] for i in range(0, 8, 2)]
                                text = ','.join(parts[8:]) if len(parts) > 8 else ""
                                polygon = TextPolygon(points=points, text=text, word_type="word")
                                polygons.append(polygon)
                            except ValueError:
                                continue
        
        # If no annotations, skip
        if not polygons:
            return None
        
        output_prefix = f"clappertext_{image_path.stem}"
        
        sample = UnifiedSample(
            image_id=output_prefix,
            source_dataset="clappertext",
            original_path=str(image_path),
            width=width,
            height=height,
            polygons=polygons
        )
        
        self._copy_image(image_path, output_prefix)
        self._create_masks(sample, output_prefix)
        self._save_polygons(sample, output_prefix)
        
        return sample


class CORDConverter(BaseConverter):
    """Convert CORD-v2 dataset to unified format."""
    
    def convert(self) -> List[UnifiedSample]:
        print(f"\n{'='*60}")
        print("Converting CORD-v2 Dataset")
        print(f"{'='*60}")
        
        samples = []
        
        # CORD structure from our downloader: split/images + split/annotations
        for split in ["train", "validation", "test"]:
            split_dir = self.dataset_dir / split
            
            if not split_dir.exists():
                continue
            
            images_dir = split_dir / "images"
            annotations_dir = split_dir / "annotations"
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            
            print(f"  Processing {split} split ({len(image_files)} images)...")
            
            for image_path in tqdm(image_files, desc=f"CORD {split}"):
                ann_path = annotations_dir / f"{image_path.stem}.json"
                sample = self._parse_cord_sample(image_path, ann_path, split)
                if sample:
                    samples.append(sample)
        
        print(f"  ✓ Converted {len(samples)} samples")
        return samples
    
    def _parse_cord_sample(
        self,
        image_path: Path,
        ann_path: Path,
        split: str
    ) -> Optional[UnifiedSample]:
        """Parse a single CORD sample."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        height, width = img.shape[:2]
        polygons = []
        
        if ann_path.exists():
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # CORD format has nested structure
            # Try different possible structures
            def extract_words(obj, polygons):
                if isinstance(obj, dict):
                    # Check for word-level annotation
                    if "quad" in obj:
                        quad = obj["quad"]
                        points = [
                            [quad.get("x1", 0), quad.get("y1", 0)],
                            [quad.get("x2", 0), quad.get("y2", 0)],
                            [quad.get("x3", 0), quad.get("y3", 0)],
                            [quad.get("x4", 0), quad.get("y4", 0)]
                        ]
                        polygon = TextPolygon(
                            points=points,
                            text=obj.get("text", ""),
                            word_type="word"
                        )
                        polygons.append(polygon)
                    
                    # Recurse into nested structures
                    for key, value in obj.items():
                        if key in ["words", "lines", "valid_line", "gt_parse"]:
                            extract_words(value, polygons)
                        elif isinstance(value, (dict, list)):
                            extract_words(value, polygons)
                
                elif isinstance(obj, list):
                    for item in obj:
                        extract_words(item, polygons)
            
            extract_words(data, polygons)
        
        if not polygons:
            return None
        
        output_prefix = f"cord_{split}_{image_path.stem}"
        
        sample = UnifiedSample(
            image_id=output_prefix,
            source_dataset="cord",
            original_path=str(image_path),
            width=width,
            height=height,
            polygons=polygons
        )
        
        self._copy_image(image_path, output_prefix)
        self._create_masks(sample, output_prefix)
        self._save_polygons(sample, output_prefix)
        
        return sample


# ============================================================================
# Dataset Merger
# ============================================================================

class DatasetMerger:
    """Merge multiple datasets into unified format."""
    
    CONVERTERS = {
        "hiertext": HierTextConverter,
        "cocotext": COCOTextConverter,
        "textocr": TextOCRConverter,
        "clappertext": ClapperTextConverter,
        "cord": CORDConverter,
    }
    
    DATASET_DIRS = {
        "hiertext": "hiertext",
        "cocotext": "coco_text_v2",
        "textocr": "textocr",
        "clappertext": "clappertext",
        "cord": "cord_v2",
    }
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
    
    def merge(
        self, 
        datasets: List[str], 
        val_split: float = 0.1,
        max_samples_per_dataset: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Merge specified datasets.
        
        Args:
            datasets: List of dataset names or ["all"]
            val_split: Fraction of samples for validation
            max_samples_per_dataset: Optional limit per dataset
            
        Returns:
            Dictionary with counts per dataset
        """
        if "all" in datasets:
            datasets = list(self.CONVERTERS.keys())
        
        print(f"\n{'#'*60}")
        print(f"# Dataset Merger")
        print(f"# Input: {self.input_dir}")
        print(f"# Output: {self.output_dir}")
        print(f"# Datasets: {', '.join(datasets)}")
        print(f"{'#'*60}")
        
        all_samples = []
        counts = {}
        
        for dataset_name in datasets:
            if dataset_name not in self.CONVERTERS:
                print(f"\n⚠ Unknown dataset: {dataset_name}")
                counts[dataset_name] = 0
                continue
            
            # Find dataset directory
            dataset_dir = None
            for dir_name in [self.DATASET_DIRS[dataset_name], dataset_name]:
                candidate = self.input_dir / dir_name
                if candidate.exists():
                    dataset_dir = candidate
                    break
            
            if dataset_dir is None:
                print(f"\n⚠ Dataset directory not found for: {dataset_name}")
                counts[dataset_name] = 0
                continue
            
            # Create converter and run
            converter_class = self.CONVERTERS[dataset_name]
            converter = converter_class(dataset_dir, self.output_dir / "temp" / dataset_name)
            
            try:
                samples = converter.convert()
                
                if max_samples_per_dataset and len(samples) > max_samples_per_dataset:
                    import random
                    random.shuffle(samples)
                    samples = samples[:max_samples_per_dataset]
                
                all_samples.extend(samples)
                counts[dataset_name] = len(samples)
            except Exception as e:
                print(f"\n✗ Error converting {dataset_name}: {e}")
                counts[dataset_name] = 0
        
        # Split into train/val
        print(f"\n{'='*60}")
        print(f"Splitting {len(all_samples)} samples into train/val...")
        
        import random
        random.shuffle(all_samples)
        
        val_count = int(len(all_samples) * val_split)
        val_samples = all_samples[:val_count]
        train_samples = all_samples[val_count:]
        
        # Move files to final structure
        self._organize_split(train_samples, "train")
        self._organize_split(val_samples, "val")
        
        # Create manifest
        manifest = {
            "total_samples": len(all_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "datasets": counts,
            "structure": {
                "images": "RGB images (PNG/JPG)",
                "masks": "Binary masks (0=bg, 1=text)",
                "instances": "Instance masks (unique ID per region)",
                "polygons": "JSON with polygon coordinates"
            }
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Cleanup temp directory
        temp_dir = self.output_dir / "temp"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Merge Summary:")
        print(f"{'='*60}")
        for name, count in counts.items():
            print(f"  {name}: {count} samples")
        print(f"\n  Total: {len(all_samples)} samples")
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val: {len(val_samples)} samples")
        print(f"\n  Output: {self.output_dir}")
        
        return counts
    
    def _organize_split(self, samples: List[UnifiedSample], split: str):
        """Organize samples into split directory structure."""
        split_dir = self.output_dir / split
        
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"
        instances_dir = split_dir / "instances"
        polygons_dir = split_dir / "polygons"
        
        for d in [images_dir, masks_dir, instances_dir, polygons_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        for sample in tqdm(samples, desc=f"Organizing {split}"):
            # Move files from temp location to final
            source_dataset = sample.source_dataset
            temp_base = self.output_dir / "temp" / source_dataset
            
            # Find and move image
            for ext in ['.jpg', '.jpeg', '.png']:
                src_img = temp_base / "images" / f"{sample.image_id}{ext}"
                if src_img.exists():
                    dst_img = images_dir / f"{sample.image_id}{ext}"
                    src_img.rename(dst_img)
                    break
            
            # Move mask
            src_mask = temp_base / "masks" / f"{sample.image_id}.png"
            if src_mask.exists():
                dst_mask = masks_dir / f"{sample.image_id}.png"
                src_mask.rename(dst_mask)
            
            # Move instance mask
            src_inst = temp_base / "instances" / f"{sample.image_id}.png"
            if src_inst.exists():
                dst_inst = instances_dir / f"{sample.image_id}.png"
                src_inst.rename(dst_inst)
            
            # Move polygon JSON
            src_poly = temp_base / "polygons" / f"{sample.image_id}.json"
            if src_poly.exists():
                dst_poly = polygons_dir / f"{sample.image_id}.json"
                src_poly.rename(dst_poly)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert and merge text detection datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Merge all datasets
    python dataset_merge.py --input_dir ./raw_datasets --output_dir ./merged
    
    # Merge specific datasets
    python dataset_merge.py --input_dir ./raw_datasets --output_dir ./merged --datasets hiertext cord
    
    # Limit samples per dataset
    python dataset_merge.py --input_dir ./raw_datasets --output_dir ./merged --max-samples 5000
"""
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        default="./raw_datasets",
        help="Directory containing downloaded datasets"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./merged_dataset",
        help="Output directory for merged dataset"
    )
    
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["all"],
        choices=["all", "hiertext", "cocotext", "textocr", "clappertext", "cord"],
        help="Datasets to merge"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of samples for validation (default: 0.1)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for testing)"
    )
    
    args = parser.parse_args()
    
    merger = DatasetMerger(args.input_dir, args.output_dir)
    merger.merge(args.datasets, args.val_split, args.max_samples)


if __name__ == "__main__":
    main()
