"""
Dataset Downloader for Text Detection Training.

Downloads and extracts the following datasets with FULL IMAGE SUPPORT:
- HierText (Google's hierarchical text dataset) + Open Images images
- COCO-Text v2 (Text annotations for COCO images) + COCO 2014 images
- TextOCR (Facebook's OCR dataset) + Open Images images
- ClapperText (Movie clapper text dataset from Zenodo)
- CORD-v2 (Naver's receipt OCR dataset from HuggingFace)

This script downloads ALL required files including images.

Usage:
    python dataset_downloader.py --output_dir ./raw_datasets --datasets all
    python dataset_downloader.py --output_dir ./raw_datasets --datasets hiertext cocotext
"""

import os
import sys
import json
import gzip
import shutil
import argparse
import zipfile
import tarfile
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# Try importing optional dependencies
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not available. CORD-v2 download will be skipped.")
    print("Install with: pip install datasets")

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("Warning: 'gdown' library not available. Some Google Drive downloads may fail.")
    print("Install with: pip install gdown")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: 'Pillow' not available. Some image processing may fail.")
    print("Install with: pip install Pillow")


# ============================================================================
# Dataset Download URLs and Configurations
# ============================================================================

DATASET_CONFIGS = {
    "hiertext": {
        "name": "HierText",
        "description": "Google's hierarchical text detection dataset (~12K images)",
        "urls": {
            "train": "https://github.com/google-research-datasets/hiertext/raw/main/gt/train.jsonl.gz",
            "val": "https://github.com/google-research-datasets/hiertext/raw/main/gt/validation.jsonl.gz",
        },
        "requires_open_images": True,
        "format": "jsonl",
    },
    "cocotext": {
        "name": "COCO-Text v2",
        "description": "Text annotations for MS COCO images (~63K images, 19GB)",
        "urls": {
            "annotations": "https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip",
            # COCO 2014 images - these ARE required
            "train_images": "http://images.cocodataset.org/zips/train2014.zip",
            "val_images": "http://images.cocodataset.org/zips/val2014.zip",
        },
        "format": "json",
    },
    "textocr": {
        "name": "TextOCR",
        "description": "Facebook's large-scale OCR dataset (~28K images)",
        "urls": {
            "annotations_train": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json",
            "annotations_val": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json",
        },
        "requires_open_images": True,
        "format": "json",
    },
    "clappertext": {
        "name": "ClapperText",
        "description": "Movie clapper text dataset (~94K word instances)",
        "urls": {
            # Official ClapperText from Zenodo (linty5/ClapperText)
            "dataset": "https://zenodo.org/records/17366964/files/ClapperText_v1.0.0.zip?download=1",
            # Fallback GitHub repo
            "github": "https://github.com/linty5/ClapperText/archive/refs/heads/main.zip",
        },
        "format": "zenodo",
    },
    "cord": {
        "name": "CORD-v2",
        "description": "Naver's Consolidated Receipt Dataset (~11K images)",
        "huggingface_id": "naver-clova-ix/cord-v2",
        "format": "huggingface",
    },
}

# Open Images base URLs
OPEN_IMAGES_BASE_URLS = {
    "s3": "https://s3.amazonaws.com/open-images-dataset",
    "gcs": "https://storage.googleapis.com/cvdf-datasets/oid",
}


# ============================================================================
# Download Utilities
# ============================================================================

def download_file(url: str, dest_path: Path, chunk_size: int = 8192, 
                  desc: Optional[str] = None) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: Source URL
        dest_path: Destination file path
        chunk_size: Download chunk size
        desc: Progress bar description
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        desc = desc or dest_path.name
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_gdrive(file_id: str, dest_path: Path, desc: Optional[str] = None) -> bool:
    """Download file from Google Drive using gdown."""
    if not GDOWN_AVAILABLE:
        print(f"Cannot download from Google Drive: gdown not installed")
        return False
    
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(dest_path), quiet=False)
        return dest_path.exists()
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """
    Extract zip or tar archive.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
        
    Returns:
        True if successful
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            mode = 'r:gz' if archive_path.suffix in ['.gz', '.tgz'] else 'r'
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(extract_dir)
        else:
            print(f"Unknown archive format: {archive_path.suffix}")
            return False
        return True
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False


def decompress_gzip(gzip_path: Path, output_path: Path) -> bool:
    """Decompress a gzip file."""
    import gzip as gz
    try:
        with gz.open(gzip_path, 'rb') as f_in:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"Error decompressing {gzip_path}: {e}")
        return False


# ============================================================================
# Dataset-Specific Downloaders
# ============================================================================

class BaseDownloader:
    """Base class for dataset downloaders."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        self.output_dir = output_dir
        self.config = config
        self.dataset_dir = output_dir / config["name"].lower().replace(" ", "_").replace("-", "_")
    
    def download(self) -> bool:
        """Download the dataset. Override in subclasses."""
        raise NotImplementedError
    
    def verify(self) -> bool:
        """Verify the download is complete."""
        return self.dataset_dir.exists()


class HierTextDownloader(BaseDownloader):
    """Download HierText dataset with images from Open Images."""
    
    def download(self) -> bool:
        print(f"\n{'='*60}")
        print(f"Downloading HierText Dataset")
        print(f"{'='*60}")
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        gt_dir = self.dataset_dir / "ground_truth"
        gt_dir.mkdir(exist_ok=True)
        
        # Download annotations
        image_ids_by_split = {}
        for split, url in [("train", self.config["urls"]["train"]), 
                           ("validation", self.config["urls"]["val"])]:
            gz_path = gt_dir / f"{split}.jsonl.gz"
            jsonl_path = gt_dir / f"{split}.jsonl"
            
            if not jsonl_path.exists():
                print(f"  Downloading {split} annotations...")
                if download_file(url, gz_path, desc=f"HierText {split}"):
                    decompress_gzip(gz_path, jsonl_path)
                    if gz_path.exists():
                        gz_path.unlink()
            
            # Extract image IDs from annotations
            if jsonl_path.exists():
                image_ids = set()
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            # HierText uses image_id field
                            if 'image_id' in data:
                                image_ids.add(data['image_id'])
                        except json.JSONDecodeError:
                            continue
                image_ids_by_split[split] = image_ids
                print(f"    Found {len(image_ids)} images for {split}")
        
        # Download images from Open Images
        images_dir = self.dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        total_images = sum(len(ids) for ids in image_ids_by_split.values())
        print(f"\n  Downloading {total_images} images from Open Images...")
        
        for split, image_ids in image_ids_by_split.items():
            split_dir = images_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Map HierText split names to Open Images split names
            oi_split = "train" if split == "train" else "validation"
            
            self._download_open_images(image_ids, split_dir, oi_split)
        
        print(f"  ✓ HierText download complete")
        return True
    
    def _download_open_images(self, image_ids: Set[str], output_dir: Path, 
                               split: str, max_workers: int = 8) -> int:
        """Download images from Open Images dataset."""
        
        def download_single(image_id: str) -> bool:
            dest = output_dir / f"{image_id}.jpg"
            if dest.exists():
                return True
            
            # Try S3 first, then GCS
            urls = [
                f"{OPEN_IMAGES_BASE_URLS['s3']}/{split}/{image_id}.jpg",
                f"https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg",
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(dest, 'wb') as f:
                            f.write(response.content)
                        return True
                except:
                    continue
            return False
        
        # Filter out already downloaded
        to_download = [img_id for img_id in image_ids 
                       if not (output_dir / f"{img_id}.jpg").exists()]
        
        if not to_download:
            print(f"    All {len(image_ids)} images already downloaded for {split}")
            return len(image_ids)
        
        print(f"    Downloading {len(to_download)} images for {split}...")
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, img_id): img_id 
                      for img_id in to_download}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc=f"Open Images {split}"):
                if future.result():
                    success_count += 1
        
        already_had = len(image_ids) - len(to_download)
        print(f"    Downloaded {success_count}/{len(to_download)} new images "
              f"({already_had} already existed)")
        return success_count + already_had


class COCOTextDownloader(BaseDownloader):
    """Download COCO-Text v2 dataset with COCO 2014 images."""
    
    def download(self) -> bool:
        print(f"\n{'='*60}")
        print(f"Downloading COCO-Text v2 Dataset")
        print(f"{'='*60}")
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download annotations
        annotations_zip = self.dataset_dir / "cocotext.v2.zip"
        annotations_dir = self.dataset_dir / "annotations"
        
        if not annotations_dir.exists() or not any(annotations_dir.iterdir()):
            print("  Downloading annotations...")
            if download_file(self.config["urls"]["annotations"], annotations_zip, 
                           desc="COCO-Text annotations"):
                extract_archive(annotations_zip, annotations_dir)
                if annotations_zip.exists():
                    annotations_zip.unlink()
        else:
            print("  Annotations already exist, skipping...")
        
        # ALWAYS download COCO images - they are required
        print("\n  Downloading COCO 2014 images (required, ~19GB total)...")
        print("  This may take a while depending on your connection...")
        self._download_coco_images()
        
        print(f"  ✓ COCO-Text v2 download complete")
        return True
    
    def _download_coco_images(self) -> bool:
        """Download COCO 2014 images."""
        images_dir = self.dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for split, url in [("train", self.config["urls"]["train_images"]),
                           ("val", self.config["urls"]["val_images"])]:
            zip_path = images_dir / f"{split}2014.zip"
            extract_dir = images_dir / f"{split}2014"
            
            if extract_dir.exists() and any(extract_dir.glob("*.jpg")):
                existing = len(list(extract_dir.glob("*.jpg")))
                print(f"    {split}2014: {existing} images already exist, skipping...")
                continue
            
            print(f"    Downloading {split}2014 images...")
            if download_file(url, zip_path, desc=f"COCO {split}2014"):
                print(f"    Extracting {split}2014...")
                extract_archive(zip_path, images_dir)
                if zip_path.exists():
                    zip_path.unlink()
        
        return True


class TextOCRDownloader(BaseDownloader):
    """Download TextOCR dataset with images from Open Images."""
    
    def download(self) -> bool:
        print(f"\n{'='*60}")
        print(f"Downloading TextOCR Dataset")
        print(f"{'='*60}")
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir = self.dataset_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        
        # Download annotations and collect image IDs
        image_ids_by_split = {}
        
        for split in ["train", "val"]:
            url = self.config["urls"][f"annotations_{split}"]
            dest = annotations_dir / f"TextOCR_0.1_{split}.json"
            
            if not dest.exists():
                print(f"  Downloading {split} annotations...")
                download_file(url, dest, desc=f"TextOCR {split}")
            else:
                print(f"  {split} annotations already exist")
            
            # Extract image IDs from annotations
            if dest.exists():
                image_ids = set()
                try:
                    with open(dest, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # TextOCR format: has 'imgs' dict with image_id -> image_info
                        if 'imgs' in data:
                            for img_id, img_info in data['imgs'].items():
                                # Image IDs in TextOCR reference Open Images
                                image_ids.add(img_id)
                        # Also check 'data' key
                        elif 'data' in data:
                            for item in data['data']:
                                if 'image_id' in item:
                                    image_ids.add(str(item['image_id']))
                except Exception as e:
                    print(f"    Warning: Could not parse {split} annotations: {e}")
                
                image_ids_by_split[split] = image_ids
                print(f"    Found {len(image_ids)} images for {split}")
        
        # Download images from Open Images
        images_dir = self.dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        total_images = sum(len(ids) for ids in image_ids_by_split.values())
        print(f"\n  Downloading {total_images} images from Open Images...")
        
        # TextOCR uses train split from Open Images for both train and val
        all_image_ids = set()
        for ids in image_ids_by_split.values():
            all_image_ids.update(ids)
        
        self._download_open_images(all_image_ids, images_dir, "train")
        
        print(f"  ✓ TextOCR download complete")
        return True
    
    def _download_open_images(self, image_ids: Set[str], output_dir: Path, 
                               split: str, max_workers: int = 8) -> int:
        """Download images from Open Images dataset."""
        
        def download_single(image_id: str) -> bool:
            dest = output_dir / f"{image_id}.jpg"
            if dest.exists():
                return True
            
            # Try multiple URL patterns
            urls = [
                f"https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg",
                f"https://s3.amazonaws.com/open-images-dataset/train/{image_id}.jpg",
                f"https://s3.amazonaws.com/open-images-dataset/validation/{image_id}.jpg",
                f"https://s3.amazonaws.com/open-images-dataset/test/{image_id}.jpg",
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(dest, 'wb') as f:
                            f.write(response.content)
                        return True
                except:
                    continue
            return False
        
        # Filter out already downloaded
        to_download = [img_id for img_id in image_ids 
                       if not (output_dir / f"{img_id}.jpg").exists()]
        
        if not to_download:
            print(f"    All {len(image_ids)} images already downloaded")
            return len(image_ids)
        
        print(f"    Downloading {len(to_download)} images...")
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, img_id): img_id 
                      for img_id in to_download}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Open Images"):
                if future.result():
                    success_count += 1
        
        already_had = len(image_ids) - len(to_download)
        print(f"    Downloaded {success_count}/{len(to_download)} new images "
              f"({already_had} already existed)")
        return success_count + already_had


class ClapperTextDownloader(BaseDownloader):
    """Download ClapperText dataset from Zenodo."""
    
    def download(self) -> bool:
        print(f"\n{'='*60}")
        print(f"Downloading ClapperText Dataset")
        print(f"{'='*60}")
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Try Zenodo download first (official source)
        zip_path = self.dataset_dir / "clappertext.zip"
        success = False
        
        print("  Attempting to download from Zenodo (official)...")
        success = download_file(
            self.config["urls"]["dataset"], 
            zip_path,
            desc="ClapperText Zenodo"
        )
        
        # Fallback to GitHub if Zenodo fails
        if not success or not zip_path.exists():
            print("  Zenodo download failed, trying GitHub fallback...")
            success = download_file(
                self.config["urls"]["github"], 
                zip_path,
                desc="ClapperText GitHub"
            )
        
        if success and zip_path.exists():
            print("  Extracting...")
            extract_archive(zip_path, self.dataset_dir)
            if zip_path.exists():
                zip_path.unlink()
            
            # Organize files if needed
            self._organize_files()
            
            print("  ✓ ClapperText downloaded")
            return True
        else:
            # Create manual download instructions
            instructions_path = self.dataset_dir / "MANUAL_DOWNLOAD.md"
            with open(instructions_path, 'w') as f:
                f.write("""# ClapperText Manual Download

The automatic download failed. Please download manually:

## Option 1: Zenodo (Official)
1. Visit: https://zenodo.org/records/17366964
2. Download ClapperText_v1.0.0.zip
3. Extract to this directory

## Option 2: GitHub
1. Visit: https://github.com/linty5/ClapperText
2. Clone or download the repository
3. Follow their instructions to access the data

## Option 3: HISTORIAN Source Videos
ClapperText is derived from the HISTORIAN dataset:
1. Visit: https://zenodo.org/record/6644516
2. Download source videos and extract frames

Expected structure after download:
- {dataset_dir}/images/
- {dataset_dir}/annotations/
""")
            print(f"  ⚠ Auto-download failed. See: {instructions_path}")
            return False
    
    def _organize_files(self):
        """Organize extracted files into standard structure."""
        images_dir = self.dataset_dir / "images"
        annotations_dir = self.dataset_dir / "annotations"
        
        # Look for common extracted folder patterns
        for extracted_dir in self.dataset_dir.iterdir():
            if extracted_dir.is_dir() and extracted_dir.name not in ['images', 'annotations']:
                # Check if this contains images or annotations
                if (extracted_dir / 'images').exists():
                    if not images_dir.exists():
                        shutil.move(str(extracted_dir / 'images'), str(images_dir))
                if (extracted_dir / 'annotations').exists():
                    if not annotations_dir.exists():
                        shutil.move(str(extracted_dir / 'annotations'), str(annotations_dir))
                
                # Also check for image files directly
                image_files = list(extracted_dir.glob('*.jpg')) + list(extracted_dir.glob('*.png'))
                if image_files and not images_dir.exists():
                    images_dir.mkdir(exist_ok=True)
                    for img in image_files:
                        shutil.move(str(img), str(images_dir / img.name))
                
                # Check for JSON annotation files
                json_files = list(extracted_dir.glob('*.json'))
                if json_files and not annotations_dir.exists():
                    annotations_dir.mkdir(exist_ok=True)
                    for jf in json_files:
                        shutil.move(str(jf), str(annotations_dir / jf.name))


class CORDDownloader(BaseDownloader):
    """Download CORD-v2 dataset from HuggingFace."""
    
    def download(self) -> bool:
        print(f"\n{'='*60}")
        print(f"Downloading CORD-v2 Dataset")
        print(f"{'='*60}")
        
        if not HF_AVAILABLE:
            print("  ⚠ HuggingFace datasets library not available")
            print("  Install with: pip install datasets")
            return False
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print("  Loading from HuggingFace Hub...")
            dataset = load_dataset(self.config["huggingface_id"])
            
            # Save to disk
            for split in dataset.keys():
                split_dir = self.dataset_dir / split
                split_dir.mkdir(exist_ok=True)
                
                images_dir = split_dir / "images"
                annotations_dir = split_dir / "annotations"
                images_dir.mkdir(exist_ok=True)
                annotations_dir.mkdir(exist_ok=True)
                
                print(f"  Processing {split} split ({len(dataset[split])} samples)...")
                
                for idx, sample in enumerate(tqdm(dataset[split], desc=f"CORD {split}")):
                    # Save image
                    image = sample["image"]
                    image_path = images_dir / f"{idx:06d}.png"
                    image.save(image_path)
                    
                    # Save ground truth
                    gt = sample.get("ground_truth", {})
                    gt_path = annotations_dir / f"{idx:06d}.json"
                    with open(gt_path, 'w') as f:
                        # Handle both string and dict ground_truth
                        if isinstance(gt, str):
                            try:
                                gt = json.loads(gt)
                            except:
                                gt = {"raw": gt}
                        json.dump(gt, f, indent=2)
            
            print("  ✓ CORD-v2 downloaded and extracted")
            return True
            
        except Exception as e:
            print(f"  Error downloading CORD-v2: {e}")
            return False


# ============================================================================
# Open Images Downloader (shared by HierText and TextOCR)
# ============================================================================

class OpenImagesDownloader:
    """Download images from Open Images dataset."""
    
    def __init__(self, output_dir: Path, image_ids: List[str]):
        self.output_dir = output_dir
        self.image_ids = image_ids
        self.base_url = "https://s3.amazonaws.com/open-images-dataset"
    
    def download(self, split: str = "train", max_workers: int = 8) -> bool:
        """
        Download specific images from Open Images.
        
        Args:
            split: 'train', 'validation', or 'test'
            max_workers: Number of parallel download threads
        """
        images_dir = self.output_dir / split
        images_dir.mkdir(parents=True, exist_ok=True)
        
        def download_single(image_id: str) -> bool:
            url = f"{self.base_url}/{split}/{image_id}.jpg"
            dest = images_dir / f"{image_id}.jpg"
            if dest.exists():
                return True
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(dest, 'wb') as f:
                        f.write(response.content)
                    return True
            except:
                pass
            return False
        
        print(f"  Downloading {len(self.image_ids)} images from Open Images {split}...")
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, img_id): img_id 
                      for img_id in self.image_ids}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Open Images"):
                if future.result():
                    success_count += 1
        
        print(f"  Downloaded {success_count}/{len(self.image_ids)} images")
        return success_count > 0


# ============================================================================
# Main Downloader Orchestrator
# ============================================================================

class DatasetDownloader:
    """Main orchestrator for downloading all datasets."""
    
    DOWNLOADERS = {
        "hiertext": HierTextDownloader,
        "cocotext": COCOTextDownloader,
        "textocr": TextOCRDownloader,
        "clappertext": ClapperTextDownloader,
        "cord": CORDDownloader,
    }
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, datasets: List[str], download_images: bool = False) -> Dict[str, bool]:
        """
        Download specified datasets.
        
        Args:
            datasets: List of dataset names to download, or ["all"]
            download_images: Whether to download large image files
            
        Returns:
            Dictionary of dataset_name -> success status
        """
        if "all" in datasets:
            datasets = list(self.DOWNLOADERS.keys())
        
        results = {}
        
        print(f"\n{'#'*60}")
        print(f"# Text Detection Dataset Downloader")
        print(f"# Output directory: {self.output_dir}")
        print(f"# Datasets: {', '.join(datasets)}")
        print(f"{'#'*60}")
        
        for dataset_name in datasets:
            if dataset_name not in self.DOWNLOADERS:
                print(f"\n⚠ Unknown dataset: {dataset_name}")
                print(f"  Available: {', '.join(self.DOWNLOADERS.keys())}")
                results[dataset_name] = False
                continue
            
            config = DATASET_CONFIGS[dataset_name]
            downloader_class = self.DOWNLOADERS[dataset_name]
            downloader = downloader_class(self.output_dir, config)
            
            try:
                success = downloader.download()
                
                # Download images if requested and available
                if success and download_images and hasattr(downloader, 'download_images'):
                    downloader.download_images()
                
                results[dataset_name] = success
            except Exception as e:
                print(f"\n✗ Error downloading {dataset_name}: {e}")
                results[dataset_name] = False
        
        # Print summary
        print(f"\n{'='*60}")
        print("Download Summary:")
        print(f"{'='*60}")
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
        
        return results
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all datasets."""
        status = {}
        
        for name, config in DATASET_CONFIGS.items():
            dataset_dir = self.output_dir / config["name"].lower().replace(" ", "_").replace("-", "_")
            status[name] = {
                "name": config["name"],
                "description": config["description"],
                "downloaded": dataset_dir.exists(),
                "path": str(dataset_dir),
            }
        
        return status


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download text detection datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all datasets
    python dataset_downloader.py --output_dir ./raw_datasets --datasets all
    
    # Download specific datasets
    python dataset_downloader.py --output_dir ./raw_datasets --datasets hiertext cord
    
    # Download with COCO images (large)
    python dataset_downloader.py --output_dir ./raw_datasets --datasets cocotext --download-images
    
    # Check download status
    python dataset_downloader.py --output_dir ./raw_datasets --status
"""
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./raw_datasets",
        help="Output directory for downloaded datasets"
    )
    
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["all"],
        choices=["all", "hiertext", "cocotext", "textocr", "clappertext", "cord"],
        help="Datasets to download"
    )
    
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download large image files (COCO, Open Images)"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show download status and exit"
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output_dir)
    
    if args.status:
        status = downloader.get_status()
        print("\nDataset Status:")
        print("="*60)
        for name, info in status.items():
            status_icon = "✓" if info["downloaded"] else "✗"
            print(f"  {status_icon} {info['name']}")
            print(f"      {info['description']}")
            print(f"      Path: {info['path']}")
        return
    
    results = downloader.download(args.datasets, args.download_images)
    
    # Exit with error if any download failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
