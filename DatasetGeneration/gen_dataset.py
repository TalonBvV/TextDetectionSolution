"""
Text Detection Dataset Generation Pipeline.

Complete pipeline to generate training data for the PatchFormer text detection model.

Pipeline stages:
1. Download: Fetch datasets (HierText, COCO-Text, TextOCR, ClapperText, CORD-v2)
2. Merge: Convert to unified format and combine
3. Augment: Apply augmentations (flip, rotate, mosaic) for 2x samples
4. Finalize: Combine and prepare final training-ready dataset

Output format:
- Images: RGB, ≥1024px resolution
- Masks: Binary PNG (0=background, 1=text)
- Instances: Integer PNG (unique ID per text region)
- Polygons: JSON with vertex coordinates

Usage:
    # Run full pipeline
    python gen_dataset.py --output_dir ./dataset --run-all
    
    # Run specific stages
    python gen_dataset.py --output_dir ./dataset --stages download merge augment finalize
    
    # Quick test with limited samples
    python gen_dataset.py --output_dir ./dataset --run-all --max-samples 100
"""

import os
import sys
import json
import time
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import pipeline components
from dataset_downloader import DatasetDownloader
from dataset_merge import DatasetMerger
from dataset_augment import DatasetAugmenter
from dataset_finalize import DatasetFinalizer


# ============================================================================
# Pipeline Configuration
# ============================================================================

class PipelineConfig:
    """Configuration for the dataset generation pipeline."""
    
    def __init__(
        self,
        output_dir: str = "./dataset",
        datasets: List[str] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        augmentation_multiplier: float = 2.0,
        target_size: int = 1024,
        max_samples_per_dataset: Optional[int] = None,
        download_images: bool = False,
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.datasets = datasets or ["all"]
        self.val_split = val_split
        self.test_split = test_split
        self.augmentation_multiplier = augmentation_multiplier
        self.target_size = target_size
        self.max_samples_per_dataset = max_samples_per_dataset
        self.download_images = download_images
        self.num_workers = num_workers
        
        # Derived paths
        self.raw_dir = self.output_dir / "raw_datasets"
        self.merged_dir = self.output_dir / "merged"
        self.augmented_dir = self.output_dir / "augmented"
        self.final_dir = self.output_dir / "final"
        self.logs_dir = self.output_dir / "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "datasets": self.datasets,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "augmentation_multiplier": self.augmentation_multiplier,
            "target_size": self.target_size,
            "max_samples_per_dataset": self.max_samples_per_dataset,
            "download_images": self.download_images,
            "num_workers": self.num_workers,
        }
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'PipelineConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# Pipeline Runner
# ============================================================================

class DatasetPipeline:
    """
    Main pipeline orchestrator for dataset generation.
    
    Runs the complete pipeline:
    1. Download datasets
    2. Merge and convert to unified format
    3. Apply augmentations
    4. Finalize for training
    """
    
    STAGES = ["download", "merge", "augment", "finalize"]
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = None
        self.stage_times = {}
        
        # Create output directories
        for d in [self.config.output_dir, self.config.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def run(self, stages: List[str] = None) -> Dict[str, Any]:
        """
        Run the pipeline.
        
        Args:
            stages: List of stages to run, or None for all stages
            
        Returns:
            Results dictionary with statistics from each stage
        """
        if stages is None:
            stages = self.STAGES
        
        # Validate stages
        for stage in stages:
            if stage not in self.STAGES:
                raise ValueError(f"Unknown stage: {stage}. Valid stages: {self.STAGES}")
        
        self.start_time = time.time()
        results = {"stages": {}, "config": self.config.to_dict()}
        
        print("\n" + "=" * 70)
        print("  TEXT DETECTION DATASET GENERATION PIPELINE")
        print("=" * 70)
        print(f"\n  Output directory: {self.config.output_dir}")
        print(f"  Datasets: {', '.join(self.config.datasets)}")
        print(f"  Stages: {', '.join(stages)}")
        print(f"  Target size: {self.config.target_size}px")
        print(f"  Augmentation: {self.config.augmentation_multiplier}x")
        print("\n" + "=" * 70)
        
        # Save config
        config_path = self.config.logs_dir / "pipeline_config.json"
        self.config.save(config_path)
        
        try:
            # Run each stage
            if "download" in stages:
                results["stages"]["download"] = self._run_download()
            
            if "merge" in stages:
                results["stages"]["merge"] = self._run_merge()
            
            if "augment" in stages:
                results["stages"]["augment"] = self._run_augment()
            
            if "finalize" in stages:
                results["stages"]["finalize"] = self._run_finalize()
            
            # Calculate total time
            total_time = time.time() - self.start_time
            results["total_time_seconds"] = total_time
            results["total_time_formatted"] = self._format_time(total_time)
            results["success"] = True
            
            # Save results
            results_path = self.config.logs_dir / "pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Print summary
            self._print_summary(results)
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            print(f"\n{'!'*70}")
            print(f"  PIPELINE ERROR: {e}")
            print(f"{'!'*70}")
            raise
        
        return results
    
    def _run_download(self) -> Dict[str, Any]:
        """Run download stage."""
        stage_start = time.time()
        
        print("\n" + "-" * 70)
        print("  STAGE 1: DOWNLOAD")
        print("-" * 70)
        
        downloader = DatasetDownloader(str(self.config.raw_dir))
        download_results = downloader.download(
            self.config.datasets,
            self.config.download_images
        )
        
        stage_time = time.time() - stage_start
        self.stage_times["download"] = stage_time
        
        return {
            "datasets": download_results,
            "output_dir": str(self.config.raw_dir),
            "time_seconds": stage_time,
        }
    
    def _run_merge(self) -> Dict[str, Any]:
        """Run merge stage."""
        stage_start = time.time()
        
        print("\n" + "-" * 70)
        print("  STAGE 2: MERGE")
        print("-" * 70)
        
        merger = DatasetMerger(
            str(self.config.raw_dir),
            str(self.config.merged_dir)
        )
        merge_results = merger.merge(
            self.config.datasets,
            val_split=self.config.val_split,
            max_samples_per_dataset=self.config.max_samples_per_dataset
        )
        
        stage_time = time.time() - stage_start
        self.stage_times["merge"] = stage_time
        
        return {
            "datasets": merge_results,
            "output_dir": str(self.config.merged_dir),
            "time_seconds": stage_time,
        }
    
    def _run_augment(self) -> Dict[str, Any]:
        """Run augmentation stage."""
        stage_start = time.time()
        
        print("\n" + "-" * 70)
        print("  STAGE 3: AUGMENT")
        print("-" * 70)
        
        augmenter = DatasetAugmenter(
            str(self.config.merged_dir),
            str(self.config.augmented_dir)
        )
        augment_results = augmenter.augment(
            target_multiplier=self.config.augmentation_multiplier,
            num_workers=self.config.num_workers
        )
        
        stage_time = time.time() - stage_start
        self.stage_times["augment"] = stage_time
        
        return {
            "statistics": augment_results,
            "output_dir": str(self.config.augmented_dir),
            "time_seconds": stage_time,
        }
    
    def _run_finalize(self) -> Dict[str, Any]:
        """Run finalization stage."""
        stage_start = time.time()
        
        print("\n" + "-" * 70)
        print("  STAGE 4: FINALIZE")
        print("-" * 70)
        
        finalizer = DatasetFinalizer(
            merged_dir=str(self.config.merged_dir),
            augmented_dir=str(self.config.augmented_dir),
            output_dir=str(self.config.final_dir),
            target_size=(self.config.target_size, self.config.target_size)
        )
        finalize_results = finalizer.finalize(
            test_split=self.config.test_split,
            validate_samples=True,
            num_workers=self.config.num_workers
        )
        
        stage_time = time.time() - stage_start
        self.stage_times["finalize"] = stage_time
        
        return {
            "statistics": finalize_results,
            "output_dir": str(self.config.final_dir),
            "time_seconds": stage_time,
        }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print pipeline summary."""
        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE")
        print("=" * 70)
        
        print(f"\n  Total time: {results.get('total_time_formatted', 'N/A')}")
        
        print("\n  Stage times:")
        for stage, time_sec in self.stage_times.items():
            print(f"    {stage}: {self._format_time(time_sec)}")
        
        if "finalize" in results.get("stages", {}):
            stats = results["stages"]["finalize"].get("statistics", {})
            print(f"\n  Final dataset:")
            print(f"    Train samples: {stats.get('train', 0)}")
            print(f"    Val samples: {stats.get('val', 0)}")
            print(f"    Test samples: {stats.get('test', 0)}")
            print(f"    Total: {stats.get('total', 0)}")
        
        print(f"\n  Output location: {self.config.final_dir}")
        print("\n  Dataset structure:")
        print("    final/")
        print("    ├── train/")
        print("    │   ├── images/    # RGB images (≥1024px)")
        print("    │   ├── masks/     # Binary masks (0=bg, 1=text)")
        print("    │   ├── instances/ # Instance masks (unique IDs)")
        print("    │   └── polygons/  # JSON annotations")
        print("    ├── val/")
        print("    ├── test/")
        print("    ├── manifest.json")
        print("    └── splits.json")
        
        print("\n" + "=" * 70)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def cleanup(self, keep_final: bool = True):
        """
        Clean up intermediate files.
        
        Args:
            keep_final: Whether to keep the final dataset
        """
        print("\nCleaning up intermediate files...")
        
        dirs_to_remove = []
        
        if self.config.raw_dir.exists():
            dirs_to_remove.append(self.config.raw_dir)
        
        if self.config.merged_dir.exists():
            dirs_to_remove.append(self.config.merged_dir)
        
        if self.config.augmented_dir.exists():
            dirs_to_remove.append(self.config.augmented_dir)
        
        if not keep_final and self.config.final_dir.exists():
            dirs_to_remove.append(self.config.final_dir)
        
        for d in dirs_to_remove:
            print(f"  Removing: {d}")
            shutil.rmtree(d)
        
        print("  Done.")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate text detection training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python gen_dataset.py --output_dir ./dataset --run-all
    
    # Run specific stages
    python gen_dataset.py --output_dir ./dataset --stages download merge
    
    # Quick test with limited samples
    python gen_dataset.py --output_dir ./dataset --run-all --max-samples 100
    
    # Specify datasets
    python gen_dataset.py --output_dir ./dataset --run-all --datasets hiertext cord
    
    # High augmentation
    python gen_dataset.py --output_dir ./dataset --run-all --augmentation 3.0

Datasets available:
    - hiertext: Google's hierarchical text dataset
    - cocotext: COCO-Text v2 (text in natural images)
    - textocr: Facebook's TextOCR dataset
    - clappertext: Movie clapper text dataset
    - cord: Naver's receipt OCR dataset (CORD-v2)
    - all: All datasets
"""
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./dataset",
        help="Base output directory for all pipeline outputs"
    )
    
    # Stage control
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all pipeline stages"
    )
    
    parser.add_argument(
        "--stages", "-s",
        nargs="+",
        choices=["download", "merge", "augment", "finalize"],
        help="Specific stages to run"
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["all"],
        choices=["all", "hiertext", "cocotext", "textocr", "clappertext", "cord"],
        help="Datasets to include"
    )
    
    # Split configuration
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction for validation set (default: 0.1)"
    )
    
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction for test set (default: 0.1)"
    )
    
    # Augmentation
    parser.add_argument(
        "--augmentation", "-a",
        type=float,
        default=2.0,
        help="Augmentation multiplier (default: 2.0)"
    )
    
    # Image configuration
    parser.add_argument(
        "--target-size",
        type=int,
        default=1024,
        help="Target minimum image size (default: 1024)"
    )
    
    # Resource limits
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for testing)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    # Download options
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download large image files (COCO, Open Images)"
    )
    
    # Cleanup
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove intermediate files after completion"
    )
    
    # Resume from config
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from saved config file"
    )
    
    args = parser.parse_args()
    
    # Determine stages to run
    if args.run_all:
        stages = DatasetPipeline.STAGES
    elif args.stages:
        stages = args.stages
    else:
        parser.error("Must specify --run-all or --stages")
    
    # Create or load config
    if args.resume:
        config = PipelineConfig.load(Path(args.resume))
        print(f"Resuming from config: {args.resume}")
    else:
        config = PipelineConfig(
            output_dir=args.output_dir,
            datasets=args.datasets,
            val_split=args.val_split,
            test_split=args.test_split,
            augmentation_multiplier=args.augmentation,
            target_size=args.target_size,
            max_samples_per_dataset=args.max_samples,
            download_images=args.download_images,
            num_workers=args.workers,
        )
    
    # Run pipeline
    pipeline = DatasetPipeline(config)
    
    try:
        results = pipeline.run(stages)
        
        if args.cleanup and results.get("success"):
            pipeline.cleanup(keep_final=True)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
