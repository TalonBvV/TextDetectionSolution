# Text Detection Dataset Generation Pipeline

Complete pipeline to generate training data for the PatchFormer text detection model.

## Overview

This pipeline downloads, converts, merges, augments, and prepares text detection datasets for training. It produces a unified dataset format optimized for the PatchFormer v2 model.

**The download script automatically downloads ALL required files including images.**

## Supported Datasets

| Dataset | Description | Source | Size |
|---------|-------------|--------|------|
| **HierText** | Google's hierarchical text dataset | Open Images | ~12K images |
| **COCO-Text v2** | Text annotations for MS COCO | COCO 2014 | ~63K images, 19GB |
| **TextOCR** | Facebook's large-scale OCR dataset | Open Images | ~28K images |
| **ClapperText** | Movie clapper text dataset | Zenodo | ~94K word instances |
| **CORD-v2** | Naver's receipt OCR dataset | HuggingFace | ~11K images |

## Installation

```bash
cd DatasetGeneration
pip install -r requirements.txt
```

## Quick Start

### Run Full Pipeline
```bash
python gen_dataset.py --output_dir ./dataset --run-all
```

### Run with Specific Datasets
```bash
python gen_dataset.py --output_dir ./dataset --run-all --datasets hiertext cord
```

### Quick Test (Limited Samples)
```bash
python gen_dataset.py --output_dir ./dataset --run-all --max-samples 100
```

## Pipeline Stages

### 1. Download
Downloads datasets from their sources **including all images**.

```bash
python dataset_downloader.py --output_dir ./raw_datasets --datasets all
```

**What gets downloaded (from official sources):**
- **HierText**: Annotations (GitHub) + images from CVDF OCR bucket as .tgz (~12K images)
  - Source: `s3://open-images-dataset/ocr/{train,validation,test}.tgz`
- **COCO-Text**: Annotations (GitHub) + COCO 2014 images (~63K images, ~19GB)
  - Source: `images.cocodataset.org`
- **TextOCR**: Annotations + images from Facebook servers (~28K images)
  - Source: `dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip`
- **ClapperText**: Full dataset from Zenodo
  - Source: `zenodo.org/records/17366964`
- **CORD-v2**: Full dataset from HuggingFace (~11K images)
  - Source: HuggingFace Hub

⚠️ **Note:** Full download requires ~60GB+ of disk space and may take several hours.

### 2. Merge
Converts each dataset to unified format and combines them.

```bash
python dataset_merge.py --input_dir ./raw_datasets --output_dir ./merged
```

### 3. Augment
Applies augmentations (flip, rotate, mosaic) to generate 2x samples.

```bash
python dataset_augment.py --input_dir ./merged --output_dir ./augmented --multiplier 2.0
```

### 4. Finalize
Combines merged and augmented data, resizes to target resolution, validates, and creates final splits.

```bash
python dataset_finalize.py --merged_dir ./merged --augmented_dir ./augmented --output_dir ./final
```

## Output Format

The final dataset follows this structure:

```
final/
├── train/
│   ├── images/      # RGB images (JPEG/PNG, ≥1024px)
│   ├── masks/       # Binary masks (PNG, 0=bg, 1=text)
│   ├── instances/   # Instance masks (16-bit PNG, unique ID per region)
│   └── polygons/    # JSON annotations with vertex coordinates
├── val/
│   └── ...
├── test/
│   └── ...
├── manifest.json    # Dataset metadata
└── splits.json      # Sample IDs per split
```

### Data Specifications

| Component | Format | Details |
|-----------|--------|---------|
| **Images** | RGB JPEG/PNG | Minimum 1024px on shorter side |
| **Masks** | 8-bit PNG | 0=background, 1=text |
| **Instances** | 16-bit PNG | 0=background, 1,2,3...=unique instance IDs |
| **Polygons** | JSON | Vertex coordinates, text content, metadata |

### Polygon JSON Format

```json
{
  "image_id": "hiertext_train_abc123",
  "source_dataset": "hiertext",
  "width": 1024,
  "height": 768,
  "polygons": [
    {
      "points": [[100, 50], [200, 50], [200, 80], [100, 80]],
      "text": "HELLO",
      "legibility": "legible",
      "language": "en",
      "word_type": "word"
    }
  ]
}
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output_dir` | ./dataset | Base output directory |
| `--datasets` | all | Datasets to include |
| `--val-split` | 0.1 | Validation set fraction |
| `--test-split` | 0.1 | Test set fraction |
| `--augmentation` | 2.0 | Augmentation multiplier |
| `--target-size` | 1024 | Minimum image size |
| `--max-samples` | None | Limit samples per dataset |
| `--workers` | 4 | Parallel workers |
| `--download-images` | False | Download large image files |
| `--cleanup` | False | Remove intermediate files |

## Augmentation Strategies

The pipeline applies these augmentation strategies:

1. **Horizontal Flip** - Mirror along vertical axis
2. **Vertical Flip** - Mirror along horizontal axis
3. **90° Rotation** - Rotate 90° counter-clockwise
4. **180° Rotation** - Rotate 180°
5. **270° Rotation** - Rotate 270° counter-clockwise
6. **Random Rotation** - Small random rotation (±15°)
7. **Flip + Rotation** - Combined transformations
8. **Mosaic** - Combine 4 images into one

All augmentations properly transform:
- Images
- Binary masks
- Instance masks
- Polygon coordinates

## Integration with PatchFormer

The output dataset is ready for use with the PatchFormer model:

```python
from src.dataset import SegmentationDataset
from src.config import get_config

config = get_config("small", num_classes=2)
dataset = SegmentationDataset(config, augment=True)

train_ds = dataset.create_dataset(
    image_paths=train_images,
    mask_paths=train_masks,
    batch_size=4
)
```

## Troubleshooting

### Missing Images
Some datasets require manual image downloads:
- **HierText/TextOCR**: Require Open Images dataset
- **COCO-Text**: Requires COCO 2014 images

Check the `DOWNLOAD_IMAGES.md` files in each dataset's raw directory.

### Memory Issues
For large datasets, use `--max-samples` to limit processing:
```bash
python gen_dataset.py --run-all --max-samples 5000
```

### Resume Failed Pipeline
The pipeline saves config, allowing resume:
```bash
python gen_dataset.py --resume ./dataset/logs/pipeline_config.json --stages merge augment finalize
```

## License

This pipeline is provided for research purposes. Individual datasets have their own licenses:
- HierText: CC BY 4.0
- COCO-Text: Creative Commons
- TextOCR: CC BY 4.0
- CORD-v2: MIT License

Please respect the licensing terms of each dataset.
