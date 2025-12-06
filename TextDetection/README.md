# PatchFormer: Lightweight Semantic Segmentation

A high-resolution semantic segmentation architecture optimized for text detection and CPU/TensorFlow Lite deployment.

## Features

- **Dual-Stream Encoding**: Combines explicit patch features with learned convolutional features
- **Masked Autoencoder Training**: Pixel-perfect reconstruction for robust feature learning
- **Contrastive Embedding Learning**: Structured latent space with cosine similarity alignment
- **Progressive Transformer Encoding**: Multi-stage processing with intermediate supervision
- **Token Classification**: Interior/boundary/background classification for better edge detection
- **Lightweight Design**: Optimized for CPU inference with TFLite compatibility

## Architecture Overview

```
Input (1024×1024×3)
    │
    ├──────────────────┬───────────────────┐
    ▼                  ▼                   │
Patch Extraction    Conv Encoder           │
(8×8 patches)       (Progressive)          │
    │                  │                   │
    ▼                  ▼                   │
(128×128×192)      (128×128×320)          │
    │                  │                   │
    └────────┬─────────┘                   │
             ▼                             │
        Concatenate                        │
      (128×128×512)                        │
             │                             │
             ▼                             │
    2D Positional Encoding                 │
             │                             │
             ▼                             │
    Initial Transformer                    │
             │                             │
    ┌────────┴────────┐                    │
    ▼                 ▼                    │
Reconstruction    Progressive              │
+ Contrastive     Encoding                 │
(Training Only)       │                    │
                      ▼                    │
              Context Generation           │
                      │                    │
                      ▼                    │
            Token Classification           │
                      │                    │
                      ▼                    │
            Final Segmentation             │
                      │                    │
                      ▼                    │
              Lightweight Decoder          │
                      │                    │
                      ▼                    │
        Output (1024×1024×num_classes)     │
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train with your data
python train.py \
    --train-images /path/to/images \
    --train-masks /path/to/masks \
    --val-images /path/to/val/images \
    --val-masks /path/to/val/masks \
    --variant small \
    --num-classes 2 \
    --epochs 100 \
    --batch-size 4

# Test with dummy data
python train.py --use-dummy-data --epochs 5
```

### Export to TFLite

```bash
# Standard export
python export.py \
    --checkpoint checkpoints/final_model \
    --format tflite \
    --output model.tflite

# Quantized export
python export.py \
    --checkpoint checkpoints/final_model \
    --format tflite \
    --quantize \
    --calibration-data /path/to/calibration/images \
    --output model_int8.tflite

# Efficient inference version
python export.py \
    --checkpoint checkpoints/final_model \
    --format tflite-efficient \
    --output model_efficient.tflite
```

### Python API

```python
from src import PatchFormer, get_config

# Create model
config = get_config("small", num_classes=2)
model = PatchFormer(config)

# Training mode
outputs = model(images, training=True)
# Returns dict with: main, reconstruction, contrastive_embeddings, token_logits, etc.

# Inference mode
segmentation = model(images, training=False)
# Returns: (B, H, W, num_classes) logits

# Efficient inference (faster, slightly less accurate)
segmentation = model.predict_efficient(images)
```

## Model Variants

| Variant | Parameters | Target Platform |
|---------|------------|-----------------|
| `tiny` | ~5M | Mobile CPU |
| `small` | ~15M | Desktop CPU |
| `base` | ~30M | GPU |

## Project Structure

```
TextDetectionModel/
├── MODEL_SPEC.md        # Detailed architecture specification
├── README.md            # This file
├── requirements.txt     # Dependencies
├── train.py            # Training script
├── export.py           # Export script
└── src/
    ├── __init__.py
    ├── config.py       # Configuration classes
    ├── layers.py       # Custom layers
    ├── transformer.py  # Transformer blocks
    ├── encoder.py      # Encoder module
    ├── decoder.py      # Decoder module
    ├── losses.py       # Loss functions
    ├── model.py        # Main model class
    ├── dataset.py      # Data pipeline
    └── utils.py        # Utilities
```

## Training Strategy

### Phase 1: Representation Learning (Epochs 1-30)
- Focus on reconstruction and contrastive losses
- High masking ratio (75%)
- Lower segmentation loss weight

### Phase 2: Task-Specific Fine-tuning (Epochs 31+)
- Increase segmentation loss weight
- Reduce reconstruction loss weight
- Lower masking ratio (50%)

### Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| Segmentation (Dice + Focal) | 1.0 | Main segmentation task |
| Token Classification | 0.5 | Boundary awareness |
| Reconstruction | 0.3→0.1 | Feature learning |
| Contrastive | 0.1 | Embedding structure |

## Data Format

### Images
- Format: JPEG or PNG
- Size: Any (will be resized to 1024×1024)
- Channels: RGB

### Masks
- Format: PNG (single channel)
- Values: Class indices (0, 1, 2, ...)
- Size: Same as corresponding image

### Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── masks/
│       ├── img001.png
│       └── ...
└── val/
    ├── images/
    └── masks/
```

## Performance Targets

| Platform | PatchFormer-Tiny | PatchFormer-Small |
|----------|------------------|-------------------|
| Desktop CPU (i7) | 150ms | 400ms |
| Mobile CPU | 500ms | 1200ms |
| TFLite INT8 | 100ms | 300ms |

## Key Design Decisions

1. **8×8 Patches**: Small patches preserve fine detail for precise boundaries
2. **Dual-Stream Encoding**: Combines explicit pixels with learned features
3. **Separate Reconstruction Path**: Training-only path prevents reconstruction from dominating gradients
4. **Token Classification**: Provides geometric priors for boundary detection
5. **Contrastive Learning on Projection Head**: Adds structure without affecting reconstruction

## Citation

If you use this architecture in your work, please cite:

```bibtex
@misc{patchformer2024,
  title={PatchFormer: Lightweight Semantic Segmentation},
  year={2024},
  note={High-resolution semantic segmentation for text detection}
}
```

## License

MIT License
