"""
Evaluation script for PatchFormer semantic segmentation model.
"""

import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from src.config import get_config
from src.model import create_model
from src.dataset import SegmentationDataset
from src.utils import IoUMetric, DiceMetric, BoundaryF1Metric, visualize_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PatchFormer model")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--variant", type=str, default="small",
                       choices=["tiny", "small", "base"])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=1024)
    
    # Data
    parser.add_argument("--images", type=str, required=True,
                       help="Path to test images directory")
    parser.add_argument("--masks", type=str, required=True,
                       help="Path to test masks directory")
    parser.add_argument("--batch-size", type=int, default=1)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="eval_results",
                       help="Output directory for results")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save prediction visualizations")
    
    # Inference mode
    parser.add_argument("--efficient", action="store_true",
                       help="Use efficient inference mode")
    
    return parser.parse_args()


def load_data_paths(images_dir: str, masks_dir: str):
    """Load image and mask file paths."""
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_paths = []
    mask_paths = []
    
    for ext in image_extensions:
        for img_path in images_path.glob(f'*{ext}'):
            mask_path = masks_path / f"{img_path.stem}.png"
            if mask_path.exists():
                image_paths.append(str(img_path))
                mask_paths.append(str(mask_path))
    
    return image_paths, mask_paths


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("Loading model...")
    config = get_config(args.variant, args.num_classes)
    config.input_size = (args.input_size, args.input_size)
    
    model = create_model(
        variant=args.variant,
        num_classes=args.num_classes,
        input_size=(args.input_size, args.input_size),
    )
    model.load_weights(args.checkpoint)
    print("Model loaded")
    
    # Load data
    print("Loading data...")
    image_paths, mask_paths = load_data_paths(args.images, args.masks)
    print(f"Found {len(image_paths)} test samples")
    
    dataset_loader = SegmentationDataset(config, augment=False, include_token_labels=False)
    test_dataset = dataset_loader.create_dataset(
        image_paths, mask_paths,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create metrics
    metrics = {
        'iou': IoUMetric(args.num_classes),
        'dice': DiceMetric(),
        'boundary_f1': BoundaryF1Metric(),
    }
    
    # Per-class IoU
    class_ious = {i: [] for i in range(args.num_classes)}
    
    # Evaluate
    print("Evaluating...")
    sample_idx = 0
    
    for batch in tqdm(test_dataset):
        images, targets = batch
        masks = targets['segmentation']
        
        # Forward pass
        if args.efficient:
            outputs = model.predict_efficient(images)
        else:
            outputs = model(images, training=False)
        
        predictions = tf.argmax(outputs, axis=-1)
        
        # Update metrics
        for metric in metrics.values():
            metric.update_state(masks, predictions)
        
        # Save visualizations
        if args.save_predictions:
            for i in range(images.shape[0]):
                img = images[i].numpy()
                gt = masks[i].numpy()
                pred = predictions[i].numpy()
                
                vis = visualize_predictions(img, gt, pred)
                
                vis_path = output_dir / f"pred_{sample_idx:04d}.png"
                tf.io.write_file(
                    str(vis_path),
                    tf.image.encode_png(vis)
                )
                sample_idx += 1
    
    # Compute final metrics
    results = {name: float(metric.result().numpy()) for name, metric in metrics.items()}
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    print("=" * 50)
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    if args.save_predictions:
        print(f"Predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
