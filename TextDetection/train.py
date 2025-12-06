"""
Training script for PatchFormer semantic segmentation model.
"""

import argparse
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import json

from src.config import PatchFormerConfig, TrainingConfig, get_config
from src.model import PatchFormer, create_model
from src.losses import PatchFormerLoss, PatchFormerV2Loss, generate_token_labels
from src.dataset import (
    SegmentationDataset, create_dummy_dataset,
    create_unified_dataset, load_unified_dataset_paths,
    UnifiedTextDetectionDataset
)
from src.utils import (
    IoUMetric, DiceMetric, BoundaryF1Metric,
    CosineDecayWithWarmup, CheckpointManager,
    print_model_summary, save_config
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PatchFormer model")
    
    # Model configuration
    parser.add_argument("--variant", type=str, default="small",
                       choices=["tiny", "small", "base"],
                       help="Model variant")
    parser.add_argument("--num-classes", type=int, default=2,
                       help="Number of segmentation classes")
    parser.add_argument("--input-size", type=int, default=1024,
                       help="Input image size")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                       help="Warmup steps")
    
    # Data
    parser.add_argument("--dataset-dir", type=str, default=None,
                       help="Path to unified dataset (from DatasetGeneration pipeline)")
    parser.add_argument("--train-images", type=str, default=None,
                       help="Path to training images directory (legacy)")
    parser.add_argument("--train-masks", type=str, default=None,
                       help="Path to training masks directory (legacy)")
    parser.add_argument("--val-images", type=str, default=None,
                       help="Path to validation images directory (legacy)")
    parser.add_argument("--val-masks", type=str, default=None,
                       help="Path to validation masks directory (legacy)")
    parser.add_argument("--use-dummy-data", action="store_true",
                       help="Use dummy data for testing")
    
    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="TensorBoard log directory")
    
    # GPU
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Use mixed precision training")
    
    # Model options
    parser.add_argument("--use-v2-loss", action="store_true",
                       help="Use PatchFormerV2Loss with affinity, vertex, DB losses")
    
    return parser.parse_args()


def load_data_paths(images_dir: str, masks_dir: str):
    """Load image and mask file paths from directories."""
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_paths = []
    mask_paths = []
    
    for ext in image_extensions:
        for img_path in images_path.glob(f'*{ext}'):
            # Find corresponding mask
            mask_path = masks_path / f"{img_path.stem}.png"
            if mask_path.exists():
                image_paths.append(str(img_path))
                mask_paths.append(str(mask_path))
    
    return image_paths, mask_paths


def train_step(
    model: PatchFormer,
    loss_fn: PatchFormerLoss,
    optimizer: tf.keras.optimizers.Optimizer,
    images: tf.Tensor,
    masks: tf.Tensor,
    token_labels: tf.Tensor,
    current_epoch: int,
    instance_map: tf.Tensor = None,
):
    """Single training step."""
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(images, training=True)
        
        # Prepare ground truth
        ground_truth = {
            'segmentation': masks,
            'images': images,
            'token_labels': token_labels,
        }
        
        # Add instance_map if available (for affinity loss)
        if instance_map is not None:
            ground_truth['instance_map'] = instance_map
        
        # Compute losses
        total_loss, loss_dict = loss_fn(outputs, ground_truth, current_epoch)
    
    # Compute and apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss_dict


def validate(
    model: PatchFormer,
    val_dataset: tf.data.Dataset,
    metrics: dict,
):
    """Run validation."""
    # Reset metrics
    for metric in metrics.values():
        metric.reset_state()
    
    for batch in val_dataset:
        images, targets = batch
        masks = targets['segmentation']
        
        # Forward pass
        outputs = model(images, training=False, return_all_outputs=True)
        predictions = tf.argmax(outputs['main'], axis=-1)
        
        # Update metrics
        for metric in metrics.values():
            metric.update_state(masks, predictions)
    
    # Return results
    return {name: metric.result().numpy() for name, metric in metrics.items()}


def main():
    args = parse_args()
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Create configuration
    model_config = get_config(args.variant, args.num_classes)
    model_config.input_size = (args.input_size, args.input_size)
    
    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        variant=args.variant,
        num_classes=args.num_classes,
        input_size=(args.input_size, args.input_size),
        use_multi_scale=True,
    )
    print_model_summary(model, (args.input_size, args.input_size, 3))
    
    # Create datasets
    print("Loading data...")
    dataset_loader = SegmentationDataset(model_config, augment=True)
    
    if args.use_dummy_data:
        print("Using dummy data for testing...")
        train_dataset = create_dummy_dataset(
            model_config, 
            num_samples=16, 
            batch_size=args.batch_size
        )
        val_dataset = create_dummy_dataset(
            model_config, 
            num_samples=8, 
            batch_size=args.batch_size
        )
    elif args.dataset_dir:
        # Use unified dataset format (from DatasetGeneration pipeline)
        print(f"Loading unified dataset from {args.dataset_dir}...")
        
        train_dataset = create_unified_dataset(
            args.dataset_dir,
            model_config,
            split="train",
            batch_size=args.batch_size,
            augment=True
        )
        
        val_dataset = create_unified_dataset(
            args.dataset_dir,
            model_config,
            split="val",
            batch_size=args.batch_size,
            augment=False
        )
    else:
        # Legacy: separate image/mask directories
        if args.train_images is None or args.train_masks is None:
            raise ValueError(
                "Must provide --dataset-dir (unified format) or "
                "--train-images and --train-masks, or use --use-dummy-data"
            )
        
        train_image_paths, train_mask_paths = load_data_paths(
            args.train_images, args.train_masks
        )
        print(f"Found {len(train_image_paths)} training samples")
        
        train_dataset = dataset_loader.create_dataset(
            train_image_paths, train_mask_paths,
            batch_size=args.batch_size, shuffle=True
        )
        
        if args.val_images and args.val_masks:
            val_dataset_loader = SegmentationDataset(model_config, augment=False)
            val_image_paths, val_mask_paths = load_data_paths(
                args.val_images, args.val_masks
            )
            print(f"Found {len(val_image_paths)} validation samples")
            
            val_dataset = val_dataset_loader.create_dataset(
                val_image_paths, val_mask_paths,
                batch_size=args.batch_size, shuffle=False
            )
        else:
            val_dataset = None
    
    # Create optimizer with learning rate schedule
    steps_per_epoch = len(list(train_dataset))
    total_steps = steps_per_epoch * args.epochs
    
    lr_schedule = CosineDecayWithWarmup(
        initial_learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=total_steps,
        alpha=1e-6,
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=training_config.weight_decay,
    )
    
    # Create loss function
    if args.use_v2_loss:
        print("Using PatchFormerV2Loss (full refinement pipeline supervision)")
        loss_fn = PatchFormerV2Loss(
            model_config, training_config,
            use_db_loss=True,
            use_affinity_loss=True,
            use_vertex_loss=True
        )
    else:
        print("Using PatchFormerLoss (basic supervision)")
        loss_fn = PatchFormerLoss(model_config, training_config)
    
    # Create metrics
    metrics = {
        'iou': IoUMetric(args.num_classes),
        'dice': DiceMetric(),
        'boundary_f1': BoundaryF1Metric(),
    }
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        model, optimizer, args.checkpoint_dir
    )
    
    # Restore from checkpoint if exists
    if checkpoint_manager.restore():
        print("Restored from checkpoint")
    
    # Create TensorBoard writer
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(log_dir))
    
    # Save configuration
    save_config(model_config, str(log_dir / "model_config.json"))
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training
        epoch_losses = {}
        pbar = tqdm(train_dataset, desc="Training")
        
        for batch in pbar:
            images, targets = batch
            masks = targets['segmentation']
            token_labels = targets.get('token_labels')
            instance_map = targets.get('instance_map')  # For affinity loss
            
            if token_labels is None:
                token_labels = generate_token_labels(masks, model_config.grid_size)
            
            loss_dict = train_step(
                model, loss_fn, optimizer,
                images, masks, token_labels, epoch,
                instance_map=instance_map
            )
            
            # Accumulate losses
            for name, value in loss_dict.items():
                if name not in epoch_losses:
                    epoch_losses[name] = []
                epoch_losses[name].append(float(value))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'seg': f"{loss_dict['segmentation']:.4f}",
            })
            
            # Log to TensorBoard
            with summary_writer.as_default():
                for name, value in loss_dict.items():
                    tf.summary.scalar(f'train/{name}', value, step=global_step)
            
            global_step += 1
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        for name, values in epoch_losses.items():
            mean_value = sum(values) / len(values)
            print(f"  {name}: {mean_value:.4f}")
        
        # Validation
        if val_dataset is not None:
            print("\nValidating...")
            val_metrics = validate(model, val_dataset, metrics)
            
            print("Validation metrics:")
            for name, value in val_metrics.items():
                print(f"  {name}: {value:.4f}")
            
            # Log validation metrics
            with summary_writer.as_default():
                for name, value in val_metrics.items():
                    tf.summary.scalar(f'val/{name}', value, step=epoch)
            
            # Save checkpoint
            checkpoint_manager.save(epoch, metric=val_metrics['iou'])
        else:
            checkpoint_manager.save(epoch)
    
    print("\nTraining complete!")
    
    # Save final model
    final_model_path = Path(args.checkpoint_dir) / "final_model"
    model.save_weights(str(final_model_path))
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
