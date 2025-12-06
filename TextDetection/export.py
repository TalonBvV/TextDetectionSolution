"""
Export PatchFormer model to TensorFlow Lite and other formats.
"""

import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path

from src.config import get_config
from src.model import PatchFormer, create_model, TFLiteExporter


def parse_args():
    parser = argparse.ArgumentParser(description="Export PatchFormer model")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--variant", type=str, default="small",
                       choices=["tiny", "small", "base"])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=1024)
    
    # Export format
    parser.add_argument("--format", type=str, default="tflite",
                       choices=["tflite", "tflite-efficient", "saved_model", "onnx"])
    parser.add_argument("--output", type=str, required=True,
                       help="Output path")
    
    # Quantization
    parser.add_argument("--quantize", action="store_true",
                       help="Apply INT8 quantization")
    parser.add_argument("--calibration-data", type=str, default=None,
                       help="Path to calibration images for quantization")
    
    return parser.parse_args()


def create_representative_dataset(image_dir: str, input_size: int, num_samples: int = 100):
    """Create representative dataset generator for quantization calibration."""
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    image_paths = image_paths[:num_samples]
    
    def representative_dataset():
        for path in image_paths:
            image = tf.io.read_file(str(path))
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, [input_size, input_size])
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.expand_dims(image, 0)
            yield [image]
    
    return representative_dataset


def export_saved_model(model: PatchFormer, output_path: str, input_size: int):
    """Export as SavedModel format."""
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, input_size, input_size, 3], dtype=tf.float32)
    ])
    def serving_fn(images):
        return model(images, training=False)
    
    tf.saved_model.save(
        model,
        output_path,
        signatures={'serving_default': serving_fn}
    )
    print(f"SavedModel exported to {output_path}")


def export_onnx(model: PatchFormer, output_path: str, input_size: int):
    """Export to ONNX format."""
    try:
        import tf2onnx
    except ImportError:
        print("Please install tf2onnx: pip install tf2onnx")
        return
    
    # First export to SavedModel
    temp_dir = Path(output_path).parent / "temp_saved_model"
    export_saved_model(model, str(temp_dir), input_size)
    
    # Convert to ONNX
    import subprocess
    subprocess.run([
        "python", "-m", "tf2onnx.convert",
        "--saved-model", str(temp_dir),
        "--output", output_path,
        "--opset", "13"
    ])
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"ONNX model exported to {output_path}")


def main():
    args = parse_args()
    
    # Create model
    print("Loading model...")
    model = create_model(
        variant=args.variant,
        num_classes=args.num_classes,
        input_size=(args.input_size, args.input_size),
    )
    
    # Load weights
    model.load_weights(args.checkpoint)
    print("Weights loaded")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export based on format
    if args.format == "tflite":
        # Prepare representative dataset for quantization
        rep_dataset = None
        if args.quantize and args.calibration_data:
            rep_dataset = create_representative_dataset(
                args.calibration_data, args.input_size
            )
        
        TFLiteExporter.export(
            model,
            str(output_path),
            quantize=args.quantize,
            representative_dataset=rep_dataset,
        )
    
    elif args.format == "tflite-efficient":
        TFLiteExporter.export_efficient(model, str(output_path))
    
    elif args.format == "saved_model":
        export_saved_model(model, str(output_path), args.input_size)
    
    elif args.format == "onnx":
        export_onnx(model, str(output_path), args.input_size)
    
    print("Export complete!")


if __name__ == "__main__":
    main()
