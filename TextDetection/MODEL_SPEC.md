# PatchFormer: Lightweight Semantic Segmentation Architecture

## Executive Summary

PatchFormer is a lightweight, high-resolution semantic segmentation architecture designed for efficient inference on CPU and TensorFlow Lite deployment. The architecture combines patch-based encoding with convolutional feature extraction, utilizing a dual-path training strategy that ensures both pixel-perfect reconstruction capabilities and structured embedding spaces through contrastive learning.

---

## 1. Architecture Overview

### 1.1 Input/Output Specification

| Parameter | Value |
|-----------|-------|
| Input Resolution | 1024×1024×3 |
| Patch Size | 8×8 |
| Number of Patches | 16,384 (128×128 grid) |
| Latent Dimension | 512 |
| Output Resolution | 1024×1024×num_classes |

### 1.2 Core Design Principles

1. **Dual-Path Training**: Separates reconstruction/representation learning from task-specific segmentation
2. **Progressive Context Aggregation**: Multi-stage transformer processing with intermediate supervision
3. **Hybrid Encoding**: Combines explicit patch flattening with learned convolutional features
4. **Token Classification Bridge**: Provides geometric priors (in-object, boundary, background) to guide segmentation
5. **Lightweight Design**: Optimized for CPU inference with TFLite compatibility

---

## 2. Detailed Architecture

### 2.1 Stage 1: Dual-Stream Patch Encoding

#### 2.1.1 Explicit Patch Stream

```
Input: (B, 1024, 1024, 3)
    ↓
Extract Patches: 8×8 patches
    ↓
Flatten: (B, 16384, 192)  # 8×8×3 = 192
    ↓
Reshape: (B, 128, 128, 192)
```

**Rationale**: Direct pixel access preserves fine-grained detail essential for precise boundary detection.

#### 2.1.2 Convolutional Stream

```
Input: (B, 1024, 1024, 3)
    ↓
Conv Block 1: (B, 512, 512, 64)   # stride 2
    ↓
Conv Block 2: (B, 256, 256, 128)  # stride 2
    ↓
Conv Block 3: (B, 128, 128, 256)  # stride 2
    ↓
Conv Block 4: (B, 128, 128, 320)  # stride 1
```

**Each Conv Block**:
- DepthwiseSeparableConv2D (efficient for TFLite)
- BatchNormalization
- GELU activation
- Optional: Squeeze-Excitation attention

**Rationale**: Provides multi-scale learned features that complement explicit patch information.

#### 2.1.3 Feature Fusion

```
Patch Features: (B, 128, 128, 192)
Conv Features:  (B, 128, 128, 320)
    ↓
Concatenate: (B, 128, 128, 512)
    ↓
Add 2D Positional Encoding
    ↓
Reshape: (B, 16384, 512)
```

### 2.2 Learned 2D Positional Encoding

```python
class Learned2DPositionalEncoding:
    """
    Learnable positional embeddings that encode 2D spatial relationships.
    
    Parameters:
        height: 128 (grid height)
        width: 128 (grid width)
        embed_dim: 512
    
    Implementation:
        - Separate row and column embeddings: (128, 256) each
        - Broadcast and concatenate for full 2D encoding
        - Additive application to fused features
    """
```

**Design Choice**: Learned embeddings allow the model to discover optimal spatial relationships during training, outperforming fixed sinusoidal encodings for structured grid layouts.

### 2.3 Initial Transformer Block (Pre-Split)

```
Input: (B, 16384, 512)
    ↓
Multi-Head Self-Attention (8 heads)
    ↓
Layer Normalization
    ↓
Feed-Forward Network (512 → 1024 → 512)
    ↓
Layer Normalization
    ↓
Output: (B, 16384, 512)
```

**Attention Configuration**:
- Heads: 8
- Key/Value dimension: 64 per head
- Use efficient windowed attention (16×16 windows) for TFLite compatibility

---

## 3. Dual-Path Split

After the initial transformer block, the architecture splits into two paths:

### 3.1 Path A: Reconstruction + Contrastive Learning (Training Only)

This path is **only active during training** and provides two critical learning signals:

#### 3.1.1 Masked Autoencoder Reconstruction

```
Input: (B, 16384, 512) [with masking applied]
    ↓
Mask Application: Random 50-75% token masking
    ↓
Lightweight Decoder:
    - Linear: 512 → 384
    - Reshape: (B, 128, 128, 384)
    - TransposedConv: (B, 256, 256, 192)
    - TransposedConv: (B, 512, 512, 96)
    - TransposedConv: (B, 1024, 1024, 3)
    ↓
Reconstruction Output: (B, 1024, 1024, 3)
```

**Reconstruction Loss** (L_recon):
```
L_recon = MSE(masked_patches_pred, masked_patches_true) + 
          λ_perceptual * PerceptualLoss(pred, true)
```

#### 3.1.2 Contrastive Cosine Embedding Alignment

**Purpose**: Add structure to the latent space without competing with reconstruction quality.

```
Embedding: (B, 16384, 512)
    ↓
Projection Head: Linear(512 → 128)
    ↓
L2 Normalize
    ↓
Contrastive Loss Computation
```

**Contrastive Strategy**:
- **Positive pairs**: Patches from the same semantic region (determined by ground truth masks)
- **Negative pairs**: Patches from different semantic regions
- **Key insight**: Uses ground truth segmentation masks to define positives/negatives, aligning embedding similarity with semantic similarity

**Contrastive Loss** (L_contrast):
```
L_contrast = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

where:
- sim(a, b) = cosine_similarity(a, b)
- τ = temperature (0.07)
- (i, j) are positive pairs
- k iterates over all samples including negatives
```

**Critical Design**: The contrastive loss operates on a **separate projection head**, leaving the main 512-dim embeddings unaffected. This ensures:
1. Reconstruction path receives unmodified gradients
2. Contrastive learning adds structure without degrading pixel-level accuracy
3. The projection head acts as an information bottleneck

### 3.2 Path B: Progressive Segmentation Encoding

This path continues from the initial transformer output:

#### 3.2.1 Transformer Stage 2 + Context Generation

```
Input: (B, 16384, 512)
    ↓
Lightweight Transformer Block ×2
    - Reduced FFN: 512 → 768 → 512
    - Windowed attention (8×8 windows)
    ↓
Context Head: Dense(512 → 32)
    ↓
Context Output: (B, 16384, 32)
    ↓
Concatenate: (B, 16384, 512) + (B, 16384, 32) = (B, 16384, 544)
```

**Context Generation Purpose**: Produces low-dimensional context features that encode:
- Local texture patterns
- Relative positioning within objects
- Boundary proximity indicators

#### 3.2.2 Transformer Stage 3 + Token Classification

```
Input: (B, 16384, 544)
    ↓
Lightweight Transformer Block ×2
    ↓
Token Classification Head: Dense(544 → 3)
    ↓
Token Classes:
    - Class 0: Inside object (confident interior)
    - Class 1: Object boundary (adjacent to object edge)
    - Class 2: Background (not in or near object)
    ↓
Concatenate: transformer_out + token_logits = (B, 16384, 547)
```

**Token Classification Loss** (L_token):
```
L_token = CrossEntropy(token_pred, token_true) * sample_weights

where sample_weights upweight boundary class (typically 3-5×)
```

**Boundary Generation from Ground Truth**:
```python
def generate_token_labels(segmentation_mask):
    """
    Args:
        segmentation_mask: (H, W) with class indices
    Returns:
        token_labels: (num_patches,) with values {0, 1, 2}
    """
    # Resize to patch grid
    patch_mask = resize(segmentation_mask, (128, 128), method='nearest')
    
    # Detect boundaries using morphological operations
    dilated = binary_dilation(patch_mask > 0, iterations=1)
    eroded = binary_erosion(patch_mask > 0, iterations=1)
    boundary = dilated & ~eroded
    
    token_labels = np.where(patch_mask > 0,
                           np.where(boundary, 1, 0),  # boundary or interior
                           2)  # background
    return token_labels.flatten()
```

#### 3.2.3 Final Transformer Stage + Class Prediction

```
Input: (B, 16384, 547)
    ↓
Transformer Block ×1 (full attention, not windowed)
    ↓
Classification Head: Dense(547 → num_classes)
    ↓
Class Logits: (B, 16384, num_classes)
    ↓
Concatenate: transformer_out + class_logits = (B, 16384, 547 + num_classes)
```

---

## 4. Lightweight Decoder

### 4.1 Architecture

```
Input: (B, 16384, 547 + num_classes)
    ↓
Linear Projection: → (B, 16384, 256)
    ↓
Reshape: (B, 128, 128, 256)
    ↓
Decoder Block 1:
    - DepthwiseSeparableConv2D (256 filters, 3×3)
    - BatchNorm + GELU
    - Bilinear Upsample 2×
    ↓
Output: (B, 256, 256, 256)
    ↓
Decoder Block 2:
    - DepthwiseSeparableConv2D (128 filters, 3×3)
    - BatchNorm + GELU
    - Bilinear Upsample 2×
    ↓
Output: (B, 512, 512, 128)
    ↓
Decoder Block 3:
    - DepthwiseSeparableConv2D (64 filters, 3×3)
    - BatchNorm + GELU
    - Bilinear Upsample 2×
    ↓
Output: (B, 1024, 1024, 64)
    ↓
Final Conv: Conv2D(num_classes, 1×1)
    ↓
Output: (B, 1024, 1024, num_classes)
```

### 4.2 Skip Connections (Optional Enhancement)

For maximum segmentation quality, add skip connections from the convolutional encoder:

```
Conv Block 2 output (256×256×128) → Decoder Block 2 (concat)
Conv Block 1 output (512×512×64)  → Decoder Block 3 (concat)
```

---

## 5. Loss Functions

### 5.1 Complete Training Loss

```
L_total = λ_seg * L_segmentation + 
          λ_token * L_token + 
          λ_recon * L_reconstruction + 
          λ_contrast * L_contrastive
```

**Recommended Weights**:
| Loss | Weight | Schedule |
|------|--------|----------|
| L_segmentation | 1.0 | Constant |
| L_token | 0.5 | Constant |
| L_reconstruction | 0.3 | Decay to 0.1 after epoch 50 |
| L_contrastive | 0.1 | Constant |

### 5.2 Segmentation Loss

```python
def segmentation_loss(y_true, y_pred, num_classes):
    """
    Combined Dice + Focal loss for robust segmentation.
    """
    # Focal Loss (handles class imbalance)
    focal = tfa.losses.SigmoidFocalCrossEntropy(
        alpha=0.25, gamma=2.0
    )(y_true, y_pred)
    
    # Dice Loss (optimizes overlap)
    dice = dice_loss(y_true, y_pred)
    
    return 0.5 * focal + 0.5 * dice
```

### 5.3 Reconstruction Loss

```python
def reconstruction_loss(y_true, y_pred, mask):
    """
    MSE loss only on masked regions + perceptual consistency.
    """
    # Pixel-level MSE on masked patches only
    mse = tf.reduce_mean(tf.square(y_true - y_pred) * mask)
    
    # Optional: Perceptual loss using frozen VGG features
    # (Consider removing for TFLite deployment)
    
    return mse
```

### 5.4 Contrastive Loss

```python
def contrastive_cosine_loss(embeddings, labels, temperature=0.07):
    """
    InfoNCE-style contrastive loss using segmentation labels.
    
    Args:
        embeddings: (B, N, D) projected embeddings
        labels: (B, N) patch-level class labels
        temperature: softmax temperature
    """
    # Normalize embeddings
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
    
    # Compute similarity matrix
    similarity = tf.matmul(embeddings, embeddings, transpose_b=True)
    similarity = similarity / temperature
    
    # Create positive mask (same class = positive)
    labels_expand = tf.expand_dims(labels, -1)
    positive_mask = tf.equal(labels_expand, tf.transpose(labels_expand, [0, 2, 1]))
    positive_mask = tf.cast(positive_mask, tf.float32)
    
    # Remove self-similarity
    identity_mask = tf.eye(tf.shape(embeddings)[1], batch_shape=[tf.shape(embeddings)[0]])
    positive_mask = positive_mask * (1 - identity_mask)
    
    # InfoNCE loss
    exp_sim = tf.exp(similarity) * (1 - identity_mask)
    log_prob = similarity - tf.math.log(tf.reduce_sum(exp_sim, axis=-1, keepdims=True))
    
    # Average over positives
    num_positives = tf.reduce_sum(positive_mask, axis=-1)
    loss = -tf.reduce_sum(log_prob * positive_mask, axis=-1) / (num_positives + 1e-6)
    
    return tf.reduce_mean(loss)
```

---

## 6. Training Strategy

### 6.1 Two-Phase Training

**Phase 1: Representation Learning (Epochs 1-30)**
- Focus on reconstruction and contrastive losses
- Lower weight on segmentation loss (0.3)
- Higher masking ratio (75%)
- Goal: Learn robust, structured patch representations

**Phase 2: Task-Specific Fine-tuning (Epochs 31-100+)**
- Increase segmentation loss weight (1.0)
- Reduce reconstruction loss weight (0.1)
- Lower masking ratio (50%)
- Goal: Optimize for segmentation accuracy

### 6.2 Data Augmentation

```python
augmentation_pipeline = {
    'geometric': [
        RandomRotation(degrees=15),
        RandomScale(scale_range=(0.8, 1.2)),
        RandomHorizontalFlip(p=0.5),
        RandomCrop(size=1024, pad_if_needed=True),
    ],
    'photometric': [
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        RandomGaussianBlur(kernel_size=5, p=0.3),
        RandomNoise(std=0.02, p=0.2),
    ],
    'specialized': [
        # For text detection specifically
        RandomTextDistortion(p=0.2),  # Perspective transforms
        RandomBackgroundSwap(p=0.1),  # Composite text on new backgrounds
    ]
}
```

### 6.3 Learning Rate Schedule

```python
schedule = CosineDecayWithWarmup(
    initial_learning_rate=1e-4,
    warmup_steps=1000,
    decay_steps=100000,
    alpha=0.01  # Final LR = 1e-6
)
```

---

## 7. TensorFlow Lite Optimization

### 7.1 Architecture Constraints for TFLite

1. **Use TFLite-compatible operations**:
   - Avoid custom ops where possible
   - Use DepthwiseSeparableConv instead of standard Conv where efficient
   - Replace LayerNorm with BatchNorm in conv layers

2. **Attention Optimization**:
   - Use windowed attention (not global) in all but final layer
   - Consider replacing attention with efficient alternatives (e.g., Linear Attention)

3. **Quantization-Aware Training**:
   ```python
   import tensorflow_model_optimization as tfmot
   
   quantize_model = tfmot.quantization.keras.quantize_model
   q_aware_model = quantize_model(model)
   ```

### 7.2 Model Variants

| Variant | Params | FLOPs | Target |
|---------|--------|-------|--------|
| PatchFormer-Tiny | ~5M | ~10G | Mobile CPU |
| PatchFormer-Small | ~15M | ~30G | Desktop CPU |
| PatchFormer-Base | ~30M | ~60G | GPU Inference |

### 7.3 Inference Mode

During inference, Path A (reconstruction + contrastive) is completely disabled:

```python
class PatchFormer:
    def call(self, inputs, training=False):
        # ... encoding ...
        
        if training:
            reconstruction_output = self.reconstruction_path(encoded)
            contrastive_embeddings = self.contrastive_projection(encoded)
        
        # Path B always runs
        segmentation_output = self.segmentation_path(encoded)
        
        if training:
            return {
                'segmentation': segmentation_output,
                'reconstruction': reconstruction_output,
                'contrastive': contrastive_embeddings,
                'token_classification': token_logits,
            }
        return segmentation_output
```

---

## 8. Intermediate Outputs & Supervision

### 8.1 Multi-Scale Supervision

For improved boundary detection, add auxiliary losses at multiple scales:

```
Decoder Block 1 output (256×256) → Aux Loss 1 (4× downsampled GT)
Decoder Block 2 output (512×512) → Aux Loss 2 (2× downsampled GT)
Final output (1024×1024) → Main Loss
```

### 8.2 Token Classification Interpretation

The 3-class token classification provides geometric priors:

| Class | Meaning | Training Signal |
|-------|---------|-----------------|
| 0 | Interior | High confidence for class prediction |
| 1 | Boundary | Attention to fine details |
| 2 | Background | Suppress false positives |

---

## 9. Parameter Counts

### 9.1 Detailed Breakdown (PatchFormer-Small)

| Component | Parameters |
|-----------|------------|
| Conv Encoder | 2.1M |
| Positional Encoding | 65K |
| Initial Transformer | 2.1M |
| Reconstruction Decoder | 1.5M |
| Contrastive Projection | 66K |
| Stage 2 Transformers (×2) | 2.6M |
| Context Head | 17K |
| Stage 3 Transformers (×2) | 2.8M |
| Token Classification Head | 1.6K |
| Final Transformer | 1.5M |
| Classification Head | Varies |
| Segmentation Decoder | 1.8M |
| **Total** | **~15M** |

---

## 10. Implementation Checklist

### 10.1 Core Components

- [ ] `layers.py` - Custom layers (DepthwiseSeparable, Learned2DPosEnc, etc.)
- [ ] `encoder.py` - Dual-stream encoder with fusion
- [ ] `transformer.py` - Efficient transformer blocks
- [ ] `decoder.py` - Lightweight upsampling decoder
- [ ] `losses.py` - All loss functions
- [ ] `model.py` - Main PatchFormer class

### 10.2 Training Infrastructure

- [ ] `config.py` - Hyperparameters and model variants
- [ ] `dataset.py` - Data pipeline with augmentations
- [ ] `train.py` - Training loop with multi-loss handling
- [ ] `callbacks.py` - Custom callbacks for monitoring

### 10.3 Deployment

- [ ] `export.py` - TFLite conversion utilities
- [ ] `inference.py` - Optimized inference pipeline

---

## 11. Expected Performance

### 11.1 Training Metrics to Monitor

| Metric | Phase 1 Target | Phase 2 Target |
|--------|----------------|----------------|
| Reconstruction MSE | < 0.01 | < 0.02 |
| Contrastive Loss | < 2.0 | < 1.5 |
| Token Accuracy | > 85% | > 92% |
| Segmentation mIoU | > 60% | > 85% |
| Boundary F1 | > 50% | > 80% |

### 11.2 Inference Performance Targets

| Platform | PatchFormer-Tiny | PatchFormer-Small |
|----------|------------------|-------------------|
| Desktop CPU (i7) | 150ms | 400ms |
| Mobile CPU (SD865) | 500ms | 1200ms |
| TFLite (INT8) | 100ms | 300ms |

---

## 12. Key Design Decisions & Rationale

### Q: Why 8×8 patches instead of larger?
**A**: Smaller patches preserve fine detail essential for text boundaries. 8×8 provides 192-dim features (manageable) while maintaining high spatial resolution (128×128 grid).

### Q: Why dual-stream encoding (patches + conv)?
**A**: Patch flattening provides explicit pixel access; convolutions provide learned multi-scale features. The combination outperforms either alone.

### Q: Why separate reconstruction and segmentation paths?
**A**: Prevents reconstruction objective from dominating segmentation gradients. Allows reconstruction to be disabled at inference for efficiency.

### Q: Why add token classification?
**A**: Provides explicit geometric supervision. The network learns to distinguish object interiors, boundaries, and background, improving boundary precision.

### Q: Why contrastive learning if reconstruction already works?
**A**: Reconstruction ensures pixel accuracy; contrastive learning adds semantic structure to embeddings. Patches of the same class should cluster, enabling better generalization.

---

## 13. Architectural Refinements (v1.1)

The following refinements address critical design issues identified during review:

### 13.1 Fix: Masking Placement
**Problem**: Masking was applied after the split, so only reconstruction path saw masked input.
**Solution**: Masking now applied BEFORE split. Both reconstruction and segmentation paths see masked input during training, properly regularizing the shared encoder.

### 13.2 Fix: Global Attention on 16,384 Tokens
**Problem**: Final transformer used global attention on 16,384 tokens (268M operations).
**Solution**: Replaced with Swin-style windowed attention throughout. Cross-window communication achieved via alternating regular/shifted windows.

### 13.3 Fix: Contrastive Loss Class Imbalance
**Problem**: Using class labels for positive pairs caused ~90% background to collapse.
**Solution**: Contrastive loss now uses:
- Spatial proximity for positive pairs (nearby patches)
- Balanced sampling (50% foreground, 50% background)
- Same-class constraint within proximity radius

### 13.4 Fix: Dimension Explosion via Concatenation
**Problem**: Concatenating small auxiliary features (3-32d) to main embeddings (512d) caused gradient scale issues.
**Solution**: Replaced concatenation with gating mechanisms:
```
gate = sigmoid(W_g * [main, aux])
output = main + gate * (W_v * aux)
```
Dimensions now stay at embed_dim (512) throughout.

### 13.5 Fix: No Cross-Window Communication
**Problem**: 8×8 windows meant no communication across word-spanning regions.
**Solution**: Added Swin-style shifted window attention. Each SwinTransformerBlock contains:
1. W-MSA: Regular windowed attention
2. SW-MSA: Shifted windowed attention (shift = window_size // 2)

---

## 14. PatchFormer v2.0: Unified Refinement Pipeline

v2.0 implements a **unified refinement pipeline** where all components feed into
a SINGLE final segmentation output. This is the key architectural change.

### 14.1 Design Philosophy: Everything Feeds Into Final Output

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER                                  │
│  Multi-Scale Token Aggregation → Boundary-Aware Attention        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DECODER + UPSAMPLING                          │
│        128×128 → 256×256 → 512×512 → 1024×1024                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              CASCADED REFINEMENT MODULE                          │
│                                                                  │
│  Stage 1: Initial Segmentation (intermediate supervision)        │
│                              ↓                                   │
│  Stage 2: Compute Affinity → Affinity-Guided Feature Propagation │
│           (features flow through, affinity supervised)           │
│                              ↓                                   │
│  Stage 3: Compute Vertices → Vertex-Aware Boundary Sharpening    │
│           (features refined, vertices supervised)                │
│                              ↓                                   │
│  Stage 4: DB Head → Differentiable Binarization                  │
│           (final sharpening, DB supervised)                      │
│                              ↓                                   │
│                    FINAL OUTPUT                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 14.2 Encoder Enhancements

**Multi-Scale Token Aggregation (MSTA)**
- Creates 3-scale feature pyramid (128×128, 64×64, 32×32) via token pooling
- Fine-resolution tokens cross-attend to coarse scales for global context
- Result: Better handling of text at varying sizes

**Boundary-Aware Cross-Attention (BACA)**
- Uses token classification to bias attention toward boundary patches
- Boundary patches attend more strongly to other boundary patches
- Result: Sharper boundary features in encoder output

### 14.3 Cascaded Refinement Components

**Affinity-Guided Refiner**
- Computes 8-directional affinity from decoder features
- Uses affinity to propagate features between related regions
- High affinity → features shared → coherent text regions
- Low affinity → features isolated → sharp boundaries

**Vertex-Aware Boundary Refiner**
- Predicts vertex probability and offset at each location
- Applies extra sharpening near predicted vertices
- Uses offset direction to guide boundary orientation
- Result: Accurate polygon corners for curved text

**Differentiable Binarization (DB) Head**
- Final stage producing the output
- Predicts probability map, threshold map
- Binary = sigmoid(k * (prob - thresh)) with k=50
- Result: Crisp binary output without post-processing

### 14.4 Key Insight: How Each Component Improves Final Output

| Component | Located In | How It Helps Final Segmentation |
|-----------|------------|--------------------------------|
| MSTA | Encoder | Tokens have multi-scale context → better at all text sizes |
| BACA | Encoder | Encoder emphasizes boundaries → sharper edges flow downstream |
| Affinity Refiner | Decoder | Features propagate within text → coherent regions, no holes |
| Vertex Refiner | Decoder | Sharpening at corners → accurate polygon boundaries |
| DB Head | Decoder | Final sharpening → crisp binary without post-processing |

**All paths converge to single output** - no auxiliary outputs at inference.

---

## 15. PatchFormer v2.1: SOTA Improvements

v2.1 adds three key improvements from SegFormer and DBNet++ to close remaining gaps.

### 15.1 Overlapping Patch Embeddings (from SegFormer)

**Problem**: Non-overlapping 8×8 patches create visible boundary artifacts.

**Solution**: Use strided convolution with stride < kernel_size:
```python
# Before: Non-overlapping (stride = patch_size)
patches = extract_patches(image, size=8, stride=8)

# After: 50% overlap (stride = patch_size // 2)
self.patch_embed = OverlappingPatchEmbed(
    patch_size=8,
    stride=4,  # 50% overlap
    embed_dim=192
)
```

**Benefits**:
- Eliminates boundary artifacts at patch edges
- Smoother feature transitions
- Better local continuity preservation

### 15.2 Deformable Convolutions (from DBNet++)

**Problem**: Standard convolutions have fixed receptive fields, struggle with curved/rotated text.

**Solution**: Learn spatial offsets for each sampling position:
```python
class DeformableConv2D:
    # Predicts 2*k*k offsets (x,y per position)
    # Plus optional modulation scalars
    # Samples at offset positions with bilinear interpolation
```

**Integration**: Applied in ConvolutionalEncoderV2 blocks 3 and 4:
```
Block 1: Standard Conv (512×512)
Block 2: Standard Conv (256×256)
Block 3: Deformable Conv (128×128) ← adapts to text shape
Block 4: Deformable Conv (128×128) ← refines adaptation
```

**Benefits**:
- Adapts receptive field to curved text
- Handles rotated text better
- Focuses on text-relevant regions

### 15.3 Adaptive Scale Fusion (from DBNet++)

**Problem**: Static multi-scale aggregation uses fixed weights for all locations.

**Solution**: Learn to weight features from different scales per location:
```python
class AdaptiveScaleFusion:
    # Creates 3-scale pyramid (128×128, 64×64, 32×32)
    # Learns spatial attention for scale selection
    # Small text → high-res features
    # Large text → low-res features
```

**Replaces**: MultiScaleTokenAggregator (static attention across scales)

**Benefits**:
- Per-location scale selection
- Better handling of varying text sizes in same image
- Learned end-to-end

### 15.4 MixFFN (from SegFormer)

**Addition**: 3×3 depthwise conv in FFN blocks provides implicit positional encoding:
```python
class MixFFN:
    def call(self, x, grid_size):
        x = self.fc1(x)  # Expand
        x = reshape_to_spatial(x, grid_size)
        x = self.dwconv(x)  # 3×3 depthwise (adds position info)
        x = reshape_to_sequence(x)
        x = self.fc2(x)  # Contract
        return x
```

**Benefits**:
- Position-awareness through convolution
- Can reduce reliance on explicit positional encoding
- Better local feature mixing

---

## 16. v2.1 Architecture Summary

### Complete v2.1 Pipeline
```
Input Image (1024×1024×3)
         ↓
┌────────────────────────────────────────────────────┐
│ DUAL-STREAM ENCODER V2                              │
│                                                     │
│ Stream 1: Overlapping Patch Embed (50% overlap)     │
│           → Eliminates boundary artifacts           │
│                                                     │
│ Stream 2: ConvEncoder V2 (Deformable Conv)          │
│           → Adapts to curved/rotated text           │
│                                                     │
│ Fusion → Positional Encoding → Initial Transformer  │
└────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│ PROGRESSIVE ENCODER                                 │
│                                                     │
│ Adaptive Scale Fusion                               │
│ → Per-location scale weighting                      │
│         ↓                                           │
│ Stage 2: Context + Swin Attention                   │
│         ↓                                           │
│ Stage 3: Token Classification + BACA                │
│         ↓                                           │
│ Final: Segmentation Head                            │
└────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│ CASCADED REFINEMENT (v2.0)                          │
│                                                     │
│ Affinity-Guided → Vertex-Aware → DB Head            │
│ → Single refined output                             │
└────────────────────────────────────────────────────┘
         ↓
    Final Output (1024×1024×1)
```

### Parameter Count (v2.1 small variant)
| Component | v2.0 | v2.1 | Change |
|-----------|------|------|--------|
| Dual-Stream Encoder | ~12M | ~13M | +1M (deformable) |
| Progressive Encoder | ~8M | ~8M | ~ (ASF similar) |
| Cascaded Refinement | ~3M | ~3M | - |
| Decoder | ~3.5M | ~3.5M | - |
| **Total** | **~26M** | **~27M** | **+4%** |

---

## 17. v2.0/v2.1 Training Configuration

### Loss Weights
```python
total_loss = (
    1.0 * segmentation_loss +    # Main Dice+Focal
    1.0 * db_loss +              # Differentiable binarization
    0.5 * affinity_loss +        # Patch affinity
    0.3 * token_loss +           # Interior/boundary/background
    0.2 * reconstruction_loss +  # MAE (decayed after phase 1)
    0.1 * contrastive_loss +     # Spatial proximity contrastive
    0.3 * polygon_loss           # Optional, for curved text
)
```

### Parameter Count (v2.0 small variant)
| Component | Parameters |
|-----------|------------|
| Dual-Stream Encoder | ~12M |
| Multi-Scale Aggregator | ~1.5M |
| Transformer Stages | ~7M |
| Boundary Attention | ~1M |
| Affinity Head | ~0.5M |
| DB Head | ~0.5M |
| Polygon Head | ~0.3M |
| Decoder | ~3.5M |
| **Total** | **~26M** |

### Expected Performance (v2.0)
| Benchmark | v1.x | v2.0 | Target |
|-----------|------|------|--------|
| ICDAR15 F1 | 80-85 | 87-89 | ≥87 (DBNet: 84.3) |
| TotalText F1 | 75-80 | 85-87 | ≥85 (TextFuseNet: 85.3) |
| CTW1500 F1 | 72-78 | 84-86 | ≥84 (CRAFT: 81.1) |
| MSRA-TD500 F1 | 80-85 | 86-88 | ≥86 (TextFuseNet: 87.4) |

---

## Appendix A: File Structure

```
TextDetectionModel/
├── MODEL_SPEC.md           # This document
├── requirements.txt        # Dependencies
├── README.md              # Quick start guide
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters
│   ├── layers.py          # Custom layers (GatingMechanism, etc.)
│   ├── heads.py           # v2.0: Advanced prediction heads
│   ├── encoder.py         # Encoder module
│   ├── transformer.py     # Transformer blocks (Swin-style)
│   ├── decoder.py         # Decoder module (PatchFormerDecoderV2)
│   ├── losses.py          # Loss functions (PatchFormerV2Loss)
│   ├── model.py           # Main model (PatchFormer, PatchFormerV2)
│   ├── dataset.py         # Data pipeline
│   └── utils.py           # Utilities
├── train.py               # Training script
├── evaluate.py            # Evaluation script
└── export.py              # TFLite export
```

---

## Appendix B: Quick Reference - Tensor Shapes

```
Input:                    (B, 1024, 1024, 3)
Patches (flat):           (B, 16384, 192)
Patches (spatial):        (B, 128, 128, 192)
Conv features:            (B, 128, 128, 320)
Fused:                    (B, 128, 128, 512)
Sequence:                 (B, 16384, 512)

# With gating (v1.1), dimensions stay at embed_dim:
After Stage 2 (gated):    (B, 16384, 512)  # Was 544 with concat
After Stage 3 (gated):    (B, 16384, 512)  # Was 547 with concat
After Final (gated):      (B, 16384, 512)  # Was 547+num_classes
Decoder input:            (B, 128, 128, 256)
Output:                   (B, 1024, 1024, num_classes)

# Auxiliary outputs (available during training):
Context head:             (B, 16384, 32)
Token logits:             (B, 16384, 3)
Class logits:             (B, 16384, num_classes)
Reconstruction:           (B, 1024, 1024, 3)
Contrastive embeddings:   (B, 16384, 128)
```
