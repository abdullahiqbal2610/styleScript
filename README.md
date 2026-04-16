# StyleScript

A Python implementation of the **StyleScript** research paper — a pipeline for generating style-controlled handwritten text images from engineering documents using deep learning.

> 📄 See [`styleScript.pdf`](./styleScript.pdf) for the full research paper.

---

## Overview

StyleScript is a multi-phase pipeline that:
1. **Extracts** typographic style features (stroke thickness, slant angle) from source images.
2. **Encodes** input text at the character level with noise modulation, using a vocabulary of 60,000 tokens compatible with TrOCR.
3. **Generates** images conditioned on the extracted style using Conditional Batch Normalization (CBN) and geometric transformations (font scaling + shear).
4. **Augments** generated images with rotation, perspective distortion, Gaussian noise, and photometric changes.
5. **Validates** output quality by checking for blank images and edge sharpness.
6. **Optimises** the full pipeline using a composite loss (style loss + TrOCR cross-entropy content loss + quality loss).
7. **Recognises** text in generated images via a dedicated TrOCR inference pipeline (Section 3.2).

---

## Architecture

```
Input Image ──► Phase 1: Style Extraction   ──► Style Vector s = [τ, θ]
                                                        │
Input Text  ──► Phase 2: Text Encoder E(y) ──► Text Map M
                                                        │
                         Phase 3: Generator G(M, s)    │
                         (CBN + Font Scale + Shear) ◄──┘
                                │
                         Phase 4: Augmentation T(x̂)
                                │
                         Phase 5: Quality Validation Q
                                │
                         Phase 6: Loss Computation
                          (L_style + TrOCR Cross-Entropy + L_quality)
                                │
                          Phase 7: AdamW Optimisation
                                │
                    Section 3.2: TrOCR OCR Inference Pipeline
```

---

## Project Structure

```
styleScript/
├── main.py                    # Training entry point (Phases 6 & 7)
├── generate_dummy_data.py     # Script to create a synthetic dataset for testing
├── training_loss_curve.png    # Loss curve graph saved after each training run
├── styleScript.pdf            # Original research paper
├── data/
│   ├── annotations.csv        # Image filename ↔ text label mapping
│   └── raw/                   # Source images (real or generated)
└── src/
    ├── phase1_extraction.py      # Style vector extraction (stroke thickness + slant angle)
    ├── phase2_3_model.py         # Text encoder + style-controlled generator
    ├── phase4_5_utils.py         # Augmentation pipeline + quality validator
    └── section3_2_pipeline.py    # TrOCR OCR inference pipeline (Section 3.2)
```

---

## Phases Explained

### Phase 1 — Style Extraction (`src/phase1_extraction.py`)
Computes a 2D style vector **s = [τ, θ]** from a source word image:

| Feature | Formula | Method |
|---------|---------|--------|
| Stroke Thickness τ | τ = (1/N) Σ(Aᵢ/Pᵢ) | Contour area / perimeter via OpenCV |
| Slant Angle θ | θ = arctan2(Δy, Δx) | Hough Line Transform on Canny edges |

### Phase 2 — Text Encoder (`src/phase2_3_model.py`)
- Character embeddings → noise modulation **F(y, ε) = f_c ⊗ ε**
- Embeddings are concatenated horizontally to form a text map **M**.
- Vocabulary size is set to **60,000** to support the full TrOCR token space.

### Phase 3 — Style-Controlled Generator (`src/phase2_3_model.py`)
- **Conditional Batch Normalization**: `CBN(h, s) = γ(s) · (h − μ)/σ + β(s)`
- **Font scaling** (Eq. 8): `scale = clamp(τ / 2, 0.8, 1.2)`
- **Shear transformation** (Eq. 9): affine matrix derived from clamped slant angle θ ∈ [−30°, 30°]
- Output: 64 × 128 greyscale image tensor.

### Phase 4 — Augmentation (`src/phase4_5_utils.py`)
Composite transform **T = T_rot ∘ T_pers ∘ T_noise ∘ T_photo**:
- **T_rot**: Random rotation ±3°
- **T_pers**: Random perspective distortion (scale 0.1)
- **T_noise**: Gaussian noise η ~ N(0, σ²), σ = 0.05
- **T_photo**: Brightness and contrast jitter U(0.95, 1.05)

### Phase 5 — Quality Validation (`src/phase4_5_utils.py`)
`Q_total = Q_blank ∧ Q_bound ∧ (R ≥ R_min)`:
- **Q_blank**: Rejects images where all pixels are near-white (threshold ≥ 0.98)
- **R**: Edge sharpness via Laplacian convolution variance

### Phase 6 & 7 — Loss & Optimisation (`main.py`)
```
L_total = λ₁·L_style + λ₂·L_content + λ₃·L_quality
```
| Term | Description |
|------|-------------|
| L_style | L1 loss between generated and target style vectors |
| L_content | **TrOCR Cross-Entropy** — generated image is passed through the frozen `microsoft/trocr-small-printed` decoder with target token IDs as labels |
| L_quality | −log(Q_total) — penalises low-quality outputs |

Optimiser: **AdamW** (lr = 0.0002), λ₁ = 1.0, λ₂ = 1.0, λ₃ = 0.1.

TrOCR is loaded in **frozen** evaluation mode; only the StyleScript generator parameters are trained.

### Section 3.2 — TrOCR OCR Inference (`src/section3_2_pipeline.py`)
Runs a standalone OCR pass on any image using the `microsoft/trocr-small-printed` model:
- Opens an image, converts to RGB, and processes it with `TrOCRProcessor`.
- Calls `model.generate()` and decodes the predicted text.
- Can be used independently to verify the OCR quality of any generated image.

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision opencv-python numpy pandas transformers matplotlib
```

### 1. Generate a Dummy Dataset

```bash
python generate_dummy_data.py
```

This creates 10 synthetic greyscale word images in `data/raw/` and an `annotations.csv` mapping each image to its text label.

### 2. Run Training

```bash
python main.py
```

The script loads the TrOCR model, then runs **10 training epochs** and prints per-batch losses, per-epoch averages, a final summary table, and saves a loss curve graph:

```
--- Starting StyleScript Training (10 Epochs) ---
Loading TrOCR model and Tokenizer...

========== EPOCH 1/10 ==========
Batch 1/10 | L_style: 0.4821 | L_content: 3.1234 | L_total: 3.7890
Batch 2/10 | L_style: 0.4503 | L_content: 3.0812 | L_total: 3.7128
...
-> End of Epoch 1 | Avg L_style: 0.4631 | Avg L_content: 3.0994 | Avg L_total: 3.7444

========== EPOCH 2/10 ==========
...

==================================================
 📊 FINAL TRAINING SUMMARY (AVERAGES PER EPOCH)
==================================================
Epoch      | Style Loss   | Content Loss | Total Loss
--------------------------------------------------
Epoch 1    | 0.4631       | 3.0994       | 3.7444
...
==================================================

✅ Training Complete! A summary table has been printed and 'training_loss_curve.png' has been saved to your folder.
```

A `training_loss_curve.png` plot (Style Loss, Content Loss, and Total Loss over 10 epochs) is saved automatically in the project root.

### 3. Run TrOCR OCR Inference (Section 3.2)

```bash
python src/section3_2_pipeline.py
```

Runs `microsoft/trocr-small-printed` inference on `data/raw/dummy_0.png` and prints the recognised text:

```
--- Running Section 3.2: Practical OCR Pipeline ---
Loading TrOCR model (this may take a minute to download weights)...
Target Image: data/raw/dummy_0.png
Recognized Text: 'STRUCTI'
---------------------------------------------------
```

### 4. Test Individual Phases

Each module can be run independently:

```bash
python src/phase1_extraction.py       # Test style vector extraction
python src/phase2_3_model.py          # Test text encoder + generator
python src/phase4_5_utils.py          # Test augmentation + quality validation
python src/section3_2_pipeline.py     # Test TrOCR OCR inference
```

---

## Key Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Neural network, loss functions, AdamW optimiser |
| `torchvision` | Augmentation transforms (rotation, perspective) |
| `transformers` | TrOCR model & processor (tokenisation + cross-entropy content loss + OCR inference) |
| `opencv-python` | Style feature extraction (contours, Hough lines) |
| `numpy` | Array operations |
| `pandas` | CSV dataset loading |
| `matplotlib` | Training loss curve visualisation (`training_loss_curve.png`) |

---

## Citation

If you use this implementation, please refer to the original StyleScript paper included in this repository (`styleScript.pdf`).
