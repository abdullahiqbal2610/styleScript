# StyleScript

This repository is a **project implementation of the paper**:
**StyleScript: A Structured Data Augmentation Framework for Transformer-Based OCR in Engineering Documents**.

- Paper in repo: [`styleScript.pdf`](./styleScript.pdf)
- Latest experimental notebook: [`ai-project.ipynb`](./ai-project.ipynb)

## StyleScript Overview

The paper proposes a structured augmentation pipeline for OCR in engineering-document settings. The core flow is:

1. Extract style features from handwriting images.
2. Generate style-controlled synthetic word images.
3. Fine-tune TrOCR using generated data.
4. Evaluate impact with CER/WER.

Our repository implements this pipeline in Python modules, covering training, generation, OCR inference, and downstream evaluation.

## Repository Structure

```text
styleScript/
├── ai-project.ipynb                # Main notebook entry point
├── main.py                         # Training entry point (Phases 6 & 7)
├── config.json                     # Hyperparameters, loss weights, and file paths
├── stylescript_generator.pth       # Saved generator model weights (after training)
├── training_loss_curve.png         # Loss curve graph saved after each training run
├── styleScript.pdf                 # Original research paper
├── data/
│   ├── annotations.csv             # Image filename ↔ text label mapping
│   └── test/                       # Optional local/test image folder (no raw dataset tracked)
├── src/
│   ├── phase1_extraction.py        # Style vector extraction (stroke thickness + slant angle)
│   ├── phase2_3_model.py           # Text encoder + style-controlled generator
│   └── phase4_5_utils.py           # Augmentation pipeline + quality validator
└── testing/
    ├── phase8_9_evaluation.py      # Optional synthetic evaluation and downstream OCR fine-tuning
    └── section3_2_pipeline.py      # Optional TrOCR OCR inference pipeline (Section 3.2)
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

All hyperparameters and paths are loaded from **`config.json`** (see [Configuration](#configuration)). Optimiser: **AdamW** (lr = 0.0002), λ₁ = 1.0, λ₂ = 1.0, λ₃ = 0.1.

TrOCR is loaded in **frozen** evaluation mode; only the StyleScript generator parameters are trained. After training, the generator weights are saved to `stylescript_generator.pth`.

### Phases 8 & 9 — Evaluation & Downstream Fine-Tuning (`testing/phase8_9_evaluation.py`)
Evaluates the trained StyleScript pipeline against a standard TrOCR baseline:

1. **Baseline evaluation** — runs `microsoft/trocr-small-printed` directly on the dataset and records CER & WER.
2. **Synthetic data generation (Phase 8)** — loads the saved generator (`stylescript_generator.pth`) and produces style-perturbed synthetic images from each annotation.
3. **Downstream fine-tuning (Phase 9)** — fine-tunes TrOCR on the synthetic images with a low learning rate (5 × 10⁻⁵).
4. **StyleScript evaluation** — re-evaluates TrOCR after fine-tuning to measure improvement.

Outputs a comparison table (Table 1 from the paper):

```
 📊 TABLE 1: DOWNSTREAM OCR PERFORMANCE COMPARISON
Model                     | CER (Lower is better)  | WER
------------------------------------------------------------
Baseline TrOCR            | 0.XXXX                 | 0.XXXX
StyleScript Enhanced OCR  | 0.XXXX                 | 0.XXXX
```

### Section 3.2 — TrOCR OCR Inference (`testing/section3_2_pipeline.py`)
Runs a standalone OCR pass on any image using the `microsoft/trocr-small-printed` model:
- Opens an image, converts to RGB, and processes it with `TrOCRProcessor`.
- Calls `model.generate()` and decodes the predicted text.
- Can be used independently to verify the OCR quality of any generated image.

---

## Configuration

All training hyperparameters and file paths are stored in **`config.json`** at the project root:

```json
{
  "hyperparameters": {
    "num_epochs": 10,
    "learning_rate": 0.0002,
    "batch_size": 1,
    "vocab_size": 60000
  },
  "loss_weights": {
    "lambda_style": 1.0,
    "lambda_content": 1.0,
    "lambda_quality": 0.1
  },
  "paths": {
    "annotations_csv": "data/annotations.csv",
    "img_dir": "data/test/",
    "model_save_path": "stylescript_generator.pth"
  }
}
```

Edit this file to change epochs, learning rate, or dataset paths without touching any Python source files.

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision opencv-python numpy pandas transformers matplotlib jiwer
```

### 1. Prepare Your Dataset

Use your Kaggle dataset and keep `data/annotations.csv` mapped to your image filenames.  
If you run locally, place optional test images in `data/test/`.
If you are migrating from older repo layouts, move any previously tracked local files from `data/raw/` into `data/test/`.

### 2. Run Training

```bash
python main.py
```

## Latest `ai-project.ipynb` Findings and Results

The latest notebook compares baseline OCR, ScrabbleGAN-style augmentation, and StyleScript-style augmentation.

### Reported replication table (from notebook output)

```bash
python testing/phase8_9_evaluation.py
```

### Additional notebook run outputs

- The notebook also contains intermediate run logs from earlier evaluation cells.
- The final comparative table above shows the canonical replication metrics used in this README, taken from the notebook's consolidated summary section.

```bash
python testing/section3_2_pipeline.py
```

Runs `microsoft/trocr-small-printed` inference on an image from `data/test/` and prints the recognised text:

```
--- Running Section 3.2: Practical OCR Pipeline ---
Loading TrOCR model (this may take a minute to download weights)...
Target Image: data/test/sample.png
Recognized Text: 'STRUCTI'
---------------------------------------------------
```

### 5. Test Individual Phases

Each module can be run independently:

```bash
python src/phase1_extraction.py       # Test style vector extraction
python src/phase2_3_model.py          # Test text encoder + generator
python src/phase4_5_utils.py          # Test augmentation + quality validation
python testing/phase8_9_evaluation.py # Run downstream evaluation (Phases 8 & 9)
python testing/section3_2_pipeline.py # Test TrOCR OCR inference
```

- style extraction → generation → augmentation/validation → optimization → synthetic data creation → TrOCR fine-tuning → CER/WER evaluation.

Some paper-scale components are simplified/partially replicated in this academic setting, and those limitations are documented in `Final_report.docx`.

## Team Reference

This project was completed by:

- **Aaleen Fatima** — 23L-0652
- **Muhammad Abdullah Iqbal** — 23L-0811
- **Laiba Amjad** — 23L-0642
- **Mariyam Akram** — 23L-0809

GitHub Repository: `abdullahiqbal2610/styleScript`
