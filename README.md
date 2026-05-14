# StyleScript

This repository is a **project implementation of the paper**:
**StyleScript: A Structured Data Augmentation Framework for Transformer-Based OCR in Engineering Documents**.

- Paper in repo: [`styleScript.pdf`](./styleScript.pdf)
- Implementation report: [`Final_report.docx`](./Final_report.docx)
- Latest experimentation notebook: [`ai-project.ipynb`](./ai-project.ipynb)

## 1) StyleScript Paper Focus

The paper proposes a structured augmentation pipeline for OCR in engineering-document settings. The core flow is:

1. Extract style features from handwriting images.
2. Generate style-controlled synthetic word images.
3. Fine-tune TrOCR using generated data.
4. Evaluate impact with CER/WER.

Our repository implements this pipeline in Python modules, covering training, generation, OCR inference, and downstream evaluation.

## 2) Repository Structure (Current)

```text
styleScript/
├── README.md
├── styleScript.pdf                 # Original StyleScript paper
├── Final_report.docx               # Formal implementation analysis and findings
├── ai-project.ipynb                # Latest notebook experiments/results
├── main.py                         # Training entry point (generator + losses)
├── config.json                     # Hyperparameters, paths, loss weights
├── stylescript_generator.pth       # Saved generator checkpoint
├── training_loss_curve.png         # Training loss visualization
├── data/
│   ├── annotations.csv             # filename ↔ text mapping
│   └── raw/                        # input images
└── src/
    ├── phase1_extraction.py        # Style feature extraction
    ├── phase2_3_model.py           # Text encoder + style-conditioned generator
    ├── phase4_5_utils.py           # Augmentations + quality checks
    ├── section3_2_pipeline.py      # TrOCR OCR inference pipeline
    └── phase8_9_evaluation.py      # Synthetic generation + fine-tuning + CER/WER
```

## 3) Latest `ai-project.ipynb` Findings and Results

The latest notebook compares baseline OCR, ScrabbleGAN-style augmentation, and StyleScript-style augmentation.

### Reported replication table (from notebook output)

| Condition | Paper Small (CER/WER) | Our Replication (CER/WER) |
|---|---|---|
| Baseline | 1.81% / 6.74% | 44.62% / 85.33% |
| ScrabbleGAN | 2.07% / 6.87% | 6.08% / 11.33% |
| StyleScript | 1.54% / 6.48% | **1.4% / 2.67%** |

### Additional notebook run outputs

- StyleScript enhanced OCR run reported: **CER 0.0140, WER 0.0200**.
- ScrabbleGAN enhanced OCR run reported: **CER 0.0608, WER 0.1133**.

### What these results show

- Within our replication environment, the StyleScript pipeline gives the best OCR performance among tested settings.
- The implementation captures the paper’s core objective: using structured synthetic augmentation to improve TrOCR recognition quality.
- Differences versus paper baselines are expected due to dataset, scale, and resource constraints described in `Final_report.docx`.

## 4) Implementation Status vs Paper

Based on the report and codebase, this project is an **honest implementation of the StyleScript paper** with end-to-end coverage of the main algorithmic path:

- style extraction → generation → augmentation/validation → optimization → synthetic data creation → TrOCR fine-tuning → CER/WER evaluation.

Some paper-scale components are simplified/partially replicated in this academic setting, and those limitations are documented in `Final_report.docx`.

## 5) Team Reference

This project was completed by:

- **Aaleen Fatima** — 23L-0652
- **Muhammad Abdullah Iqbal** — 23L-0811
- **Laiba Amjad** — 23L-0642
- **Mariyam Akram** — 23L-0809

Repository: `abdullahiqbal2610/styleScript`
