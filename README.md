# Knowledge Distillation for Compact and Deployable Facial Expression Recognition on Edge Devices

**Master's Thesis — BTU Cottbus-Senftenberg**
**Author:** Syed Faiq Hassan

---

## Overview

This repository contains the complete implementation for compressing MAE-DFER (84.9M parameters, 49.8G FLOPs) into a lightweight MobileNetV2+GRU student model (3.41M parameters, ~0.30G FLOPs) using three knowledge distillation strategies on the DFEW benchmark.

The student model achieves **9.26 ms inference per clip** on a Tesla T4 GPU — **4.5× faster** than the teacher — while retaining 71.7% of teacher accuracy.

---

## Results

| Strategy | UAR % | WAR % | Params |
|---|---|---|---|
| MAE-DFER Teacher | 63.41 | 74.43 | 84.9M |
| Strategy 1 — Soft-Logit Matching | **45.49** | **55.01** | 3.41M |
| Strategy 2 — Feature Transfer | 45.48 | 53.93 | 3.41M |
| Strategy 3 — Task-Specific Loss | 44.71 | 54.65 | 3.41M |
| No-KD Baseline | 44.06 | 53.70 | 3.41M |

All results are averaged across 5-fold cross-validation on DFEW.

---

## Requirements

### Option A — Run on Kaggle (Recommended)

This notebook was developed and trained on **Kaggle** using a free Tesla T4 GPU (15.6GB VRAM). This is the recommended way to run it because the GPU is free and the environment matches exactly.

1. Go to [kaggle.com](https://www.kaggle.com) and create a free account
2. Create a new notebook and upload `kd-fer-thesis.ipynb`
3. Enable GPU: Settings → Accelerator → GPU T4 x1
4. Follow the setup steps below

### Option B — Run Locally

You can run this on your own machine if you have a GPU with at least 8GB VRAM.

```bash
pip install -r requirements.txt
jupyter notebook kd-fer-thesis.ipynb
```

Note: Training one fold takes approximately 6–7 hours on a Tesla T4. On a weaker GPU it will take longer.

---

## Setup Before Running

### 1. Kaggle API Token (Cell 1)

Cell 1 sets your Kaggle API token. This is needed to push and restore checkpoints from a Kaggle Dataset during training.

**Get your token:**
1. Go to kaggle.com → click your profile → Settings
2. Scroll to the API section → click **Create New Token**
3. This downloads a `kaggle.json` file — open it and copy the key

**Then in Cell 1, replace the placeholder:**
```python
os.environ['KAGGLE_API_TOKEN'] = 'YOUR_KAGGLE_API_TOKEN_HERE'
```

> If you are running locally and do not want to use Kaggle Datasets for checkpoint storage, you can skip Cell 1 and comment out the `push_to_dataset()` call at the bottom of Cell 15. Your checkpoints will save locally only.

---

### 2. Dataset (Cell 8)

This project uses the **DFEW** dataset (Jiang et al., ACM MM 2020).

**Request access:** https://dfew-dataset.github.io/

After downloading, the expected folder structure is:

```
data/
└── clip_224x224_16f/
    ├── 00001/
    │   ├── 0001.jpg
    │   ├── 0002.jpg
    │   └── ...
    └── ...
```

**On Kaggle:** Upload DFEW as a Kaggle Dataset and update the paths in Cell 8:
```python
CLIP_BASE  = '/kaggle/input/YOUR-DATASET-NAME/clip_224x224_16f'
TRAIN_CSV  = '/kaggle/input/YOUR-DATASET-NAME/train(single-labeled)'
TEST_CSV   = '/kaggle/input/YOUR-DATASET-NAME/test(single-labeled)'
```

---

### 3. Teacher Weights (Cell 5)

The MAE-DFER teacher weights are downloaded automatically via Google Drive in Cell 5. No manual action needed — just run Cell 5 and it downloads all 5 fold weights (~1GB each).

Source: [github.com/sunlicai/MAE-DFER](https://github.com/sunlicai/MAE-DFER)

---

## How to Run Training

### Main Training — Strategy and Fold Selection (Cell 15)

At the top of Cell 15 there are only **two values you need to change** for each training run:

```python
# ══════════════════════════════════════════════════
# CHANGE ONLY THESE TWO VALUES FOR EACH RUN
STRATEGY = 1   # 1 = Soft-Logit Matching
               # 2 = Feature Transfer
               # 3 = Task-Specific Loss

FOLD     = 1   # 1, 2, 3, 4, or 5
# ══════════════════════════════════════════════════
```

**Full training requires 15 runs** (3 strategies × 5 folds). Run them in this order:

```
STRATEGY=1, FOLD=1  →  STRATEGY=1, FOLD=2  →  ...  →  STRATEGY=1, FOLD=5
STRATEGY=2, FOLD=1  →  STRATEGY=2, FOLD=2  →  ...  →  STRATEGY=2, FOLD=5
STRATEGY=3, FOLD=1  →  STRATEGY=3, FOLD=2  →  ...  →  STRATEGY=3, FOLD=5
```

Each run saves two checkpoints automatically:
- `student_s{STRATEGY}_fold{FOLD}.pth` — latest epoch
- `student_s{STRATEGY}_fold{FOLD}_best.pth` — best UAR epoch

---

### Simulation Experiments (Cell 15 — after main training)

After completing all 15 main runs, simulation experiments test the best strategy (Strategy 1) under hardware constraints. Change Cell 15 as follows for each simulation:

**Simulation 1 — Input Resolution (change `input_size` in `build_dataloaders`):**
```python
# Test these four levels one at a time:
train_loader, test_loader = build_dataloaders(fold=1, input_size=112)  # or 56 or 28
```

**Simulation 2 — Frame Sampling (change `num_frames` in TWO places):**
```python
# 1. In build_dataloaders:
train_loader, test_loader = build_dataloaders(fold=1, num_frames=8)  # or 4 or 2

# 2. In the student model:
student = MobileNetV2GRU(num_classes=7, width_mult=1.0, gru_hidden=256,
                         num_frames=8, dropout=0.3).to(device)  # match num_frames
```

> Note: Simulations run on Strategy 1 only. SoftLogitLoss does not have a `num_frames` parameter so no change is needed there.

**Simulation 3 — Model Width (change `width_mult` in student):**
```python
student = MobileNetV2GRU(num_classes=7, width_mult=0.75, ...)  # or 0.50
```

---

## Optional Cells — Graphs and Evaluation

The following cells at the end of the notebook are **optional**. Run them after training is complete if you want to generate plots and evaluation outputs:

| Cell | What it does | When to run |
|---|---|---|
| Inference timing cell | Measures student and teacher inference speed in ms | After training, optional |
| Per-class accuracy cell | Loads all best checkpoints and computes accuracy per emotion | After all 15 folds complete |
| Confusion matrix cell | Generates and saves confusion matrix PNGs | After all 15 folds complete |
| Simulation plots cell | Generates UAR vs constraint level plots | After all simulations complete |

These cells do not affect training — they are purely for analysis and visualisation.

---

## Notebook Structure

| Cell | Description |
|---|---|
| Cell 1 | Kaggle API token setup |
| Cell 2 | Install dependencies |
| Cell 3 | Verify GPU |
| Cell 4 | Clone MAE-DFER repository |
| Cell 5 | Download teacher weights (5 folds) |
| Cell 6 | Load teacher model |
| Cell 7 | Teacher forward pass test |
| Cell 8 | DFEW DataLoader |
| Cell 9 | Student model (MobileNetV2+GRU) |
| Cell 10 | Strategy 1 loss (Soft-Logit Matching) |
| Cell 11 | Strategy 2 loss (Feature Transfer) |
| Cell 12 | Strategy 3 loss (Task-Specific Loss) |
| Cell 13 | Training functions |
| Cell 14 | Checkpoint system (save / push / restore) |
| Cell 15 | **Training loop — change STRATEGY and FOLD here** |
| Cell 16 | Per-class accuracy and confusion matrix evaluation |

---

## Checkpoint System

Checkpoints are saved automatically after every epoch and pushed to a Kaggle Dataset after each fold completes. This protects against session expiry on Kaggle.

At the start of each new session, Cell 15 automatically restores all previous checkpoints from the dataset before training begins. You never lose progress.

If you are running locally, comment out the `push_to_dataset()` call at the bottom of Cell 15 and the `restore_from_dataset()` call at the top. Checkpoints will then save to `./checkpoints/` only.

---

## Citation

If you use this code, please cite:

```
Syed Faiq Hassan. "Knowledge Distillation for Compact and Deployable
Facial Expression Recognition on Edge Devices."
Master's Thesis, BTU Cottbus-Senftenberg, 2026.
```

**Teacher model:**
```
Sun, L. et al. "MAE-DFER: Efficient Masked Autoencoder for Self-supervised
Dynamic Facial Expression Recognition." ACM MM, 2023.
```

**Dataset:**
```
Jiang, X. et al. "DFEW: A Large-Scale Database for Recognizing Dynamic
Facial Expressions in the Wild." ACM MM, 2020.
```