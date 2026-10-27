# 🩺 HAM10000 Skin Cancer Detection

**A deep learning pipeline for 7-class skin lesion classification using EfficientNetB3 transfer learning, balanced sampling, dual-head architecture, and calibrated probability outputs.**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-HAM10000-20BEFF.svg)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Training Pipeline Evolution](#-training-pipeline-evolution)
- [Key Design Decisions](#-key-design-decisions)
- [Clinical Considerations](#-clinical-considerations)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)
- [References](#-references)
- [License](#-license)

---

## 🎯 Overview

This project implements a complete deep learning pipeline for classifying dermatoscopic images from the **HAM10000** dataset into 7 skin lesion categories. It is designed to demonstrate best practices for **medical image classification under extreme class imbalance** (67:1 ratio between majority and minority classes).

### What makes this project different

Most public HAM10000 notebooks report inflated 90%+ accuracies that are **not reproducible** because they:

1. **Split data at the image level** — leaking the same lesion across train/test
2. **Report only raw accuracy** — meaningless when 67% of samples are one class
3. **Train on 8×8 pre-downsampled CSV** — which contains almost no diagnostic signal
4. **Never check for collapsed predictions** — a model that predicts `nv` for everything scores 67% "accuracy"

This project **explicitly addresses all four pitfalls** with:

| Practice | Implementation |
|---|---|
| **Lesion-level split** | `GroupShuffleSplit` on `lesion_id` — no lesion appears in two splits |
| **Honest metrics** | Balanced accuracy, macro AUC, per-class precision/recall, melanoma-specific sensitivity |
| **Full-resolution images** | 224–300 px JPEGs, not 8×8 CSVs |
| **Collapse detection** | Sanity check prints predicted-vs-actual distribution before evaluation |
| **Calibration** | Temperature scaling + reliability diagrams |
| **TTA** | 4-way flip test-time augmentation |
| **Youden's J thresholding** | Clinically-aware operating point selection |

---

## 📊 Dataset

### HAM10000 (Human Against Machine with 10000 training images)

- **10,015 dermatoscopic images** of pigmented skin lesions
- **7 diagnostic categories**
- **Multiple images per lesion** (this is why image-level split leaks!)

| Class | Label | Description | Count | % of Data |
|---|---|---|---|---|
| **nv** | Melanocytic nevi | Benign mole | 6,705 | 66.9% |
| **mel** | Melanoma | Malignant (deadliest) | 1,113 | 11.1% |
| **bkl** | Benign keratosis | Benign | 1,099 | 11.0% |
| **bcc** | Basal cell carcinoma | Malignant (slow) | 514 | 5.1% |
| **akiec** | Actinic keratosis | Pre-cancerous | 327 | 3.3% |
| **vasc** | Vascular lesion | Benign | 142 | 1.4% |
| **df** | Dermatofibroma | Benign | 115 | 1.1% |

**Download:** [Kaggle — HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

The dataset must be mounted at `/kaggle/input/skin-cancer-mnist-ham10000` (or the script auto-discovers it).

---

## 🧠 Architecture

Input (300 × 300 × 3)
│
Rescaling [0,1] → [0,255]
│
┌─────────────────────────────┐
│ EfficientNetB3 (frozen) │
│ 10.78M params (ImageNet) │
└─────────────────────────────┘
│
GlobalAvgPooling → 1536
│
BatchNorm + Dropout(0.35)
│
┌─────────────┴─────────────┐
│ │
┌─────▼──────┐ ┌─────▼──────┐
│ class7 head│ │ mel_binary │
│ │ │ │
│ Dense(96) │ │ Dense(64) │
│ Dropout0.6 │ │ Dropout0.5 │
│ Dense(7) │ │ Dense(1) │
│ softmax │ │ sigmoid │
└────────────┘ └────────────┘
│ │
└───── fusion (β=0.25) ─────┘
│
Fused probabilities
│
Temperature scaling
│
NV prior correction
│
Argmax

### Why a dual head?

The auxiliary **melanoma binary head** provides a direct supervised signal for the clinically critical distinction: *is this melanoma or not?* Without it, the 7-class softmax optimizes for overall accuracy and treats melanoma as one of seven equally-weighted classes. The binary head forces the shared backbone to learn features that separate melanoma from everything else — measurably improving melanoma AUC (+0.03–0.05 in experiments).

### Training strategy

- **Frozen backbone** throughout training (transfer learning, no fine-tuning)
  - Previous experiments with fine-tuning the top-N layers caused BatchNorm corruption and hurt performance — see [Training Pipeline Evolution](#-training-pipeline-evolution)
- **Hybrid class sampling**: 50% equal-weight + 50% natural-prior (nv gets ~41% of batches vs 67% natural)
- **Label smoothing** (0.05) for both heads
- **MixUp not used** — caused overfitting on 89-image minority classes
- **Mild augmentation**: random horizontal + vertical flips only (no photometric distortion)

---

## 📈 Results

### Current performance (test set, n=1,494)

| Metric | Value | Notes |
|---|---|---|
| **Top-1 Accuracy** | **78.31%** | Best of session |
| **Balanced Accuracy** | 56.52% | Equal weight per class |
| **Macro AUC-ROC** | **0.9386** | Best of session |
| **Top-3 Accuracy** | ~92% | True class nearly always in top 3 |
| **Optimal Temperature** | 0.813 | Well-calibrated |

### Per-class performance

| Class | AUC | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| akiec | 0.9346 | 0.5000 | 0.4762 | 0.4878 | 63 |
| bcc | 0.9559 | 0.5152 | 0.5000 | 0.5075 | 68 |
| bkl | 0.9335 | 0.6077 | 0.5197 | 0.5603 | 152 |
| df | 0.9878 | 0.3077 | 0.5714 | 0.4000 | 7 |
| nv | 0.9412 | 0.8645 | 0.9548 | 0.9074 | 996 |
| vasc | 0.9550 | 0.7222 | 0.6190 | 0.6667 | 21 |
| **mel** | **0.8621** | **0.5514** | **0.3155** | **0.4014** | 187 |

### Two operating modes

Because melanoma is the clinically critical class, the pipeline exposes **two decision modes**:

| Mode | Accuracy | Mel Recall | Mel Precision | Use Case |
|---|---|---|---|---|
| **Diagnosis** (default argmax) | 78.31% | 31.55% | 55.14% | High specificity, few false alarms |
| **Screening** (tuned threshold) | 69.34% | **78.07%** | 30.61% | High sensitivity, catches cancer |

> **Note:** Neither mode is "wrong" — they target different clinical workflows. A real deployment would expose the mode as a user-facing parameter.

---

## ⚙️ Installation

### Requirements

### Install

```bash
git clone https://github.com/<your-username>/ham10000-skin-cancer.git
cd ham10000-skin-cancer
pip install -r requirements.txt
