
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
