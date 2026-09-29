# Finer-CAM-XAI

Independent reproduction study of **Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation (CVPR 2025)** on the **CUB-200-2011** fine-grained bird dataset.

This project was developed for an Explainable AI course and focuses on reproducing the paper's core **Grad-CAM vs Finer-CAM** experiment in the classifier setting.

## Overview

Fine-grained classification often involves visually similar classes that differ only in subtle details. Standard CAM methods explain the target class independently, so they may highlight features that are also useful for similar classes.

Finer-CAM addresses this by comparing the target class with a similar reference class.

### Core idea

Grad-CAM:

\[
y_c
\]

Finer-CAM:

\[
y_c - \gamma y_d
\]

where:

- \(c\) = target class
- \(d\) = similar reference class
- \(\gamma\) = comparison strength

We use **\(\gamma=0.6\)** and aggregate comparisons against the **top-3 similar classes**, following the paper's final design.

The goal is to suppress shared features and emphasize regions that better distinguish the target class.

---

## Pipeline

```text
CUB-200-2011
      ↓
OpenCLIP ViT-B/16
(LAION-400M, frozen)
      ↓
512-D visual features
      ↓
Linear classifier (512 → 200)
      ↓
       ├── Grad-CAM
       │     target logit y_c
       │
       └── Finer-CAM
             y_c - 0.6 y_d
                    ↓
              Saliency map
                    ↓
      Localization / RD / Deletion
```

---


## Step-by-Step: How We Built the Reproduction

This is the actual sequence followed in the project.

### Step 1 — Clone and prepare the project

```bash
git clone https://github.com/Ritik1207-ind/Finer-CAM.git
cd Finer-CAM

python -m venv .venv
source .venv/bin/activate
```

Then the required Python packages were installed. PyTorch was configured with CUDA so that the RTX 3050 could be used.

### Step 2 — Set up the CUB-200-2011 dataset

The CUB-200-2011 dataset was downloaded and extracted into:

```text
data/CUB_200_2011/
```

The dataset contains:

```text
11,788 images
200 classes
5,994 training images
5,794 test images
```

A custom dataset loader was written in:

```text
src/datasets/cub.py
```

It reads the image list, class labels and train/test split.

### Step 3 — Load the OpenCLIP backbone

We used:

```text
OpenCLIP ViT-B/16
LAION-400M pretrained weights
```

The model wrapper was implemented in:

```text
src/models/openclip.py
```

The visual encoder was frozen and used only for feature extraction.

### Step 4 — Extract image features

Each CUB image was passed through OpenCLIP.

```text
Image
  ↓
OpenCLIP preprocessing
  ↓
ViT-B/16
  ↓
512-dimensional feature
```

Features were cached so that the expensive CLIP forward pass did not have to be repeated during classifier training.

Generated files:

```text
features/cub_train.pt
features/cub_test.pt
```

### Step 5 — Train the CUB linear classifier

A simple linear classifier was added:

```text
512 → 200
```

Implementation:

```text
src/models/linear_classifier.py
```

Training setup:

```text
Adam
learning rate = 3e-4
epochs = 100
batch size = 16
```

Final checkpoint:

```text
checkpoints/cub_clip_linear_classifier.pth
```

Test accuracy obtained:

```text
81.07%
```

### Step 6 — Find similar reference classes

Finer-CAM needs classes that are visually/semantically close to the target class.

Instead of using image similarity, we used the classifier's learned weight vectors.

For every class pair:

\[
S_{pq}=
\frac{w_p\cdot w_q}
{\|w_p\|_2\|w_q\|_2}
\]

The target class itself was removed, and the remaining classes were sorted by cosine similarity.

Implementation:

```text
src/similarity/class_similarity.py
```

The top-3 similar classes were used as Finer-CAM references.

### Step 7 — Connect Grad-CAM to the ViT

For the ViT model, the final transformer block was used as the target layer:

```text
model.visual.transformer.resblocks[-1].ln_1
```

The activation has:

```text
[1, 197, 768]
```

The first token is the CLS token, so it was removed.

```text
197 tokens
   ↓ remove CLS
196 patch tokens
   ↓
14 × 14 spatial grid
```

The custom reshape function converts it into:

```text
[1, 768, 14, 14]
```

Implementation:

```text
src/models/reshape.py
```

This makes the ViT activation compatible with CAM computation.

### Step 8 — Implement the Grad-CAM baseline

Grad-CAM was first implemented as the baseline.

For target class \(c\):

\[
\alpha_k^c =
\frac{1}{Z}
\sum_{i,j}
\frac{\partial y_c}{\partial A_{ij}^k}
\]

The gradients are globally averaged over the spatial dimensions, multiplied with the activation maps, summed, passed through ReLU and normalized.

Output:

```text
results/cub_gradcam_baseline.jpg
results/cub_gradcam_baseline.npy
```

### Step 9 — Implement Finer-CAM

The main change was the explanation objective.

Instead of only differentiating the target logit:

\[
y_c
\]

we differentiate:

\[
y_c-\gamma y_d
\]

with:

```text
gamma = 0.6
```

For each of the top-3 reference classes:

```text
target logit - 0.6 × reference logit
```

A CAM is generated for each comparison, then the resulting maps are averaged, ReLU is applied and the final map is normalized.

Implementation:

```text
scripts/test_finer_cam.py
```

Output:

```text
results/cub_finer_cam.jpg
results/cub_finer_cam.npy
```

### Step 10 — Compare Grad-CAM and Finer-CAM visually

The two saliency maps were generated for the same image and combined into:

```text
results/gradcam_vs_finercam.jpg
```

This provides the qualitative comparison required for the XAI part of the assignment.

### Step 11 — Implement quantitative XAI evaluation

Three evaluation pipelines were added.

#### Localization

The CUB bounding boxes were transformed to the same 224×224 coordinate system used by the model.

Implementation:

```text
src/datasets/cub_bbox.py
scripts/eval_localization_pilot.py
```

#### Relative Drop

The salient pixels from the explanation were masked at:

```text
5%
10%
```

and the target/reference confidence changes were compared.

Implementation:

```text
scripts/eval_rd_pilot.py
```

#### Deletion AUC

Pixels were progressively masked:

```text
0%, 10%, 20%, ..., 100%
```

The target-class confidence curve was integrated to obtain the deletion AUC.

Implementation:

```text
scripts/eval_deletion_pilot.py
```

### Step 12 — Run the full test set

After validating everything on individual samples, the evaluation scripts were run over all:

```text
5,794 CUB test images
```

The final metrics were stored in:

```text
results/final_metrics.json
```

Additional outputs include:

```text
results/cub_localization_per_image.csv
results/cub_localization_summary.json
```

### Step 13 — Compare with the paper

Finally, our measured values were placed next to the published CUB results.

This showed that:

```text
Core Finer-CAM methodology reproduced ✅
Dataset reproduced ✅
Grad-CAM baseline reproduced ✅
Quantitative evaluation reproduced ✅
Exact numerical results reproduced ❌
```

The main differences were the classifier accuracy and the final ordering of Grad-CAM vs Finer-CAM on our evaluation.


## Dataset

**CUB-200-2011**

- 200 bird species
- 11,788 images
- 5,994 training images
- 5,794 test images
- Bounding-box annotations

The full 5,794-image test set was used for the final evaluation.

---

## Model

### Visual backbone

**OpenCLIP ViT-B/16**, pretrained on **LAION-400M**.

The visual encoder is frozen during classifier training.

### Classifier

A single linear layer:

```text
512 → 200
```

Training setup:

```text
Optimizer: Adam
Learning rate: 3e-4
Epochs: 100
Batch size: 16
```

The final classifier achieved:

```text
81.07% test accuracy
4697 / 5794 correct
```

The paper reports **58.4% CLIP linear-probe accuracy on CUB**, so the classifier accuracy was one of the major differences in our reproduction.

---

## XAI Implementation

### Grad-CAM

Grad-CAM uses the target-class logit:

\[
\alpha_k^c =
\frac{1}{Z}
\sum_{i,j}
\frac{\partial y_c}{\partial A_{ij}^k}
\]

The weighted feature maps are summed and passed through ReLU to obtain the saliency map.

### Finer-CAM

Finer-CAM changes the explanation target to:

\[
J_{c,d}=y_c-\gamma y_d
\]

and therefore:

\[
\alpha_k^{c,d}
=
\frac{1}{Z}
\sum_{i,j}
\frac{\partial(y_c-\gamma y_d)}
{\partial A_{ij}^k}
\]

which gives:

\[
\alpha_k^{c,d}
=
\alpha_k^c-\gamma\alpha_k^d
\]

This is the key XAI contribution of the project.

### Reference selection

Reference classes are selected using cosine similarity between classifier weight vectors:

\[
S_{pq}=
\frac{w_p\cdot w_q}
{\|w_p\|_2\|w_q\|_2}
\]

The target class is excluded, and the top-3 most similar classes are used.

### ViT feature reshape

The selected transformer activation has shape:

```text
[1, 197, 768]
```

where:

```text
1 CLS token
196 patch tokens = 14 × 14
```

The implementation removes the CLS token and reshapes:

```text
[1, 197, 768]
        ↓
[1, 196, 768]
        ↓
[1, 768, 14, 14]
```

so that standard CAM operations can be applied.

---

## Evaluation

### 1. Localization

Measures how much saliency energy falls inside the CUB object bounding box:

\[
Localization =
\frac{\text{saliency inside bbox}}
{\text{total saliency}}
\]

### 2. Relative Drop

For target confidence \(p_c\), reference confidence \(p_d\), and their post-masking values:

\[
RD=(p_c-p_c^\star)-(p_d-p_d^\star)
\]

We evaluate:

- RD@5%
- RD@10%

The reference is the most similar class according to classifier-weight cosine similarity.

### 3. Deletion AUC

Pixels are ranked by saliency and progressively masked:

```text
0%, 10%, 20%, ..., 100%
```

Target-class confidence is measured after each step and the resulting curve is integrated.

Our deletion evaluation uses **black-pixel replacement in normalized input space**. This is an explicit reproduction choice because the exact author-side perturbation implementation is not fully specified in the released material.

---

## Results

### Published CUB Results

| Metric | Grad-CAM | Finer-CAM |
|---|---:|---:|
| Localization ↑ | 0.582 | **0.629** |
| RD@5% ↑ | 0.101 | **0.112** |
| RD@10% ↑ | 0.113 | **0.121** |
| Deletion AUC ↓ | 0.024 | 0.024 |

### Our Reproduction

| Metric | Grad-CAM | Finer-CAM |
|---|---:|---:|
| Localization ↑ | **0.6243** | 0.6073 |
| RD@5% ↑ | **0.4339** | 0.3835 |
| RD@10% ↑ | **0.5578** | 0.5147 |
| Deletion AUC ↓ | 0.0887 | 0.1044 |

### Main Observation

The published paper reports higher Finer-CAM values than Grad-CAM for localization and relative-drop metrics.

Our reproduction shows the opposite ordering:

```text
Paper:
Finer-CAM > Grad-CAM

Our reproduction:
Grad-CAM > Finer-CAM
```

The classifier accuracy also differs substantially:

```text
Paper: 58.4%
Ours : 81.07%
```

Therefore, this project should be described as a **methodological reproduction with numerical divergence**, not an exact numerical reproduction.

---

## Why Can the Results Differ?

Possible sources of discrepancy include:

- different final classifier learned during linear probing
- exact OpenCLIP checkpoint/configuration
- secondary training details not fully specified in the paper
- transformer target-layer choice
- exact perturbation/masking protocol
- software/library version differences

No single cause was experimentally proven.

---

## Repository Structure

```text
Finer-CAM-XAI/
│
├── src/
│   ├── datasets/
│   │   ├── cub.py
│   │   └── cub_bbox.py
│   ├── models/
│   │   ├── openclip.py
│   │   ├── linear_classifier.py
│   │   └── reshape.py
│   └── similarity/
│       └── class_similarity.py
│
├── scripts/
│   ├── extract_features.py
│   ├── train_linear.py
│   ├── train_cached_classifier.py
│   ├── test_gradcam.py
│   ├── test_finer_cam.py
│   ├── test_rd5.py
│   ├── test_deletion.py
│   ├── eval_localization_pilot.py
│   ├── eval_rd_pilot.py
│   └── eval_deletion_pilot.py
│
├── results/
│   ├── cub_gradcam_baseline.jpg
│   ├── cub_finer_cam.jpg
│   ├── gradcam_vs_finercam.jpg
│   ├── cub_localization_per_image.csv
│   ├── cub_localization_summary.json
│   └── final_metrics.json
│
├── report/
│   └── reproduction_summary.md
│
├── README.md
├── requirements.txt
└── LICENSE.md
```

---

## Main Files

| File | Purpose |
|---|---|
| `cub.py` | CUB dataset loader |
| `cub_bbox.py` | Bounding-box handling and transformation |
| `openclip.py` | OpenCLIP ViT-B/16 wrapper |
| `linear_classifier.py` | 512→200 classifier |
| `reshape.py` | ViT token-to-spatial reshaping |
| `class_similarity.py` | Reference-class selection |
| `extract_features.py` | Feature extraction and caching |
| `train_cached_classifier.py` | Final classifier training |
| `test_gradcam.py` | Grad-CAM baseline |
| `test_finer_cam.py` | Core Finer-CAM implementation |
| `eval_localization_pilot.py` | Localization evaluation |
| `eval_rd_pilot.py` | RD@5% and RD@10% |
| `eval_deletion_pilot.py` | Deletion AUC |

---

## Reproducing the Experiment

From the repository root:

```bash
python scripts/extract_features.py
python scripts/train_cached_classifier.py
python scripts/test_gradcam.py
python scripts/test_finer_cam.py
python scripts/eval_localization_pilot.py
python scripts/eval_rd_pilot.py
python scripts/eval_deletion_pilot.py
```

The scripts generate the qualitative visualizations and quantitative evaluation files under `results/`.

---

## Course Project Coverage

This project covers the required Phase 1 deliverables:

```text
Model implementation          ✅
XAI method implementation     ✅
Same dataset                  ✅
2–3 main results              ✅
GitHub repository             ✅
Paper comparison              ✅
1-page replication summary    ✅
```

Four quantitative XAI metrics were evaluated on the full CUB test set:

```text
Localization
RD@5%
RD@10%
Deletion AUC
```

---

## Key Takeaway

The main XAI idea behind Finer-CAM is **contrastive explanation**.

Grad-CAM asks:

> Why does the model predict the target class?

Finer-CAM asks:

> Why does the model prefer the target class over a similar class?

This changes the explanation target from a single class logit to a target-versus-reference logit difference, allowing the method to suppress shared features and emphasize class-discriminative details.

---

## Citation

```bibtex
@InProceedings{zhang2025finer,
    author    = {Zhang, Ziheng and Gu, Jianyang and Chowdhury, Arpita
                 and Mai, Zheda and Carlyn, David and Berger-Wolf, Tanya
                 and Su, Yu and Chao, Wei-Lun},
    title     = {Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    year      = {2025},
    pages     = {9611-9620},
    doi       = {10.1109/CVPR52734.2025.00898}
}
```

## References

- Zhang et al., **Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation**, CVPR 2025.
- Selvaraju et al., **Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization**, ICCV 2017.
- Wah et al., **The Caltech-UCSD Birds-200-2011 Dataset**, 2011.
