# Finer-CAM-XAI

Independent reproduction study of **Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation** on the **CUB-200-2011** fine-grained bird classification dataset.

This repository contains the implementation and evaluation developed for the reproduction study, including a Grad-CAM baseline, a paper-faithful Finer-CAM formulation, localization evaluation, RD@5% / RD@10%, deletion AUC, qualitative visualizations, and the final experimental results.

## Project Setup

- Dataset: CUB-200-2011
- Visual backbone: OpenCLIP ViT-B/16 pretrained on LAION-400M
- Visual encoder: frozen during classifier training
- Classifier: linear classifier over 512-dimensional visual features
- Baseline: Grad-CAM
- Finer-CAM reference classes: top-3 classes selected using cosine similarity between classifier weight vectors
- Finer-CAM gamma: 0.6
- Test images: 5,794

## Results

| Metric | Grad-CAM | Finer-CAM |
|---|---:|---:|
| Localization | 0.6243 | 0.6073 |
| RD@5% | 0.4339 | 0.3835 |
| RD@10% | 0.5578 | 0.5147 |
| Deletion AUC | 0.0887 | 0.1044 |

Classification accuracy of the trained linear classifier was **81.07%** on the CUB-200-2011 test set.

These numbers are reproduction results from the implementation in this repository and should not be interpreted as exact numerical reproduction of the published paper. Some training and evaluation details are not fully specified in the released materials. In particular, deletion evaluation here uses 10% masking increments and black-pixel replacement in normalized input space.

## Repository Structure

```text
src/                    Dataset, model, preprocessing, and similarity code
scripts/                Training, CAM generation, and evaluation scripts
results/                Final metrics and qualitative outputs
report/                 Reproduction summary
