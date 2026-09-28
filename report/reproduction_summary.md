# Finer-CAM Reproduction Study — CUB-200-2011

## Objective

This project reproduces the core Finer-CAM visual explanation pipeline on the CUB-200-2011 fine-grained bird classification dataset. The reproduction uses a frozen OpenCLIP ViT-B/16 visual encoder, a learned linear classifier, Grad-CAM as the baseline explanation method, and a paper-faithful Finer-CAM target-versus-reference objective.

## Methodology

The CUB-200-2011 dataset contains 200 bird classes and 5,994 training images and 5,794 test images. OpenCLIP ViT-B/16 pretrained on LAION-400M was used as the visual backbone. Its parameters were frozen and a linear classifier over 512-dimensional image features was trained using Adam with a learning rate of 3e-4 for 100 epochs.

For Finer-CAM, the target class was compared against three reference classes selected using cosine similarity between classifier weight vectors. The Finer-CAM coefficient was set to gamma = 0.6. Grad-CAM and Finer-CAM explanations were evaluated using localization, relative difference (RD@5% and RD@10%), and deletion AUC metrics.

## Results

The trained classifier achieved 81.07% classification accuracy on the 5,794-image CUB test set.

| Metric | Grad-CAM | Finer-CAM |
|---|---:|---:|
| Localization | 0.6243 | 0.6073 |
| RD@5% | 0.4339 | 0.3835 |
| RD@10% | 0.5578 | 0.5147 |
| Deletion AUC | 0.0887 | 0.1044 |

## Comparison With the Paper

The paper reports 58.4% classification accuracy on CUB for its corresponding setup, with reported localization values of 0.582 for Grad-CAM and 0.629 for Finer-CAM. The reproduction therefore does not numerically match the published results.

Several implementation details required for exact numerical reproduction are not fully specified in the released materials. In particular, this reproduction uses an explicit deletion protocol with 10% masking increments and black-pixel replacement in normalized input space. Therefore, the reported values should be interpreted as results from this reproducible implementation rather than as exact re-measurements of the authors' evaluation pipeline.

## Conclusion

The reproduction successfully implements the main Finer-CAM idea, integrates it with a transformer-based OpenCLIP visual encoder, and evaluates both Grad-CAM and Finer-CAM over the complete CUB-200-2011 test set. The code, evaluation scripts, qualitative visualizations, and final metrics are included in the repository for reproducibility.

Reference: Ziheng Zhang et al., “Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation,” CVPR 2025, DOI: 10.1109/CVPR52734.2025.00898.
