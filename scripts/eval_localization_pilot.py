import csv
import json
from pathlib import Path
import numpy as npp
import torch
import torch.nn.functional as F
import open_clip

from src.datasets.cub import CUB200Dataset
from src.datasets.cub_bbox import CUBBoundingBoxes
from src.models.linear_classifier import LinearClassifier
from src.similarity.class_similarity import (
    load_classifier_weights,
    compute_similarity,
    get_top_references,
)


DEVICE = "cuda"
ROOT = "data/CUB_200_2011"
CHECKPOINT = "checkpoints/cub_clip_linear_classifier.pth"
N = 5794
GAMMA = 0.6


class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model, classifier):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.clip_model.encode_image(x))


def reshape_tokens(x):
    x = x[:, 1:, :]
    x = x.reshape(x.shape[0], 14, 14, x.shape[2])
    return x.permute(0, 3, 1, 2)


def normalize_cam(cam):
    cam = F.relu(cam)
    cam = cam - cam.min()

    if cam.max() > 0:
        cam = cam / cam.max()

    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    return cam[0, 0].detach().cpu().numpy()


def localization(cam, bbox):
    x1, y1, x2, y2 = bbox

    ix1 = max(0, int(npp.floor(x1)))
    iy1 = max(0, int(npp.floor(y1)))
    ix2 = min(224, int(npp.ceil(x2)))
    iy2 = min(224, int(npp.ceil(y2)))

    total = cam.sum()
    inside = cam[iy1:iy2, ix1:ix2].sum()

    return float(inside / (total + 1e-12))


print("Loading model...")

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion400m_e31",
    device=DEVICE,
)

clip_model.eval()

classifier = LinearClassifier(512, 200).to(DEVICE)
classifier.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE)
)
classifier.eval()

model = CLIPClassifier(clip_model, classifier)

dataset = CUB200Dataset(
    ROOT,
    train=False,
    transform=preprocess,
)

boxes = CUBBoundingBoxes(ROOT)

weights = load_classifier_weights(CHECKPOINT)
similarity = compute_similarity(weights)

target_layer = clip_model.visual.transformer.resblocks[-1].ln_1

results_grad = []
results_finer = []
records = []

test_indices = npp.linspace(
    0,
    len(dataset) - 1,
    N,
    dtype=int,
)

print(f"\nEvaluating {N} evenly-spaced test images...\n")

for index in test_indices:

    image_tensor, label = dataset[index]
    image = image_tensor.unsqueeze(0).to(DEVICE)
    image.requires_grad_(True)

    image_id = dataset.get_image_id(index)
    bbox = boxes.get_transformed_box(image_id, size=224)

    activation = {}

    def hook(module, inpputs, output):
        activation["value"] = output

    handle = target_layer.register_forward_hook(hook)

    logits = model(image)

    predicted = logits.argmax(dim=1).item()

    # -------------------------
    # Grad-CAM
    # -------------------------
    grad_objective = logits[:, label].sum()

    grad = torch.autograd.grad(
        grad_objective,
        activation["value"],
        retain_graph=True,
    )[0]

    acts = reshape_tokens(activation["value"])
    grad = reshape_tokens(grad)

    weights_grad = grad.mean(
        dim=(2, 3),
        keepdim=True,
    )

    grad_cam = normalize_cam(
        (weights_grad * acts).sum(
            dim=1,
            keepdim=True,
        )
    )

    # -------------------------
    # Finer-CAM
    # -------------------------
    reference_indices, reference_scores = get_top_references(
        similarity,
        label,
        top_k=3,
    )

    finer_maps = []

    for ref_class in reference_indices.tolist():

        objective = (
            logits[:, label]
            - GAMMA * logits[:, ref_class]
        ).sum()

        ref_grad = torch.autograd.grad(
            objective,
            activation["value"],
            retain_graph=True,
        )[0]

        ref_grad = reshape_tokens(ref_grad)

        channel_weights = ref_grad.mean(
            dim=(2, 3),
            keepdim=True,
        )

        raw_map = (
            channel_weights * acts
        ).sum(
            dim=1,
            keepdim=True,
        )

        finer_maps.append(raw_map)

    finer_cam = normalize_cam(
        torch.stack(finer_maps).mean(dim=0)
    )

    handle.remove()

    grad_score = localization(
        grad_cam,
        bbox,
    )

    finer_score = localization(
        finer_cam,
        bbox,
    )

    results_grad.append(grad_score)
    results_finer.append(finer_score)

    records.append(
        {
            "dataset_index": int(index),
            "image_id": int(image_id),
            "label": int(label),
            "prediction": int(predicted),
            "correct": bool(predicted == label),
            "gradcam_localization": float(grad_score),
            "finer_cam_localization": float(finer_score),
        }
    )

    print(
        f"[{len(records):04d}/{N}] "
        f"image_id={image_id} "
        f"label={label} "
        f"pred={predicted} "
        f"Grad-CAM={grad_score:.4f} "
        f"Finer-CAM={finer_score:.4f}"
    )


print("\n===== PILOT RESULT =====")

print(
    "Mean Grad-CAM localization:",
    npp.mean(results_grad),
)

mean_grad = float(npp.mean(results_grad))
mean_finer = float(npp.mean(results_finer))
accuracy = float(npp.mean([r["correct"] for r in records]) * 100)

print(
    "Mean Finer-CAM localization:",
    mean_finer,
)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / "cub_localization_per_image.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset_index",
            "image_id",
            "label",
            "prediction",
            "correct",
            "gradcam_localization",
            "finer_cam_localization",
        ],
    )
    writer.writeheader()
    writer.writerows(records)

summary = {
    "dataset": "CUB-200-2011",
    "num_test_images": len(records),
    "classification_accuracy_percent": accuracy,
    "mean_gradcam_localization": mean_grad,
    "mean_finer_cam_localization": mean_finer,
    "gamma": GAMMA,
    "references_per_image": 3,
}

json_path = results_dir / "cub_localization_summary.json"

with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved per-image results:", csv_path)
print("Saved summary:", json_path)
