import numpy as np
import torch
import open_clip

from src.datasets.cub import CUB200Dataset
from src.models.linear_classifier import LinearClassifier
from src.similarity.class_similarity import (
    load_classifier_weights,
    compute_similarity,
    get_top_references,
)

device = "cuda"

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion400m_e31",
    device=device,
)

clip_model.eval()

classifier = LinearClassifier(512, 200).to(device)
classifier.load_state_dict(
    torch.load(
        "checkpoints/cub_clip_linear_classifier.pth",
        map_location=device,
    )
)
classifier.eval()


class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model, classifier):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.clip_model.encode_image(x))


model = CLIPClassifier(clip_model, classifier)

dataset = CUB200Dataset(
    "data/CUB_200_2011",
    train=False,
    transform=preprocess,
)

image, label = dataset[0]
image = image.unsqueeze(0).to(device)

target_class = int(label)

weights = load_classifier_weights(
    "checkpoints/cub_clip_linear_classifier.pth"
)

similarity = compute_similarity(weights)

reference_indices, reference_scores = get_top_references(
    similarity,
    target_class,
    top_k=3,
)

reference_class = int(reference_indices[0])

mean = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073],
    device=device,
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711],
    device=device,
).view(1, 3, 1, 1)

normalized_black = -mean / std


def mask_top_percent(image, cam, percent):
    flat = cam.flatten()

    num_pixels = int(flat.size * percent)

    indices = torch.topk(
        torch.from_numpy(flat),
        k=num_pixels,
    ).indices.numpy()

    mask = np.zeros(flat.shape, dtype=bool)
    mask[indices] = True
    mask = torch.from_numpy(
        mask.reshape(cam.shape)
    ).to(device)

    masked = image.clone()

    masked = torch.where(
        mask.unsqueeze(0).unsqueeze(0),
        normalized_black,
        masked,
    )

    return masked


def confidence(logits, cls):
    return torch.softmax(logits, dim=1)[0, cls].item()


def rd_score(
    p_target,
    p_target_masked,
    p_ref,
    p_ref_masked,
):
    return (
        (p_target - p_target_masked)
        -
        (p_ref - p_ref_masked)
    )


with torch.no_grad():
    original_logits = model(image)

p_target = confidence(
    original_logits,
    target_class,
)

p_ref = confidence(
    original_logits,
    reference_class,
)

print("Target class:", target_class)
print("Reference class:", reference_class)
print("Most similar class used for RD:", reference_class)

print(
    "\nOriginal target confidence:",
    p_target
)

print(
    "Original reference confidence:",
    p_ref
)


# -----------------------------
# Grad-CAM
# -----------------------------
gradcam = np.load(
    "results/cub_gradcam_baseline.npy"
)

masked_gradcam = mask_top_percent(
    image,
    gradcam,
    0.05,
)

with torch.no_grad():
    masked_logits = model(masked_gradcam)

p_target_g = confidence(
    masked_logits,
    target_class,
)

p_ref_g = confidence(
    masked_logits,
    reference_class,
)

rd_grad = rd_score(
    p_target,
    p_target_g,
    p_ref,
    p_ref_g,
)


# -----------------------------
# Finer-CAM
# -----------------------------
finercam = np.load(
    "results/cub_finer_cam.npy"
)

masked_finercam = mask_top_percent(
    image,
    finercam,
    0.05,
)

with torch.no_grad():
    masked_logits = model(masked_finercam)

p_target_f = confidence(
    masked_logits,
    target_class,
)

p_ref_f = confidence(
    masked_logits,
    reference_class,
)

rd_finer = rd_score(
    p_target,
    p_target_f,
    p_ref,
    p_ref_f,
)


print("\n===== RD@5% =====")

print("\nGrad-CAM")
print(
    "Masked target confidence:",
    p_target_g
)
print(
    "Masked reference confidence:",
    p_ref_g
)
print("RD@5%:", rd_grad)

print("\nFiner-CAM")
print(
    "Masked target confidence:",
    p_target_f
)
print(
    "Masked reference confidence:",
    p_ref_f
)
print("RD@5%:", rd_finer)
