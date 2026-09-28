import numpy as np
import torch
import torch.nn.functional as F
import open_clip

from pathlib import Path
from PIL import Image

from src.datasets.cub import CUB200Dataset
from src.models.linear_classifier import LinearClassifier
from src.similarity.class_similarity import (
    load_classifier_weights,
    compute_similarity,
    get_top_references,
)


device = "cuda"

# -----------------------------
# Load OpenCLIP
# -----------------------------
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion400m_e31",
    device=device,
)
clip_model.eval()


# -----------------------------
# Load classifier
# -----------------------------
classifier = LinearClassifier(
    input_dim=512,
    num_classes=200,
).to(device)

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


# -----------------------------
# Load CUB sample
# -----------------------------
dataset = CUB200Dataset(
    "data/CUB_200_2011",
    train=False,
    transform=preprocess,
)

image_tensor, label = dataset[0]

input_tensor = image_tensor.unsqueeze(0).to(device)
input_tensor.requires_grad_(True)


# -----------------------------
# Target class
# -----------------------------
with torch.no_grad():
    logits = model(input_tensor)

predicted_class = logits.argmax(dim=1).item()
target_class = int(label)


# -----------------------------
# Find top-3 similar classes
# -----------------------------
weights = load_classifier_weights(
    "checkpoints/cub_clip_linear_classifier.pth"
)

similarity = compute_similarity(weights)

references = get_top_references(
    similarity,
    target_class,
    top_k=3,
)

print("Target class:", target_class)
print("Predicted class:", predicted_class)
reference_indices, reference_scores = references

print("References:")

for ref_class, score in zip(
    reference_indices.tolist(),
    reference_scores.tolist()
):
    print(f"  class={ref_class}, similarity={score:.4f}")


# -----------------------------
# Target layer
# -----------------------------
target_layer = clip_model.visual.transformer.resblocks[-1].ln_1

activation = {}


def save_activation(module, inputs, output):
    activation["value"] = output


handle = target_layer.register_forward_hook(save_activation)


# -----------------------------
# Forward pass
# -----------------------------
logits = model(input_tensor)

acts = activation["value"]

print("\nRaw activation:", acts.shape)


# Remove CLS token and reshape
acts = acts[:, 1:, :]
acts = acts.reshape(1, 14, 14, 768)
acts = acts.permute(0, 3, 1, 2)


# -----------------------------
# Finer-CAM
# Eq. 5:
# d(y_c - gamma*y_d) / dA
# -----------------------------
gamma = 0.6

raw_maps = []

for ref_class, similarity_score in zip(reference_indices.tolist(), reference_scores.tolist()):

    objective = (
        logits[:, target_class]
        - gamma * logits[:, ref_class]
    ).sum()

    grads = torch.autograd.grad(
        objective,
        activation["value"],
        retain_graph=True,
    )[0]

    # Remove CLS token
    grads = grads[:, 1:, :]
    grads = grads.reshape(1, 14, 14, 768)
    grads = grads.permute(0, 3, 1, 2)

    # Grad-CAM channel weights
    channel_weights = grads.mean(dim=(2, 3), keepdim=True)

    # Weighted feature maps
    raw_cam = (
        channel_weights * acts
    ).sum(dim=1, keepdim=True)

    raw_maps.append(raw_cam)

    print(
        f"Reference {ref_class}: "
        f"gradient norm={grads.norm().item():.6f}"
    )


handle.remove()


# -----------------------------
# Eq. 8 aggregation
# Average first, then ReLU
# -----------------------------
finer_cam = torch.stack(raw_maps).mean(dim=0)

finer_cam = F.relu(finer_cam)

# Normalize
finer_cam = finer_cam - finer_cam.min()

if finer_cam.max() > 0:
    finer_cam = finer_cam / finer_cam.max()

# Upsample to 224x224
finer_cam = F.interpolate(
    finer_cam,
    size=(224, 224),
    mode="bilinear",
    align_corners=False,
)

finer_cam = finer_cam[0, 0].detach().cpu().numpy()
np.save("results/cub_finer_cam.npy", finer_cam)


# -----------------------------
# Save visualization
# -----------------------------
image_path = Path(
    "data/CUB_200_2011/images/"
    "001.Black_Footed_Albatross/"
    "Black_Footed_Albatross_0010_796097.jpg"
)

# Use the actual image path from the dataset
image_path = next(
    Path("data/CUB_200_2011/images/001.Black_footed_Albatross")
    .glob("*.jpg")
)

original = Image.open(image_path).convert("RGB")
original = original.resize((224, 224))

rgb = np.asarray(original).astype(np.float32) / 255.0

# Simple red heatmap overlay
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(rgb)
plt.imshow(finer_cam, cmap="jet", alpha=0.45)
plt.axis("off")
plt.tight_layout(pad=0)

output_path = Path("results/cub_finer_cam.jpg")
plt.savefig(
    output_path,
    bbox_inches="tight",
    pad_inches=0,
    dpi=150,
)
plt.close()

print("\nFiner-CAM shape:", finer_cam.shape)
print("Finer-CAM min:", finer_cam.min())
print("Finer-CAM max:", finer_cam.max())
print("Saved:", output_path)
