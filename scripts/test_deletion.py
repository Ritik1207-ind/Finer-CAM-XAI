import numpy as np
import torch
import torch.nn.functional as F
import open_clip

from src.datasets.cub import CUB200Dataset
from src.models.linear_classifier import LinearClassifier

DEVICE = "cuda"

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion400m_e31",
    device=DEVICE,
)
clip_model.eval()

classifier = LinearClassifier(512, 200).to(DEVICE)
classifier.load_state_dict(
    torch.load(
        "checkpoints/cub_clip_linear_classifier.pth",
        map_location=DEVICE,
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
image = image.unsqueeze(0).to(DEVICE)

target_class = int(label)

with torch.no_grad():
    original_logits = model(image)

original_confidence = torch.softmax(
    original_logits,
    dim=1,
)[0, target_class].item()

# Use already generated CAMs
gradcam = np.load(
    "results/cub_gradcam_baseline.npy"
)

finercam = np.load(
    "results/cub_finer_cam.npy"
)

mean = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073],
    device=DEVICE,
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711],
    device=DEVICE,
).view(1, 3, 1, 1)

normalized_black = -mean / std


def mask_top_percent(image, cam, percent):

    flat = cam.flatten()

    k = int(flat.size * percent)

    if k == 0:
        return image.clone()

    indices = torch.topk(
        torch.from_numpy(flat),
        k=k,
    ).indices.numpy()

    mask = np.zeros(
        flat.shape,
        dtype=bool,
    )

    mask[indices] = True

    mask = torch.from_numpy(
        mask.reshape(cam.shape)
    ).to(DEVICE)

    return torch.where(
        mask.unsqueeze(0).unsqueeze(0),
        normalized_black,
        image,
    )


def deletion_curve(cam):

    fractions = np.arange(
        0.0,
        1.01,
        0.1,
    )

    confidences = []

    for fraction in fractions:

        masked = mask_top_percent(
            image,
            cam,
            float(fraction),
        )

        with torch.no_grad():
            logits = model(masked)

        confidence = torch.softmax(
            logits,
            dim=1,
        )[0, target_class].item()

        confidences.append(confidence)

    auc = np.trapz(
        confidences,
        fractions,
    )

    return fractions, confidences, auc


g_x, g_y, g_auc = deletion_curve(gradcam)
f_x, f_y, f_auc = deletion_curve(finercam)

print("Target class:", target_class)
print(
    "Original confidence:",
    original_confidence,
)

print("\nGrad-CAM deletion curve:")
for x, y in zip(g_x, g_y):
    print(f"  Removed {x:.1f} -> confidence {y:.6f}")

print("Grad-CAM deletion AUC:", g_auc)

print("\nFiner-CAM deletion curve:")
for x, y in zip(f_x, f_y):
    print(f"  Removed {x:.1f} -> confidence {y:.6f}")

print("Finer-CAM deletion AUC:", f_auc)
