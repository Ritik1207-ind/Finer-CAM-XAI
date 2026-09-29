import numpy as npp
import torch
import torch.nn.functional as F
import open_clip

from src.datasets.cub import CUB200Dataset
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


def generate_cams(
    model,
    clip_model,
    image,
    target_class,
    similarity,
):
    image = image.clone().detach().requires_grad_(True)

    activation = {}

    target_layer = clip_model.visual.transformer.resblocks[-1].ln_1

    def hook(module, inpputs, output):
        activation["value"] = output

    handle = target_layer.register_forward_hook(hook)

    logits = model(image)

    acts = reshape_tokens(
        activation["value"]
    )

    # Grad-CAM
    grad = torch.autograd.grad(
        logits[:, target_class].sum(),
        activation["value"],
        retain_graph=True,
    )[0]

    grad = reshape_tokens(grad)

    grad_weights = grad.mean(
        dim=(2, 3),
        keepdim=True,
    )

    gradcam = normalize_cam(
        (grad_weights * acts).sum(
            dim=1,
            keepdim=True,
        )
    )

    # Finer-CAM
    reference_indices, _ = get_top_references(
        similarity,
        target_class,
        top_k=3,
    )

    finer_maps = []

    for ref_class in reference_indices.tolist():

        objective = (
            logits[:, target_class]
            - GAMMA * logits[:, ref_class]
        ).sum()

        ref_grad = torch.autograd.grad(
            objective,
            activation["value"],
            retain_graph=True,
        )[0]

        ref_grad = reshape_tokens(ref_grad)

        ref_weights = ref_grad.mean(
            dim=(2, 3),
            keepdim=True,
        )

        raw_map = (
            ref_weights * acts
        ).sum(
            dim=1,
            keepdim=True,
        )

        finer_maps.append(raw_map)

    finer_cam = normalize_cam(
        torch.stack(finer_maps).mean(
            dim=0
        )
    )

    handle.remove()

    return gradcam, finer_cam


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

    k = max(
        1,
        int(flat.size * percent),
    )

    indices = torch.topk(
        torch.from_numpy(flat),
        k=k,
    ).indices.numpy()

    mask = npp.zeros(
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


def deletion_auc(model, image, target_class, cam):
    fractions = npp.arange(
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

    auc = npp.trapz(
        confidences,
        fractions,
    )

    return auc


print("Loading model...")

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion400m_e31",
    device=DEVICE,
)

clip_model.eval()

classifier = LinearClassifier(
    512,
    200,
).to(DEVICE)

classifier.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )
)

classifier.eval()

model = CLIPClassifier(
    clip_model,
    classifier,
)

dataset = CUB200Dataset(
    ROOT,
    train=False,
    transform=preprocess,
)

weights = load_classifier_weights(
    CHECKPOINT
)

similarity = compute_similarity(
    weights
)

indices = npp.linspace(
    0,
    len(dataset) - 1,
    N,
    dtype=int,
)

grad_aucs = []
finer_aucs = []

print(
    f"\nEvaluating {N} diverse test images...\n"
)

for count, index in enumerate(
    indices,
    1,
):
    image, label = dataset[index]

    image = image.unsqueeze(0).to(
        DEVICE
    )

    target_class = int(label)

    gradcam, finercam = generate_cams(
        model,
        clip_model,
        image,
        target_class,
        similarity,
    )

    grad_auc = deletion_auc(
        model,
        image,
        target_class,
        gradcam,
    )

    finer_auc = deletion_auc(
        model,
        image,
        target_class,
        finercam,
    )

    grad_aucs.append(grad_auc)
    finer_aucs.append(finer_auc)

    print(
        f"[{count:02d}/{N}] "
        f"index={index} "
        f"target={target_class} "
        f"Grad-AUC={grad_auc:.6f} "
        f"Finer-AUC={finer_auc:.6f}"
    )


print(
    "\n===== DELETION PILOT RESULT ====="
)

print(
    "Mean Grad-CAM deletion AUC:",
    npp.mean(grad_aucs),
)

print(
    "Mean Finer-CAM deletion AUC:",
    npp.mean(finer_aucs),
)
