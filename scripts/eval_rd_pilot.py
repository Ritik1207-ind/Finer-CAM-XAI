import numpy as np
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


def make_finer_and_gradcam(
    model,
    clip_model,
    image,
    target_class,
    similarity,
):
    image = image.clone().detach().requires_grad_(True)

    activation = {}

    target_layer = clip_model.visual.transformer.resblocks[-1].ln_1

    def hook(module, inputs, output):
        activation["value"] = output

    handle = target_layer.register_forward_hook(hook)

    logits = model(image)

    acts = reshape_tokens(activation["value"])

    # -------------------------
    # Grad-CAM
    # -------------------------
    grad = torch.autograd.grad(
        logits[:, target_class].sum(),
        activation["value"],
        retain_graph=True,
    )[0]

    grad = reshape_tokens(grad)

    weights = grad.mean(
        dim=(2, 3),
        keepdim=True,
    )

    gradcam = normalize_cam(
        (weights * acts).sum(
            dim=1,
            keepdim=True,
        )
    )

    # -------------------------
    # Finer-CAM
    # -------------------------
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
        torch.stack(finer_maps).mean(dim=0)
    )

    handle.remove()

    # Most similar classifier-weight class for RD
    reference_indices, _ = get_top_references(
        similarity,
        target_class,
        top_k=1,
    )

    reference_class = int(reference_indices[0])

    return (
        logits.detach(),
        gradcam,
        finer_cam,
        reference_class,
    )


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

    mask = np.zeros(
        flat.shape,
        dtype=bool,
    )

    mask[indices] = True

    mask = torch.from_numpy(
        mask.reshape(cam.shape)
    ).to(DEVICE)

    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    normalized_black = -mean / std

    return torch.where(
        mask.unsqueeze(0).unsqueeze(0),
        normalized_black,
        image,
    )


def get_confidence(logits, cls):
    return torch.softmax(
        logits,
        dim=1,
    )[0, cls]


def relative_drop(
    original_target,
    masked_target,
    original_reference,
    masked_reference,
):
    return (
        (original_target - masked_target)
        -
        (original_reference - masked_reference)
    )


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

indices = np.linspace(
    0,
    len(dataset) - 1,
    N,
    dtype=int,
)

rd5_grad = []
rd5_finer = []
rd10_grad = []
rd10_finer = []

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

    with torch.no_grad():
        original_logits = model(image)

    original_target = get_confidence(
        original_logits,
        target_class,
    )

    (
        _,
        gradcam,
        finercam,
        reference_class,
    ) = make_finer_and_gradcam(
        model,
        clip_model,
        image,
        target_class,
        similarity,
    )

    original_reference = get_confidence(
        original_logits,
        reference_class,
    )

    values = {}

    for percent in [0.05, 0.10]:

        masked_grad = mask_top_percent(
            image,
            gradcam,
            percent,
        )

        masked_finer = mask_top_percent(
            image,
            finercam,
            percent,
        )

        with torch.no_grad():
            grad_logits = model(
                masked_grad
            )

            finer_logits = model(
                masked_finer
            )

        grad_target = get_confidence(
            grad_logits,
            target_class,
        )

        grad_reference = get_confidence(
            grad_logits,
            reference_class,
        )

        finer_target = get_confidence(
            finer_logits,
            target_class,
        )

        finer_reference = get_confidence(
            finer_logits,
            reference_class,
        )

        rd_grad = relative_drop(
            original_target,
            grad_target,
            original_reference,
            grad_reference,
        ).item()

        rd_finer = relative_drop(
            original_target,
            finer_target,
            original_reference,
            finer_reference,
        ).item()

        values[percent] = (
            rd_grad,
            rd_finer,
        )

    rd5_grad.append(
        values[0.05][0]
    )

    rd5_finer.append(
        values[0.05][1]
    )

    rd10_grad.append(
        values[0.10][0]
    )

    rd10_finer.append(
        values[0.10][1]
    )

    print(
        f"[{count:02d}/{N}] "
        f"index={index} "
        f"target={target_class} "
        f"reference={reference_class} "
        f"RD5 Grad={values[0.05][0]:.6f} "
        f"RD5 Finer={values[0.05][1]:.6f} "
        f"RD10 Grad={values[0.10][0]:.6f} "
        f"RD10 Finer={values[0.10][1]:.6f}"
    )

print(
    "\n===== RD PILOT RESULT ====="
)

print(
    "Mean Grad-CAM RD@5%:",
    np.mean(rd5_grad),
)

print(
    "Mean Finer-CAM RD@5%:",
    np.mean(rd5_finer),
)

print(
    "Mean Grad-CAM RD@10%:",
    np.mean(rd10_grad),
)

print(
    "Mean Finer-CAM RD@10%:",
    np.mean(rd10_finer),
)
