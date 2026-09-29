import torch
import open_clip

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.datasets.cub import CUB200Dataset
from src.models.linear_classifier import LinearClassifier
from src.models.reshape import reshape_transform


device = "cuda"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion400m_e31",
    device=device
)

model.eval()

for param in model.parameters():
    param.requires_grad = False

classifier = LinearClassifier(
    input_dim=512,
    num_classes=200
).to(device)

checkpoint = torch.load(
    "checkpoints/cub_clip_linear_classifier.pth",
    map_location=device
)

classifier.load_state_dict(checkpoint)
classifier.eval()

class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model, classifier):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = classifier

    def forward(self, x):
        features = self.clip_model.encode_image(x)
        logits = self.classifier(features)
        return logits


combined_model = CLIPClassifier(model, classifier)

dataset = CUB200Dataset(
    "data/CUB_200_2011",
    train=False,
    transform=preprocess
)

image, label = dataset[0]

image = image.unsqueeze(0).to(device)
image.requires_grad_(True)

print("Ground truth class:", label)

with torch.no_grad():
    logits = combined_model(image)
    predicted_class = logits.argmax(dim=1).item()

print("Predicted class:", predicted_class)
print(
    "Predicted confidence:",
    torch.softmax(logits, dim=1)[0, predicted_class].item()
)

target_layer = model.visual.transformer.resblocks[-1].ln_1

cam = GradCAM(
    model=combined_model,
    target_layers=[target_layer],
    reshape_transform=reshape_transform
)

targets = [ClassifierOutputTarget(label)]

grayscale_cam = cam(
    input_tensor=image,
    targets=targets
)

cam_map = grayscale_cam[0]
import numpy as np
np.save("results/cub_gradcam_baseline.npy", cam_map)

print("\nCAM shape:", cam_map.shape)
print("CAM dtype:", cam_map.dtype)
print("CAM min:", cam_map.min())
print("CAM max:", cam_map.max())
print("CAM mean:", cam_map.mean())
