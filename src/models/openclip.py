import torch
import open_clip


class OpenCLIPViTB16:
    def __init__(
        self,
        pretrained="laion400m_e31",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained=pretrained,
            device=self.device,
        )

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def encode_image(self, images):
        return self.model.encode_image(images)
