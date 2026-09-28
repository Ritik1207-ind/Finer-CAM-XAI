import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cub import CUB200Dataset
from src.models.openclip import OpenCLIPViTB16


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data/CUB_200_2011"
BATCH_SIZE = 16


def extract(split_name, train):
    backbone = OpenCLIPViTB16(
        pretrained="laion400m_e31",
        device=DEVICE,
    )

    dataset = CUB200Dataset(
        DATA_ROOT,
        train=train,
        transform=backbone.preprocess,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    all_features = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Extracting {split_name}"):
            images = images.to(DEVICE, non_blocking=True)

            features = backbone.encode_image(images)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    torch.save(
        {
            "features": features,
            "labels": labels,
        },
        f"features/cub_{split_name}.pt",
    )

    print(f"Saved: features/cub_{split_name}.pt")
    print("Features:", features.shape)
    print("Labels:", labels.shape)


def main():
    extract("train", train=True)
    extract("test", train=False)


if __name__ == "__main__":
    main()
