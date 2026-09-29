import torch
import torch.nn as nn
##imported required functionalities for the script
from torch.utils.data import DataLoader


from src.datasets.cub import CUB200Dataset
from src.models.openclip import OpenCLIPViTB16
from src.models.linear_classifier import LinearClassifier


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data/CUB_200_2011"

BATCH_SIZE = 16
EPOCHS = 100
LR = 3e-4


def main():
    backbone = OpenCLIPViTB16(
        pretrained="laion400m_e31",
        device=DEVICE
    )

    train_dataset = CUB200Dataset(
        DATA_ROOT,
        train=True,
        transform=backbone.preprocess,
    )

    test_dataset = CUB200Dataset(
        DATA_ROOT,
        train=False,
        transform=backbone.preprocess,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    classifier = LinearClassifier(
        input_dim=512,
        num_classes=200,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LR,
    )

    for epoch in range(EPOCHS):
        classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                features = backbone.encode_image(images)

            logits = classifier(features)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        classifier.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                features = backbone.encode_image(images)
                logits = classifier(features)

                predictions = logits.argmax(dim=1)

                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)

        test_acc = 100.0 * test_correct / test_total

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    torch.save(
        classifier.state_dict(),
        "checkpoints/cub_clip_linear_classifier.pth",
    )

    print("Saved classifier to:")
    print("checkpoints/cub_clip_linear_classifier.pth")


if __name__ == "__main__":
    main()
