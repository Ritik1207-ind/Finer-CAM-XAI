import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.linear_classifier import LinearClassifier


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 100
LR = 3e-4


def evaluate(model, features, labels):
    model.eval()

    with torch.no_grad():
        logits = model(features)
        predictions = logits.argmax(dim=1)
        accuracy = (
            (predictions == labels).float().mean().item() * 100
        )

    return accuracy


def main():
    train_data = torch.load(
        "features/cub_train.pt",
        map_location="cpu"
    )

    test_data = torch.load(
        "features/cub_test.pt",
        map_location="cpu"
    )

    train_features = train_data["features"]
    train_labels = train_data["labels"]

    test_features = test_data["features"].to(DEVICE)
    test_labels = test_data["labels"].to(DEVICE)

    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    classifier = LinearClassifier(
        input_dim=512,
        num_classes=200
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LR
    )

    for epoch in range(EPOCHS):
        classifier.train()

        running_loss = 0.0
        total = 0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            total += features.size(0)

        train_loss = running_loss / total

        test_acc = evaluate(
            classifier,
            test_features,
            test_labels
        )

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

    torch.save(
        classifier.state_dict(),
        "checkpoints/cub_clip_linear_classifier.pth"
    )

    print(
        "\nSaved classifier to:"
        "\ncheckpoints/cub_clip_linear_classifier.pth"
    )


if __name__ == "__main__":
    main()
