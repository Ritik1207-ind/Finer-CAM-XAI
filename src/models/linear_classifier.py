import torch
import torch.nn as nnu


class LinearClassifier(nnu.Module):
    def __init__(self, input_dim=512, num_classes=200):
        super().__init__()
        self.fc = nnu.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
