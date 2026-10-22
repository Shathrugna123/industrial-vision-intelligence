
import torch
import torch.nn as nn
from torchvision import models


class IndustrialVisionModel(nn.Module):
    def __init__(self, pretrained=False, num_classes=2):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.network = models.resnet18(weights=weights)

        input_features = self.network.fc.in_features

        self.network.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(input_features, num_classes),
        )

    def forward(self, images):
        return self.network(images)


def main():
    model = IndustrialVisionModel(pretrained=False)
    model.eval()

    # Simulate a batch of four RGB images.
    sample_images = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        outputs = model(sample_images)

    print("Model: ResNet18")
    print(f"Input shape: {sample_images.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model test passed.")


if __name__ == "__main__":
    main()
