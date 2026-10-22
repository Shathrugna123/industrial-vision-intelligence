
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from feature_extractor import DEVICE, IMAGE_SIZE


class MultiLayerFeatureExtractor(nn.Module):
    """
    Experimental ResNet18 feature extractor combining Layer 2 and Layer 3.
    Output: 384 channels at 28 x 28 spatial resolution.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        backbone = models.resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)

        features_layer2 = self.layer2(x)   # [B, 128, 28, 28]
        features_layer3 = self.layer3(features_layer2)  # [B, 256, 14, 14]

        # Upsample Layer 3 spatially, then concatenate with Layer 2.
        features_layer3 = F.interpolate(
            features_layer3,
            size=features_layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        combined = torch.cat(
            [features_layer2, features_layer3],
            dim=1,
        )

        return combined


def main():
    model = MultiLayerFeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    with torch.no_grad():
        features = model(dummy_input)

    print("Multi-layer feature extractor test passed.")
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(features.shape)}")
    print(f"Device: {DEVICE}")
    print(
        "Pretrained parameters frozen:",
        all(not p.requires_grad for p in model.parameters()),
    )


if __name__ == "__main__":
    main()
