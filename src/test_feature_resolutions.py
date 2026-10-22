
from pathlib import Path

import torch
from torchvision import models

from feature_extractor import DEVICE, IMAGE_SIZE

def main():
    weights = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights).to(DEVICE)
    backbone.eval()

    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    with torch.no_grad():
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        print("After stem:", tuple(x.shape))

        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        print("Layer1:", tuple(x.shape))

        x = backbone.layer2(x)
        print("Layer2:", tuple(x.shape))

        x = backbone.layer3(x)
        print("Layer3:", tuple(x.shape))

    print("\nResolution test complete.")
    print("Device:", DEVICE)

if __name__ == "__main__":
    main()
