
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = models.resnet18(weights=weights)

        # Keep intermediate visual features instead of the classifier.
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        # Freeze all pretrained feature-extraction parameters.
        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    def forward(self, images):
        return self.features(images)


def main():
    category = "bottle"
    image_dir = DATASET_ROOT / category / "train" / "good"

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
        }
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No training images found in {image_dir}"
        )

    with Image.open(image_paths[0]) as image:
        image_tensor = IMAGE_TRANSFORM(
            image.convert("RGB")
        ).unsqueeze(0)

    model = FeatureExtractor(pretrained=True).to(DEVICE)

    with torch.no_grad():
        feature_map = model(image_tensor.to(DEVICE))

    print("Feature extractor test passed.")
    print(f"Category: {category}")
    print(f"Input shape: {image_tensor.shape}")
    print(f"Feature map shape: {feature_map.shape}")
    print(f"Device: {DEVICE}")
    print("Pretrained parameters frozen:", all(
        not parameter.requires_grad
        for parameter in model.parameters()
    ))


if __name__ == "__main__":
    main()
