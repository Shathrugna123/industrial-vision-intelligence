
import torch
import torch.nn.functional as F
from torchvision import models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NormalizedMultiLayerFeatureExtractor(torch.nn.Module):
    """
    Experimental ResNet18 feature extractor.

    Extracts layer2 and layer3 features, normalizes each layer
    across channels at each spatial location, and concatenates them.
    """

    def __init__(self):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        self.stem = torch.nn.Sequential(
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

        self.to(DEVICE)
        self.eval()

    @torch.no_grad()
    def forward(self, images):
        images = images.to(DEVICE)

        features = self.stem(images)
        features = self.layer1(features)

        layer2_features = self.layer2(features)
        layer3_features = self.layer3(layer2_features)

        layer3_features = F.interpolate(
            layer3_features,
            size=layer2_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Normalize each feature vector independently at each location.
        layer2_features = F.normalize(layer2_features, p=2, dim=1)
        layer3_features = F.normalize(layer3_features, p=2, dim=1)

        combined = torch.cat(
            [layer2_features, layer3_features],
            dim=1,
        )

        return combined


if __name__ == "__main__":
    model = NormalizedMultiLayerFeatureExtractor()
    sample = torch.randn(1, 3, 224, 224).to(DEVICE)

    with torch.no_grad():
        output = model(sample)

    print("Device:", DEVICE)
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
