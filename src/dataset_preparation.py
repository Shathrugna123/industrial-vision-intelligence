
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

IMAGE_SIZE = 224

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class IndustrialImageDataset(Dataset):
    def __init__(self, category="pcb1", samples_per_class=5):
        self.samples = []

        category_dir = RAW_DIR / category

        class_folders = [
            (category_dir / "Data" / "Images" / "Normal", 0),
            (category_dir / "Data" / "Images" / "Anomaly", 1),
        ]

        for folder, label in class_folders:
            if not folder.exists():
                raise FileNotFoundError(
                    f"Image folder not found: {folder}"
                )

            image_paths = sorted(
                path for path in folder.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )

            for path in image_paths[:samples_per_class]:
                self.samples.append((path, label))

        if not self.samples:
            raise ValueError("No images found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = IMAGE_TRANSFORM(image)

        return image_tensor, torch.tensor(label, dtype=torch.long)


def main():
    dataset = IndustrialImageDataset(
        category="pcb1",
        samples_per_class=5,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    print(f"Total images: {len(dataset)}")

    for batch_index, (images, labels) in enumerate(loader):
        print(f"\nBatch: {batch_index + 1}")
        print(f"Image tensor shape: {images.shape}")
        print(f"Labels: {labels.tolist()}")
        print(f"Tensor dtype: {images.dtype}")

    print("\nDataset preparation test passed.")


if __name__ == "__main__":
    main()
