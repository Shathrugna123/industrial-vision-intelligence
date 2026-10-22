
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"

IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

MASK_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.NEAREST,
    ),
    transforms.PILToTensor(),
])


def find_images(folder):
    if not folder.is_dir():
        return []

    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class MVTecDataset(Dataset):
    """
    MVTec AD dataset.

    Training: normal images only.
    Testing: normal and defective images, with masks for defects.
    """

    def __init__(self, category, split="train"):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.category = category
        self.split = split
        self.samples = []

        category_dir = DATASET_ROOT / category
        split_dir = category_dir / split

        if not category_dir.is_dir():
            raise FileNotFoundError(
                f"Category not found: {category_dir}"
            )

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}"
            )

        defect_types = sorted(
            path for path in split_dir.iterdir()
            if path.is_dir()
        )

        for defect_dir in defect_types:
            defect_type = defect_dir.name
            image_paths = find_images(defect_dir)

            for image_path in image_paths:
                is_anomaly = defect_type != "good"
                mask_path = None

                if is_anomaly:
                    mask_path = (
                        category_dir
                        / "ground_truth"
                        / defect_type
                        / f"{image_path.stem}_mask.png"
                    )

                    if not mask_path.is_file():
                        raise FileNotFoundError(
                            f"Mask not found for {image_path}: "
                            f"{mask_path}"
                        )

                self.samples.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "defect_type": defect_type,
                    "label": int(is_anomaly),
                })

        if not self.samples:
            raise ValueError(
                f"No images found for {category}, split={split}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with Image.open(sample["image_path"]) as image:
            image_tensor = IMAGE_TRANSFORM(image.convert("RGB"))

        if sample["mask_path"] is not None:
            with Image.open(sample["mask_path"]) as mask:
                mask_tensor = MASK_TRANSFORM(mask.convert("L"))
                mask_tensor = (mask_tensor > 0).float()
        else:
            mask_tensor = torch.zeros(
                (1, IMAGE_SIZE, IMAGE_SIZE),
                dtype=torch.float32,
            )

        return {
            "image": image_tensor,
            "label": torch.tensor(
                sample["label"], dtype=torch.long
            ),
            "mask": mask_tensor,
            "category": self.category,
            "defect_type": sample["defect_type"],
            "image_path": str(sample["image_path"]),
        }


def main():
    # Smoke test the loader with one product category.
    category = "bottle"

    train_dataset = MVTecDataset(category, split="train")
    test_dataset = MVTecDataset(category, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    print(f"Category: {category}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    print("\nTraining batch:")
    print(f"Images: {train_batch['image'].shape}")
    print(f"Labels: {train_batch['label'].tolist()}")
    print(f"Masks: {train_batch['mask'].shape}")

    print("\nTest batch:")
    print(f"Images: {test_batch['image'].shape}")
    print(f"Labels: {test_batch['label'].tolist()}")
    print(f"Masks: {test_batch['mask'].shape}")
    print(f"Defect types: {test_batch['defect_type']}")

    print("\nMVTec dataset loader test passed.")


if __name__ == "__main__":
    main()
