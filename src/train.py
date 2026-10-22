
from pathlib import Path
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from model import IndustrialVisionModel
from dataset_preparation import IMAGE_TRANSFORM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = PROJECT_ROOT / "outputs" / "reports" / "dataset_split.json"
MODEL_DIR = PROJECT_ROOT / "models"

BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 0.0001
SEED = 42

DEVICE = torch.device("cpu")


class SplitDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = PROJECT_ROOT / sample["path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = IMAGE_TRANSFORM(image)

        label = int(sample["numeric_label"])

        return image_tensor, torch.tensor(label, dtype=torch.long)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_split_manifest():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset split file not found: {SPLIT_FILE}"
        )

    with SPLIT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0

    return average_loss, accuracy


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    split_data = load_split_manifest()

    train_dataset = SplitDataset(split_data["train"])
    val_dataset = SplitDataset(split_data["validation"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = IndustrialVisionModel(pretrained=False).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    checkpoint_path = MODEL_DIR / "resnet18_best.pth"

    print(f"Device: {DEVICE}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Epochs: {EPOCHS}")
    print("Starting training...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                    "num_classes": 2,
                },
                checkpoint_path,
            )

            print(f"Best checkpoint saved: {checkpoint_path}")

        print("-" * 50)

    print("\nTraining completed.")
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
