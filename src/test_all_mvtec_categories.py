
from mvtec_dataset import DATASET_ROOT, MVTecDataset

def main():
    categories = sorted(
        path.name for path in DATASET_ROOT.iterdir()
        if path.is_dir()
    )

    total_train = 0
    total_test = 0
    failures = []

    print(f"Testing {len(categories)} MVTec categories\n")

    for category in categories:
        try:
            train_dataset = MVTecDataset(category, split="train")
            test_dataset = MVTecDataset(category, split="test")

            # Load one sample from each split to check image and mask processing.
            train_sample = train_dataset[0]
            test_sample = test_dataset[0]

            assert train_sample["image"].shape == (3, 224, 224)
            assert test_sample["image"].shape == (3, 224, 224)
            assert test_sample["mask"].shape == (1, 224, 224)

            total_train += len(train_dataset)
            total_test += len(test_dataset)

            print(
                f"{category:12} PASS | "
                f"train={len(train_dataset):3} | "
                f"test={len(test_dataset):3}"
            )

        except Exception as exc:
            failures.append((category, str(exc)))
            print(f"{category:12} FAIL | {exc}")

    print("\n" + "-" * 50)
    print(f"Categories tested: {len(categories)}")
    print(f"Total training images: {total_train}")
    print(f"Total test images: {total_test}")
    print(f"Failures: {len(failures)}")

    if failures:
        raise SystemExit(1)

    print("All MVTec category loader tests passed.")


if __name__ == "__main__":
    main()
