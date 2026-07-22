"""
Data pipeline for MedMNIST baselines (ResNet / ViT).

Handles channel conversion, normalization, augmentation, label flattening,
class-weight computation, and reproducible seeding.

Run directly to sanity-check the loaders:  python src/data.py
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import medmnist
from medmnist import INFO

# ImageNet statistics. Our ResNet/ViT weights were pretrained on ImageNet,
# so inputs must be normalized the same way those models were trained.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    """Make runs reproducible. Required by the project rubric."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_label(y):
    """MedMNIST returns labels as arrays like [5]; loss functions want int 5."""
    return int(np.asarray(y).flatten()[0])


def build_transforms(dataset_flag, size=224, train=False):
    """
    Build the preprocessing pipeline.

    Augmentation is applied to TRAINING data only -- never to val/test, because
    val/test must stay fixed so scores are comparable across runs.

    Augmentation choices are domain-specific:
      - Dermatoscopy images have no canonical orientation, so flips and
        rotations are physically plausible.
      - Chest X-rays DO have an orientation (heart on the left), so vertical
        flips would create anatomically impossible images. Avoided.
    """
    steps = [transforms.Resize((size, size))]

    if train:
        if dataset_flag == "dermamnist":
            steps += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        else:  # pneumoniamnist
            steps += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]

    steps += [
        transforms.ToTensor(),  # PIL uint8 [0,255] -> float tensor [0,1], CHW
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def get_dataset(dataset_flag, split, size=224, train_aug=False):
    """Load one MedMNIST split with the appropriate transforms."""
    info = INFO[dataset_flag]
    DataClass = getattr(medmnist, info["python_class"])
    return DataClass(
        split=split,
        transform=build_transforms(dataset_flag, size, train=train_aug),
        target_transform=flatten_label,
        download=True,
        size=size,
        as_rgb=True,  # duplicates grayscale to 3 channels (needed for pneumonia)
    )


def get_dataloaders(dataset_flag, size=224, batch_size=32, num_workers=0, seed=42):
    """Return (train_loader, val_loader, test_loader)."""
    set_seed(seed)

    train_ds = get_dataset(dataset_flag, "train", size, train_aug=True)
    val_ds = get_dataset(dataset_flag, "val", size, train_aug=False)
    test_ds = get_dataset(dataset_flag, "test", size, train_aug=False)

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def get_class_weights(dataset_flag, device="cpu"):
    """
    Inverse-frequency weights for the loss:  weight_c = N / (n_classes * count_c)

    Rare classes get larger weights, so the model is penalised more for missing
    them. Without this, a model can score 67% on DermaMNIST by always predicting
    'melanocytic nevi' while never detecting melanoma.
    """
    info = INFO[dataset_flag]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    labels = np.asarray(DataClass(split="train", download=True).labels).flatten()
    counts = np.bincount(labels, minlength=n_classes)
    weights = len(labels) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_num_classes(dataset_flag):
    return len(INFO[dataset_flag]["label"])


if __name__ == "__main__":
    for flag in ["pneumoniamnist", "dermamnist"]:
        print("=" * 60)
        print(flag.upper())
        print("=" * 60)

        train_loader, val_loader, test_loader = get_dataloaders(
            flag, size=224, batch_size=32
        )

        images, labels = next(iter(train_loader))
        print(f"Batch images shape : {tuple(images.shape)}   (B, C, H, W)")
        print(f"Label dtype        : {labels.dtype}")
        print(f"Labels in batch    : {labels.tolist()[:12]} ...")
        print(f"Pixel range        : {images.min():.2f} to {images.max():.2f}")
        print(f"Batches per epoch  : train={len(train_loader)}, "
              f"val={len(val_loader)}, test={len(test_loader)}")

        w = get_class_weights(flag)
        print(f"Class weights      : {[round(x, 3) for x in w.tolist()]}")
        print()