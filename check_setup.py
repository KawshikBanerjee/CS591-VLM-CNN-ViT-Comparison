"""
Step 1 sanity check.

Verifies that (a) PyTorch sees your GPU, and (b) both MedMNIST datasets
download correctly and have the splits/class balance we expect.

Run:  python check_setup.py
"""

import numpy as np
import torch
import medmnist
from medmnist import INFO


def check_gpu():
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name        : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory      : {total:.1f} GB")
    else:
        print("No GPU found. Training will work but will be slow.")
    print()


def check_dataset(flag, size=224):
    """Download one MedMNIST dataset and report its splits and class balance."""
    info = INFO[flag]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    print("=" * 60)
    print(f"{flag.upper()}  (size={size})")
    print("=" * 60)
    print(f"Task     : {info['task']}")
    print(f"Channels : {info['n_channels']}  (1 = grayscale, 3 = color)")
    print(f"Classes  : {n_classes}")
    for idx, name in info["label"].items():
        print(f"   {idx}: {name}")
    print()

    for split in ["train", "val", "test"]:
        ds = DataClass(split=split, download=True, size=size)
        labels = np.array(ds.labels).flatten()
        counts = np.bincount(labels, minlength=n_classes)
        pct = 100 * counts / counts.sum()

        print(f"  {split:<5} n={len(ds):>6}")
        for idx in range(n_classes):
            bar = "#" * int(pct[idx] / 2)
            print(f"        class {idx}: {counts[idx]:>5}  ({pct[idx]:5.1f}%) {bar}")
        print()

    # Look at one image to confirm shape and pixel range
    img, label = ds[0]
    arr = np.array(img)
    print(f"  Sample image shape : {arr.shape}")
    print(f"  Pixel value range  : {arr.min()} to {arr.max()}")
    print(f"  Sample label       : {label}")
    print()


if __name__ == "__main__":
    check_gpu()
    check_dataset("pneumoniamnist", size=224)
    check_dataset("dermamnist", size=224)
    print("Setup check complete.")