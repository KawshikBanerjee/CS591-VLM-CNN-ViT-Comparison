"""
Training and evaluation for CNN / ViT baselines.

Examples:
    python src/train.py --dataset pneumoniamnist --model resnet18 --epochs 10
    python src/train.py --dataset dermamnist --model resnet18 --epochs 20 --no-aug
    python src/train.py --dataset dermamnist --model resnet18 --epochs 20 --freeze-backbone
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from medmnist import INFO

from data import get_dataloaders, get_class_weights, get_num_classes, set_seed
from metrics import compute_metrics, print_metrics


def build_model(name, num_classes):
    """Load an ImageNet-pretrained backbone and replace its final layer.

    The pretrained model outputs 1000 ImageNet classes; we swap that last
    layer for one sized to our task (2 or 7). Everything before it keeps the
    learned visual features -- this is transfer learning, and it is why we can
    train well on only ~7k medical images.
    """
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def freeze_backbone(model, name):
    """Freeze everything, then unfreeze only the classification head (linear probe)."""
    for p in model.parameters():
        p.requires_grad = False
    if name.startswith("resnet"):
        head = model.fc
    elif name == "vit_b_16":
        head = model.heads.head
    elif name == "efficientnet_b0":
        head = model.classifier[1]
    else:
        raise ValueError(f"Unknown model: {name}")
    for p in head.parameters():
        p.requires_grad = True


@torch.no_grad()
def evaluate(model, loader, device):
    """Run the model over a split and return true labels, predictions, probabilities."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        all_true.append(labels.numpy())
        all_pred.append(probs.argmax(dim=1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())
    return (np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_prob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pneumoniamnist")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="ablation: train without class weighting")
    ap.add_argument("--no-aug", action="store_true",
                    help="ablation: train without data augmentation")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="ablation: train only the final layer (linear probe)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = get_num_classes(args.dataset)
    info = INFO[args.dataset]
    class_names = [info["label"][str(i)] for i in range(n_classes)]

    run_name = f"{args.model}_{args.dataset}_seed{args.seed}"
    if args.no_class_weights:
        run_name += "_noweights"
    if args.no_aug:
        run_name += "_noaug"
    if args.freeze_backbone:
        run_name += "_frozen"

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset, size=args.size, batch_size=args.batch_size,
        seed=args.seed, use_aug=not args.no_aug
    )

    model = build_model(args.model, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.freeze_backbone:
        freeze_backbone(model, args.model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Class-weighted loss: the counter-measure to the imbalance found in Step 1.
    weights = None if args.no_class_weights else get_class_weights(args.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Only optimise parameters that require gradients (matters when frozen).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"Run       : {run_name}")
    print(f"Device    : {device}  |  Params: {n_params:,}  |  Trainable: {trainable:,}")
    print(f"Classes   : {n_classes}  |  Aug: {not args.no_aug}  |  "
          f"Class weights: {'off' if weights is None else 'on'}")

    log_path = f"results/logs/{run_name}.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "val_acc", "val_f1_macro",
                     "val_balanced_acc", "lr", "epoch_seconds"])

    best_f1 = -1.0
    best_epoch = -1
    ckpt_path = f"results/checkpoints/{run_name}.pt"
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device == "cuda")):
                loss = criterion(model(images), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        scheduler.step()
        train_loss = running_loss / seen
        epoch_time = time.time() - t0

        # Model selection happens on VALIDATION only. The test set stays sealed.
        y_true, y_pred, y_prob = evaluate(model, val_loader, device)
        val = compute_metrics(y_true, y_pred, y_prob)

        print(f"epoch {epoch:>2}/{args.epochs}  loss {train_loss:.4f}  "
              f"val_acc {val['accuracy']:.4f}  val_f1 {val['f1_macro']:.4f}  "
              f"({epoch_time:.0f}s)")

        writer.writerow([epoch, round(train_loss, 5), round(val["accuracy"], 5),
                         round(val["f1_macro"], 5),
                         round(val["balanced_accuracy"], 5),
                         optimizer.param_groups[0]["lr"], round(epoch_time, 1)])
        log_file.flush()

        # Keep the checkpoint with the best macro-F1, not the best accuracy.
        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

    log_file.close()
    total_time = time.time() - t_start
    print(f"\nBest epoch {best_epoch} (val macro-F1 {best_f1:.4f}) -> {ckpt_path}")

    # Test set opened exactly once, using the best checkpoint.
    model.load_state_dict(torch.load(ckpt_path))
    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    test = compute_metrics(y_true, y_pred, y_prob, class_names=class_names)
    print_metrics(test, title=f"{run_name} — TEST")

    summary = {
        "run_name": run_name,
        "args": vars(args),
        "n_params": n_params,
        "trainable_params": trainable,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_f1,
        "total_train_seconds": round(total_time, 1),
        "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                         if device == "cuda" else None),
        "test": test,
    }
    with open(f"results/{run_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: results/{run_name}.json")


if __name__ == "__main__":
    main()