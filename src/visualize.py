"""
Qualitative visualizations for the report:
  - Grad-CAM for ResNet-18 (where the CNN looks)
  - Attention rollout for ViT-B/16 (where the transformer attends)

Produces two kinds of figure on DermaMNIST:
  1. Melanoma vs. nevus: a correctly classified melanoma next to a melanoma
     the model misclassified as nevus.
  2. A gallery: one correctly classified example per class.

Run:
    python src/visualize.py --model resnet18
    python src/visualize.py --model vit_b_16
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from medmnist import INFO

from data import get_dataset, IMAGENET_MEAN, IMAGENET_STD, set_seed
from train import build_model

MEAN = np.array(IMAGENET_MEAN)
STD = np.array(IMAGENET_STD)


def denormalize(tensor):
    """Undo ImageNet normalization for display. (3,H,W) -> (H,W,3) in [0,1]."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def overlay(img, cam, alpha=0.5):
    """Blend a [0,1] heatmap over an RGB image using the 'jet' colormap."""
    heat = cm.jet(cam)[:, :, :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def upsample_to(map2d, size):
    """Bilinearly resize a 2D numpy map to (H, W)."""
    t = torch.tensor(np.ascontiguousarray(map2d), dtype=torch.float32)[None, None]
    out = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


# ------------------------- Grad-CAM (ResNet) -------------------------

def gradcam_resnet(model, input_tensor, target_class, device):
    """Grad-CAM on ResNet-18's last conv block (layer4)."""
    model.eval()
    activations, gradients = {}, {}

    def fwd_hook(_m, _i, o):
        activations["value"] = o.detach()

    def bwd_hook(_m, _gi, go):
        gradients["value"] = go[0].detach()

    target_layer = model.layer4[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x = input_tensor.unsqueeze(0).to(device)
    logits = model(x)
    model.zero_grad()
    logits[0, target_class].backward()

    acts = activations["value"][0]          # (C, h, w)
    grads = gradients["value"][0]           # (C, h, w)
    weights = grads.mean(dim=(1, 2))        # (C,)
    cam = F.relu((weights[:, None, None] * acts).sum(0))  # (h, w)

    h1.remove()
    h2.remove()

    cam = cam.cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return upsample_to(cam, input_tensor.shape[1:])


# ------------------------- Attention rollout (ViT) -------------------------

def attention_rollout_vit(model, input_tensor, device):
    """Attention rollout for torchvision ViT-B/16.

    Hooks only RECORD each block's (normalized) self-attention input during the
    forward pass. Weights are recomputed afterwards, with hooks removed, so the
    recomputation cannot re-trigger the hooks (which previously caused infinite
    recursion). Weights are averaged over heads, combined with the residual
    identity, and multiplied across layers.
    """
    model.eval()
    layer_inputs = []

    def capture_hook(_module, inp, _out):
        layer_inputs.append(inp[0].detach())

    handles = [blk.self_attention.register_forward_hook(capture_hook)
               for blk in model.encoder.layers]

    x = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    attentions = []
    with torch.no_grad():
        for blk, x_in in zip(model.encoder.layers, layer_inputs):
            _, w = blk.self_attention(x_in, x_in, x_in,
                                      need_weights=True,
                                      average_attn_weights=True)
            attentions.append(w)

    n = attentions[0].size(-1)
    eye = torch.eye(n).to(device)
    result = eye.clone()
    for a in attentions:
        a = a[0] + eye
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    mask = result[0, 1:]                      # CLS token -> patch grid
    grid = int(np.sqrt(mask.numel()))
    mask = mask.reshape(grid, grid).cpu().numpy()
    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-8)
    return upsample_to(mask, input_tensor.shape[1:])


# ------------------------- Figure builders -------------------------

def predict(model, tensor, device):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        prob = F.softmax(logits.float(), dim=1)[0]
    return int(prob.argmax()), prob


def make_cam(model, model_name, tensor, target, device):
    if model_name.startswith("resnet"):
        return gradcam_resnet(model, tensor, target, device)
    elif model_name == "vit_b_16":
        return attention_rollout_vit(model, tensor, device)
    else:
        raise ValueError("Visualization supports resnet18 and vit_b_16.")


def melanoma_vs_nevus(model, model_name, dataset, class_names, device, out):
    """Correct melanoma vs. melanoma misclassified as nevus."""
    MEL, NEV = 4, 5
    correct, missed = None, None
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label != MEL:
            continue
        pred, _ = predict(model, img, device)
        if pred == MEL and correct is None:
            correct = (img, i)
        elif pred == NEV and missed is None:
            missed = (img, i)
        if correct and missed:
            break

    if not (correct and missed):
        print("Could not find both a correct and a missed melanoma; skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for row, (tensor, _idx), title in [
        (0, correct, "Melanoma - correctly classified"),
        (1, missed, "Melanoma - misclassified as nevus"),
    ]:
        img = denormalize(tensor)
        cammap = make_cam(model, model_name, tensor, MEL, device)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(title, fontsize=10)
        axes[row, 1].imshow(overlay(img, cammap))
        axes[row, 1].set_title("Model attention", fontsize=10)
        for c in (0, 1):
            axes[row, c].axis("off")

    method = "Grad-CAM" if model_name.startswith("resnet") else "Attention rollout"
    fig.suptitle(f"{model_name}: {method} on melanoma vs. nevus", fontsize=12)
    fig.tight_layout()
    path = f"{out}/{model_name}_melanoma_vs_nevus.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def class_gallery(model, model_name, dataset, class_names, device, out):
    """One correctly classified example per class, with its attention map."""
    n = len(class_names)
    found = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label in found:
            continue
        pred, _ = predict(model, img, device)
        if pred == label:
            found[label] = img
        if len(found) == n:
            break

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for c in range(n):
        if c not in found:
            for r in (0, 1):
                axes[r, c].axis("off")
            continue
        tensor = found[c]
        img = denormalize(tensor)
        cammap = make_cam(model, model_name, tensor, c, device)
        axes[0, c].imshow(img)
        axes[0, c].set_title(class_names[c][:16], fontsize=8)
        axes[1, c].imshow(overlay(img, cammap))
        for r in (0, 1):
            axes[r, c].axis("off")

    method = "Grad-CAM" if model_name.startswith("resnet") else "Attention rollout"
    fig.suptitle(f"{model_name}: {method}, one correct example per class", fontsize=12)
    fig.tight_layout()
    path = f"{out}/{model_name}_class_gallery.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--dataset", default="dermamnist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    info = INFO[args.dataset]
    n_classes = len(info["label"])
    class_names = [info["label"][str(i)] for i in range(n_classes)]

    ckpt = f"results/checkpoints/{args.model}_{args.dataset}_seed{args.seed}.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Train it first, e.g. python src/train.py --dataset {args.dataset} "
            f"--model {args.model} --epochs 20"
        )

    model = build_model(args.model, n_classes).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    out = "results/figures"
    os.makedirs(out, exist_ok=True)

    dataset = get_dataset(args.dataset, "test", size=args.size, train_aug=False)

    if args.dataset == "dermamnist":
        melanoma_vs_nevus(model, args.model, dataset, class_names, device, out)
    class_gallery(model, args.model, dataset, class_names, device, out)


if __name__ == "__main__":
    main()