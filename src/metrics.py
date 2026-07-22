"""
Shared evaluation metrics for ALL models (CNN, ViT, and VLM).

Both tracks must import from this file so every model is scored identically.
Do not fork or modify per-model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Score a set of predictions.

    y_true : (N,) integer ground-truth labels
    y_pred : (N,) integer predicted labels
    y_prob : (N, C) predicted probabilities, optional -- needed for AUC
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n_classes = int(max(y_true.max(), y_pred.max())) + 1

    # Macro = every class counts equally, regardless of how rare it is.
    # This is the headline metric for our imbalanced datasets.
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # Weighted = classes count in proportion to size. Reported for context;
    # it is dominated by the majority class and flatters lazy models.
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    # Per-class breakdown: shows WHICH classes the model fails on.
    p_c, r_c, f_c, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "f1_macro": f_mac,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f_w,
        "per_class": {
            "precision": p_c.tolist(),
            "recall": r_c.tolist(),
            "f1": f_c.tolist(),
            "support": support.tolist(),
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))
        ).tolist(),
    }

    # AUC lets us compare against the published MedMNIST leaderboard,
    # which reports AUC alongside accuracy.
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if n_classes == 2:
                results["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                results["auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
        except ValueError:
            results["auc"] = float("nan")

    if class_names is not None:
        results["class_names"] = list(class_names)
    return results


def majority_baseline(y_true):
    """Score of a model that always predicts the most common class.

    Any real model must beat this, or it has learned nothing useful.
    """
    y_true = np.asarray(y_true).flatten()
    majority = np.bincount(y_true).argmax()
    y_pred = np.full_like(y_true, majority)
    return compute_metrics(y_true, y_pred)


def print_metrics(results, title=""):
    """Human-readable summary for the terminal and the lab notebook."""
    if title:
        print(f"\n--- {title} ---")
    print(f"Accuracy           : {results['accuracy']:.4f}")
    print(f"Balanced accuracy  : {results['balanced_accuracy']:.4f}")
    print(f"Precision (macro)  : {results['precision_macro']:.4f}")
    print(f"Recall    (macro)  : {results['recall_macro']:.4f}")
    print(f"F1        (macro)  : {results['f1_macro']:.4f}")
    print(f"F1     (weighted)  : {results['f1_weighted']:.4f}")
    if "auc" in results:
        print(f"AUC                : {results['auc']:.4f}")

    names = results.get("class_names")
    pc = results["per_class"]
    print("\nPer-class:")
    print(f"{'class':<45}{'prec':>7}{'rec':>8}{'f1':>8}{'n':>7}")
    for i in range(len(pc["f1"])):
        label = names[i] if names else str(i)
        print(f"{label[:44]:<45}{pc['precision'][i]:>7.3f}"
              f"{pc['recall'][i]:>8.3f}{pc['f1'][i]:>8.3f}{pc['support'][i]:>7}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    for row in results["confusion_matrix"]:
        print("  " + " ".join(f"{v:>5}" for v in row))



if __name__ == "__main__":
    # Quick self-test: what does a "always guess the most common class" model score?
    # These numbers are the floor our real models must beat.
    import numpy as np
    from medmnist import INFO
    import medmnist

    for flag in ["pneumoniamnist", "dermamnist"]:
        info = INFO[flag]
        DataClass = getattr(medmnist, info["python_class"])
        y_true = np.asarray(DataClass(split="test", download=True).labels).flatten()

        res = majority_baseline(y_true)
        res["class_names"] = [info["label"][str(i)] for i in range(len(info["label"]))]
        print_metrics(res, title=f"{flag} — majority-class baseline (test set)")