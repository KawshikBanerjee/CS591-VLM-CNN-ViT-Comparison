"""
Collect every run in results/*.json into one comparison table.

Run:  python src/compare.py
"""

import glob
import json
import os

import pandas as pd


def load_runs(results_dir="results"):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if "test" not in r:  # skip incomplete runs
            continue
        a, t = r["args"], r["test"]
        rows.append({
            "model": a["model"],
            "dataset": a["dataset"].replace("mnist", ""),
            "seed": a["seed"],
            "weighted_loss": not a.get("no_class_weights", False),
            "params_M": round(r["n_params"] / 1e6, 1),
            "vram_GB": r.get("peak_vram_gb"),
            "train_min": round(r["total_train_seconds"] / 60, 1),
            "best_ep": r["best_epoch"],
            "acc": round(t["accuracy"], 4),
            "bal_acc": round(t["balanced_accuracy"], 4),
            "prec_M": round(t["precision_macro"], 4),
            "rec_M": round(t["recall_macro"], 4),
            "f1_M": round(t["f1_macro"], 4),
            "auc": round(t["auc"], 4) if "auc" in t else None,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_runs()
    if df.empty:
        print("No completed runs found in results/")
    else:
        df = df.sort_values(["dataset", "model", "seed"])
        pd.set_option("display.width", 200)
        print(df.to_string(index=False))

        out = "results/comparison.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved: {out}")

        # Accuracy per million parameters -- the efficiency story in one number.
        print("\nEfficiency (macro-F1 per million params):")
        eff = df.assign(f1_per_M=(df["f1_M"] / df["params_M"]).round(4))
        print(eff[["model", "dataset", "seed", "f1_M", "params_M",
                   "vram_GB", "train_min", "f1_per_M"]].to_string(index=False))