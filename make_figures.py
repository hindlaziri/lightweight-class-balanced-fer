"""Generate figures from a completed run without embedding simulated values."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fer_model import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data_train.pt"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle:
        examples = pickle.load(handle)
    counts = [sum(x["labels"] == name for x in examples) for name in CLASS_NAMES]
    plt.figure(figsize=(6.4, 3.6))
    bars = plt.bar(CLASS_NAMES, counts, color="#2f6690")
    plt.ylabel("Number of images")
    plt.title("FER-2013 class distribution")
    plt.xticks(rotation=25, ha="right")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out / "class_distribution.png", dpi=300)
    plt.close()

    history_path = args.run / "history.csv"
    if history_path.exists():
        with history_path.open() as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(r["epoch"]) for r in rows]
        plt.figure(figsize=(6.4, 3.6))
        plt.plot(epochs, [float(r["train_loss"]) for r in rows], label="Training loss")
        plt.plot(epochs, [float(r["val_loss"]) for r in rows], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(args.out / "learning_curve.png", dpi=300)
        plt.close()

    evaluation_path = args.run / "evaluation.json"
    if not evaluation_path.exists():
        evaluation_path = args.run / "evaluation" / "evaluation.json"
    if evaluation_path.exists():
        result = json.loads(evaluation_path.read_text())
        matrix = np.asarray(result["confusion_matrix"])
        plt.figure(figsize=(5.2, 4.6))
        plt.imshow(matrix, cmap="Blues")
        plt.colorbar(label="Count")
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
        plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion matrix")
        threshold = matrix.max() / 2 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, int(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black", fontsize=8)
        plt.tight_layout()
        plt.savefig(args.out / "confusion_matrix.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
