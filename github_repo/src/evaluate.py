"""Evaluate a trained checkpoint and export publication-ready metrics."""
from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from fer_model import CLASS_NAMES, MobileNetV3FER
from train import split_examples


class EvalFER(Dataset):
    def __init__(self, examples, image_size=96):
        self.examples = examples
        self.labels = [CLASS_NAMES.index(x["labels"]) for x in examples]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        image = Image.open(io.BytesIO(row["img_bytes"])).convert("L")
        return self.transform(image), self.labels[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data_train.pt"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/eval"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle:
        examples = pickle.load(handle)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = checkpoint.get("args", {}).get("image_size", 96)
    _, _, test_idx = split_examples(examples, args.seed, 0.10, 0.20)
    dataset = EvalFER(examples, image_size)
    loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = MobileNetV3FER(pretrained=False).cpu()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    y_true, y_pred, probabilities = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(1).numpy().tolist())
            probabilities.extend(logits.softmax(1).numpy().tolist())
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    batch = torch.zeros(1, 1, image_size, image_size)
    for _ in range(10):
        _ = model(batch)
    start = time.perf_counter()
    repeats = 100
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(batch)
    latency_ms = (time.perf_counter() - start) * 1000.0 / repeats
    result = {
        "checkpoint": str(args.checkpoint),
        "n_test": len(y_true),
        "classification_report": report,
        "confusion_matrix": matrix,
        "latency_ms_per_image_cpu_batch1": latency_ms,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (args.out / "evaluation.json").write_text(json.dumps(result, indent=2))
    with (args.out / "per_class.csv").open("w", newline="") as handle:
        keys = ["class", "precision", "recall", "f1-score", "support"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for name in CLASS_NAMES:
            row = {"class": name, **report[name]}
            writer.writerow(row)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
