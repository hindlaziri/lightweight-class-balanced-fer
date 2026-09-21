"""Train MobileNetV3 FER models on the public FER-2013 pickle/pt file."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from fer_model import CLASS_NAMES, MobileNetV3FER, make_loss


class FERBytes(Dataset):
    def __init__(self, examples, train: bool, image_size: int = 96):
        self.examples = examples
        self.labels = [CLASS_NAMES.index(x["labels"]) for x in examples]
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        image = Image.open(__import__("io").BytesIO(row["img_bytes"])).convert("L")
        return self.transform(image), self.labels[index]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_examples(examples, seed: int, val_fraction: float, test_fraction: float):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(examples))
    # The public mirror stores the original FER-2013 examples in label blocks;
    # stratified splitting avoids leaking an image while preserving class ratios.
    train_idx, val_idx, test_idx = [], [], []
    labels = np.asarray([CLASS_NAMES.index(x["labels"]) for x in examples])
    for label in range(len(CLASS_NAMES)):
        group = indices[labels == label]
        rng.shuffle(group)
        n_test = max(1, int(round(len(group) * test_fraction)))
        n_val = max(1, int(round(len(group) * val_fraction)))
        test_idx.extend(group[:n_test])
        val_idx.extend(group[n_test:n_test + n_val])
        train_idx.extend(group[n_test + n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def evaluate(model, loader, device):
    model.eval()
    ys, ps, total_loss = [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            ys.extend(labels.cpu().numpy().tolist())
            ps.extend(logits.argmax(1).cpu().numpy().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(ys, ps),
        "balanced_accuracy": balanced_accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro"),
        "n": len(ys),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data_train.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/run"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["ce", "cbfocal"], default="cbfocal")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    with args.data.open("rb") as handle:
        examples = pickle.load(handle)
    if args.max_samples:
        sample_rng = np.random.default_rng(args.seed)
        chosen = sample_rng.choice(len(examples), size=min(args.max_samples, len(examples)), replace=False)
        examples = [examples[int(i)] for i in chosen]

    tr_idx, va_idx, te_idx = split_examples(examples, args.seed, 0.10, 0.20)
    train_ds = FERBytes(examples, True, args.image_size)
    eval_ds = FERBytes(examples, False, args.image_size)
    train_set, val_set, test_set = Subset(train_ds, tr_idx), Subset(eval_ds, va_idx), Subset(eval_ds, te_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    counts = np.bincount([train_ds.labels[i] for i in tr_idx], minlength=len(CLASS_NAMES)).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3FER(pretrained=not args.no_pretrained).to(device)
    criterion = make_loss(args.loss, counts, args.label_smoothing, args.gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_f1 = -1.0
    best_path = args.out / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * labels.size(0)
        scheduler.step()
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / len(train_loader.dataset), "lr": scheduler.get_last_lr()[0], "seconds": time.perf_counter() - start, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            serial_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            torch.save({"model": model.state_dict(), "args": serial_args, "class_counts": counts, "class_names": CLASS_NAMES, "epoch": epoch}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader, device)
    summary = {"device": str(device), "seed": args.seed, "epochs": args.epochs, "loss": args.loss, "label_smoothing": args.label_smoothing, "gamma": args.gamma, "class_counts_train": counts, "split_sizes": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)}, "best_epoch": checkpoint["epoch"], "test": test}
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (args.out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
