#!/usr/bin/env bash
set -euo pipefail
URL="https://huggingface.co/datasets/Jeneral/fer2013/resolve/main/train.pt"
OUT="${1:-data_train.pt}"
echo "Downloading FER-2013 mirror to ${OUT}"
curl -L --fail --retry 3 --progress-bar -o "${OUT}.part" "$URL"
mv "${OUT}.part" "$OUT"
echo "Saved $(du -h "$OUT" | cut -f1) to $OUT"
echo "Review the source dataset terms before use or redistribution."
