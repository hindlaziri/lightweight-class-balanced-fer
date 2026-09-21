# Lightweight Class-Balanced Facial Expression Recognition

Reproducible PyTorch implementation for seven-class facial-expression recognition on FER-2013. The model uses MobileNetV3-Small and an effective-number weighted focal loss with label smoothing. The repository accompanies the TELKOMNIKA manuscript **“Lightweight Class-Balanced Facial Emotion Recognition for Edge Devices.”**

## Authors

**Hind Laziri** — first and corresponding author, ORCID [0009-0004-6518-6712](https://orcid.org/0009-0004-6518-6712), `laziri.h825@ucd.ac.ma`  
**Mohammed Essaid Riffi** — supervising co-author, `saidriffi2@gmail.com`  
Faculty of Sciences, Chouaib Doukkali University, El Jadida, Morocco.

## Main features

- MobileNetV3-Small classifier for 96 × 96 grayscale faces.
- Effective-number class weighting for the imbalanced training distribution.
- Focal modulation for difficult examples.
- Label smoothing for ambiguous or noisy annotations.
- Fixed seed and stratified 70/10/20 split.
- Accuracy, balanced accuracy, macro F1, per-class metrics, confusion matrix, parameter count, and CPU latency.
- Matched cross-entropy baseline.

## Repository structure

```text
.
├── download_data.sh
├── run_experiments.sh
├── requirements.txt
├── LICENSE
├── src/
│   ├── fer_model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── make_figures.py
│   └── inspect_data.py
├── tests/
│   └── test_components.py
├── results/
└── figures/
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The code was tested with Python 3.12, PyTorch 2.14, and torchvision 0.29. CUDA is used automatically when available. Set `--num-workers 0` for the most deterministic CPU execution.

## Dataset

The images are not included in this repository. Download the public FER-2013 mirror with:

```bash
chmod +x download_data.sh
./download_data.sh
```

The resulting `data_train.pt` contains 35,887 examples with JPEG bytes and the labels `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`. The mirror identifies the original source as <https://www.kaggle.com/datasets/msambare/fer2013>. Review the dataset terms before use or redistribution. Loading Python pickle files is unsafe when the source is untrusted; use only the documented file.

## Quick validation

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run a short one-epoch smoke test on 1,000 randomly selected examples:

```bash
python src/train.py --data data_train.pt --out results/smoke \
  --epochs 1 --batch-size 64 --max-samples 1000 \
  --num-workers 0 --no-pretrained
```

## Full experiments

The complete proposed and baseline runs, evaluations, and figures are launched with:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

Equivalent proposed run:

```bash
python src/train.py --data data_train.pt --out results/proposed \
  --epochs 10 --batch-size 256 --image-size 96 --loss cbfocal \
  --label-smoothing 0.05 --gamma 2.0 --seed 42 \
  --num-workers 0 --no-pretrained
```

Matched cross-entropy baseline:

```bash
python src/train.py --data data_train.pt --out results/ce \
  --epochs 10 --batch-size 256 --image-size 96 --loss ce \
  --label-smoothing 0.05 --seed 42 \
  --num-workers 0 --no-pretrained
```

Evaluate a checkpoint and generate the figures:

```bash
python src/evaluate.py --data data_train.pt \
  --checkpoint results/proposed/best.pt \
  --out results/proposed/evaluation --num-workers 0

python src/make_figures.py --data data_train.pt \
  --run results/proposed --out figures
```

## Output files

Training writes `best.pt`, `history.csv`, and `metrics.json`. Evaluation writes `evaluation.json` and `per_class.csv`. Figure generation writes the dataset distribution, learning curve, and confusion matrix when the corresponding artifacts exist.

## Method summary

For class count `n_c`, the effective-number weight is:

```text
w_c = (1 - beta) / (1 - beta^n_c)
```

The smoothed target is:

```text
q = (1 - epsilon)y + epsilon/C
```

The proposed objective is:

```text
L = -sum_c w_c q_c (1 - p_c)^gamma log(p_c)
```

The defaults are `beta=0.9999`, `epsilon=0.05`, and `gamma=2`.

## Responsible use

The model predicts dataset-defined facial-expression categories. It does not establish a person’s internal emotional state. Do not use it for clinical diagnosis, employment screening, education decisions, policing, insurance, or other high-impact decisions. FER-2013 contains annotation uncertainty, demographic limitations, and class imbalance. External, subject-disjoint, subgroup, calibration, and privacy evaluations are required before practical deployment.

## License

The code is released under the MIT License. The FER-2013 dataset is not covered by this code license.
