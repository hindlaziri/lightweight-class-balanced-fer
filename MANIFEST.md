# Repository manifest

| Path | Purpose |
|---|---|
| `src/fer_model.py` | MobileNetV3-Small wrapper, effective-number weighting, focal loss, and label smoothing |
| `src/train.py` | Deterministic data split, augmentation, training loop, checkpoint selection, and aggregate metrics |
| `src/evaluate.py` | Per-class report, confusion matrix, parameter count, and processor-latency measurement |
| `src/make_figures.py` | Dataset distribution, learning-curve, and confusion-matrix figures |
| `src/inspect_data.py` | Inspection utility for the downloaded FER-2013 file |
| `tests/test_components.py` | Unit tests for model output and loss differentiation |
| `download_data.sh` | Download helper for the documented public data mirror |
| `run_experiments.sh` | Matched proposed/baseline training and evaluation pipeline |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes data, checkpoints, environments, bytecode, and logs |
| `LICENSE` | MIT license for the code |
| `CITATION.cff` | Citation metadata for Hind Laziri and Mohammed Essaid Riffi |
| `GITHUB_UPLOAD.md` | Commands for creating and pushing a GitHub repository |

No facial images, dataset archive, trained checkpoint, or private credential is included.
