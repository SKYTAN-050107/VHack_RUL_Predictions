# NASA CMAPSS Turbofan RUL (FD001) — Evidence-Driven Notebook Pipeline

This repository contains an end-to-end, evidence-driven workflow for Remaining Useful Life (RUL) prediction on the NASA CMAPSS turbofan engine degradation dataset (FD001 as the primary focus). The workflow is implemented as a sequence of Jupyter notebooks with diagnostics, visual evidence, and reproducible artifacts saved to disk.

## What’s Included

- A complete notebook workflow covering:
  - Data acquisition & integrity checks
  - Cleaning (constant sensor removal, outlier handling)
  - EDA (trend plots, correlation analysis, PCA)
  - Feature engineering (piecewise RUL, rolling stats)
  - Modeling (GroupKFold CV, XGBoost + classical baselines)
  - Deep learning (PyTorch LSTM + Transformer)
  - Final evaluation dashboard (metrics, radar chart, residual plots, significance tests)
- Included CMAPSS data files under `data/` (FD001–FD004) and a preserved `data/x.txt`.
- Reproducible intermediate outputs:
  - `data/processed/*.csv`
  - `artifacts/*.pkl` and `artifacts/*.pth`

## Repository Structure

```
.
├── src/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling_validation.ipynb
│   ├── 05_pytorch_deep_learning.ipynb
│   ├── 06_evaluation_deployment.ipynb
│   └── 07_external_benchmark_analysis.ipynb
├── data/
│   ├── train_FD001.txt, test_FD001.txt, RUL_FD001.txt, ...
│   ├── processed/
│   └── x.txt
├── artifacts/
├── requirements.txt
├── requirement.txt
└── Dockerfile
```

## Quick Start (Local)

### 1) Clone

```bash
git clone <YOUR_REPO_URL>
cd ml
```

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

Notes:
- `requirements.txt` is the primary dependency file.
- `requirement.txt` is kept for compatibility with older setups and the previous Docker build.

### 4) Start JupyterLab

```bash
jupyter lab
```

Open the notebooks in `src/`.

## Quick Start (Docker)

### 1) Build

```bash
docker build -t cmapss-rul .
```

### 2) Run

```bash
docker run --rm -it -p 8888:8888 -v "$PWD":/app cmapss-rul
```

Then open the Jupyter URL printed in the container logs.

## Recommended Execution Order

Run notebooks in order to reproduce the full pipeline:

1. `src/01_data_acquisition.ipynb`
2. `src/02_data_cleaning.ipynb`
3. `src/03_eda.ipynb`
4. `src/04_feature_engineering.ipynb`
5. `src/05_modeling_validation.ipynb`
6. `src/05_pytorch_deep_learning.ipynb`
7. `src/06_evaluation_deployment.ipynb`
8. (Optional) `src/07_external_benchmark_analysis.ipynb`

The later notebooks expect artifacts produced by earlier ones:

- `04_feature_engineering.ipynb` writes:
  - `data/processed/train_labeled.csv`
  - `data/processed/test_labeled.csv`
- `05_modeling_validation.ipynb` writes:
  - `artifacts/scaler.pkl`
  - `artifacts/best_regressor.pkl`
- `05_pytorch_deep_learning.ipynb` writes:
  - `artifacts/lstm_model.pth`
  - `artifacts/transformer_model.pth`

## Reproducibility

This repo aims to be reproducible:
- Random seeds are set inside notebooks (NumPy + PyTorch).
- Dependencies are pinned in `requirements.txt`.
- Intermediate datasets and trained model artifacts are saved with consistent names under `data/processed/` and `artifacts/`.

If you want to rerun from scratch:

```bash
rm -rf data/processed artifacts
mkdir -p data/processed artifacts
```

## Troubleshooting

### LightGBM on macOS (OpenMP)

If LightGBM fails with OpenMP-related errors, install:

```bash
brew install libomp
```

### PyTorch installation

PyTorch is pinned in `requirements.txt`. If you want a GPU build, follow the official PyTorch install selector and replace the `torch==...` pin accordingly.

## Dataset Reference

NASA CMAPSS turbofan engine degradation simulation dataset is commonly referenced via:
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. IEEE PHM. DOI: https://doi.org/10.1109/PHM.2008.4711411

## License

Add your license here (e.g., MIT, Apache-2.0) if you plan to share publicly.

