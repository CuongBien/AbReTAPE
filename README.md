# AbReTAPE (Absorption-Resistant Cut-Layer Transformation for Privacy in Split Learning)

Repository for studying feature inversion attacks and cut-layer defenses in Split Learning on CIFAR-10.

---

## 🚀 Environment Setup (using `uv`)

This project is configured with [`uv`](https://docs.astral.sh/uv/) for high-speed package management and PyTorch CUDA 13.0 acceleration.

### 1. Synchronize Dependencies & Virtual Environment
```bash
uv sync
```
This automatically sets up `.venv` with:
- PyTorch with CUDA 13 support (`torch==2.14.0+cu130`, `torchvision`, `torchaudio`)
- Scientific packages (`numpy`, `scipy`, `matplotlib`, `scikit-image`, `lpips`, `tqdm`)
- Jupyter Kernel support (`ipykernel`)

### 2. Verify Everything
Run unit & smoke tests across all components:
```bash
uv run run_tests.py
```

---

## 🏃 Running Experiments

All scripts can be executed safely through `uv run`:

### Step 0: Vanilla Split Learning & Centralized Baselines
```bash
# Run Split Learning baseline
uv run run_step0_vanilla.py --epochs 100 --batch-size 128

# Run Centralized baseline for comparison
uv run run_step0_vanilla.py --centralized --epochs 100
```

### Step 1: Feature Inversion Attack (Decoder)
```bash
uv run run_step1_attack.py
```

### Step 2: Channel Permutation Defense & Absorption Evaluation
```bash
uv run run_step2_absorption.py
```

### Step 3: Comparative Baselines (Gaussian Noise, DP-SGD, NoPeek)
```bash
uv run run_step3_baselines.py
```

---

## 📓 Jupyter Notebooks

A custom Jupyter kernel named **`abretape`** (`Python (AbReTAPE)`) has been registered. You can select this kernel when opening any notebook under [`notebooks/`](notebooks/):
- `train_colab.ipynb`
- `step1_colab.ipynb`
- `step2_colab.ipynb`
- `step3_colab.ipynb`
- `b2_colab.ipynb`
- `b3_colab.ipynb`

To start Jupyter Lab or Notebook via `uv`:
```bash
uv run jupyter lab
```
