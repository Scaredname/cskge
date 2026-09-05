# CSKGE: Category-Supplemented Knowledge Graph Embeddings

Experiments with TransE, RotatE, CS-TransE (CST), and CS-RotatE (CSR), including four bundled datasets.

## Quick start

The project uses **uv to manage Python, the virtual environment, and dependencies**. It does not depend on conda. Python is pinned to 3.11.16; `pyproject.toml` declares dependencies and `uv.lock` records resolved versions. See the [uv project guide](https://docs.astral.sh/uv/guides/projects/) for the workflow.

On the current host, uv is installed. If the current shell cannot find it, run `source ~/.local/bin/env`. No environment activation is required:

```bash
uv run --locked python scripts/smoke_test.py
```

On a new host, [install uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
bash scripts/setup_env.sh
```

The setup script downloads an independent Python interpreter, synchronizes `.venv` from the lockfile, extracts missing data files, and checks dependencies. The locked environment targets Linux x86_64 and Python 3.11. Use `uv sync --locked` to synchronize it manually.

The original PyKEEN development-version dependency was replaced with the tested `1.10.2` release. Numerical equivalence with the original development version has not been established. See the [PyKEEN release page](https://pypi.org/project/pykeen/1.10.2/).

## Run experiments

Check the real-data workflow with a small configuration:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python train.py \
  -d yago_new -m cs-transe --device cpu -e 1 -b 128 \
  -ed 8 -ced 4 -nen 2 -nenT 2 -eb 8 -stop nop
```

Run an original experiment preset. All 16 configurations are preserved in `configs/experiments.json`:

```bash
uv run --locked python scripts/run_experiment.py yago_new/cs-transe --device cuda --random_seed 42
uv run --locked python scripts/run_experiment.py --help
uv run --locked python train.py --help
```

Trailing arguments override preset values, for example `-e 10 -ed 32 -b 64`. The default `--device auto` selects CUDA when available, otherwise CPU. GPU execution has been verified on a host with two RTX PRO 6000 Blackwell cards using PyTorch 2.7.1 and CUDA 12.8. A restricted sandbox may hide these devices; run GPU commands in a host terminal with device access.

The original presets use 1,000 epochs and large negative-sampling counts. The complete experiment matrix has not been reproduced.

## Check and select GPUs

```bash
uv run --locked python scripts/check_gpu.py
uv run --locked python scripts/smoke_test.py --device cuda --epochs 10 --stopper early
CUDA_VISIBLE_DEVICES=1 uv run --locked python scripts/run_experiment.py yago_new/cs-transe --device cuda
```

`CUDA_VISIBLE_DEVICES=1` selects the second physical GPU, which appears as `cuda:0` inside the process. Each training run uses one GPU. PyTorch is installed from the official CUDA 12.8 index to support Blackwell. The tested host driver did not require reinstallation. See [GPU troubleshooting](docs/gpu.md) for the diagnosis and sandbox limitations.

## Repository layout

| Path | Purpose |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies, resolved versions, and managed Python version |
| `train.py` | Arguments, model/training assembly, execution, and result saving |
| `utilities.py` | Dataset loading and category-triple separation |
| `customize/` | Models, three-stage training, sampling, stopping, and pipeline extensions |
| `configs/experiments.json` | All original experiment presets |
| `scripts/` | Setup, extraction, experiment launchers, and smoke tests |
| `docs/` | Code guide, environment records, data inventory, and validation reports |
| `data.zip`, `data/` | Original archive and extracted datasets |
| `models/` | Experiment outputs organized by dataset, model, and timestamp |
| `.venv/`, `.cache/` | Local environment and caches, excluded from Git |

Each result directory contains `results.json` (losses and ranking metrics), `trained_model.pkl`, `training_triples/`, `metadata.json`, `config.json`, and `args.json` (actual command arguments). Validation artifacts are stored under `models/smoke/` and `models/validation/`.

Further reading: [code guide](docs/code_guide.md), [validation and limitations](docs/validation.md), and [original experiment commands](docs/original_experiments.md).
