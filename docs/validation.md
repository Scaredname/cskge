# Environment and validation record

This record describes local checks performed on 2026-09-05. See `environment.json` for hardware/software details and `data_inventory.json` for input file line counts.

## Current environment

- Project `.venv`: uv-managed CPython 3.11.16, PyTorch 2.7.1+cu128, and PyKEEN 1.10.2.
- NumPy 1.26.4, SciPy 1.10.1, pandas 2.0.0, and class_resolver 0.5.2.
- `pyproject.toml` declares dependencies, `uv.lock` pins resolved versions, and `.python-version` pins Python. `requirements.txt` is a compatibility export only.
- `uv pip check` passed with no dependency conflicts.
- Host driver 580.105.08 works with two RTX PRO 6000 Blackwell cards. The CUDA 12.8 build has been verified. Earlier `nvidia-smi` failures occurred inside a restricted sandbox and did not indicate a host driver failure. See `gpu.md` and `gpu_check.json`.
- The default PyKEEN cache is `.cache/pykeen`; override it with `PYKEEN_HOME` if needed.

## Initial validation before the GPU upgrade

| Check | Result | Coverage |
| --- | --- | --- |
| `python train.py --help` | Passed | All custom modules import successfully |
| `python scripts/smoke_test.py` | All four models passed | Synthetic categorized graph, two CPU epochs, stopping disabled; training, testing, and saving |
| `python scripts/smoke_test.py --epochs 10 --stopper early` | All four models passed | Validation at epoch 10, RLRP/stopping callbacks, best-weight saving, and final testing |
| CST on real `yago_new` | Passed | 32,993 training triples, 5,744 entities, 33 relations; one training epoch and evaluation of 4,092 test triples |
| Built-in Nations with TransE | Passed | One CPU epoch, evaluation, and saving; covers the previously undefined-variable branch |
| `compileall`, `bash -n`, `git diff --check` | Passed | Python syntax, setup-script syntax, and patch whitespace |

Initial real-data validation command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python train.py \
  -d yago_new -m cs-transe --device cpu -e 1 -b 128 \
  -ed 8 -ced 4 -nen 2 -nenT 2 -eb 8 -stop nop \
  --output-dir models/validation
```

This run produced loss approximately 2.00958 and realistic, both-side test MRR approximately 0.00133260. These low-dimensional, one-epoch results establish workflow execution, not paper-level or best-preset performance. Artifacts are under `models/validation/yago_new/cs-transe/`.

All initial ten-epoch synthetic runs produced finite final losses, `results.json`, and `trained_model.pkl`. Artifacts are under `models/smoke/`. Temporary synthetic input directories are removed when the script exits; bundled data is preserved.

## uv migration

The first environment used conda-provided Python 3.11.4 to create a venv. It was replaced by uv 0.12.10 with independently downloaded CPython 3.11.16. The initial ten-epoch stopping checks and CPU YAGO run predate that migration. All four models passed two-epoch CPU smoke tests after the migration.

A Linux-wheel metadata override was initially needed for PyTorch 2.0.0. It was removed when the GPU fix upgraded torch to 2.7.1+cu128. setuptools 65.5.0 remains pinned because the old PyKEEN code imports `pkg_resources`.

Use `uv run --locked python ...` for execution, `uv add`/`uv lock` to update dependencies, and `uv sync --locked` to synchronize the environment. Generate the compatibility requirements file with:

```bash
uv export --locked --format requirements-txt --no-hashes --no-header --output-file requirements.txt
```

There is no separately maintained requirements lockfile.

## Validation after the GPU fix

- `uv run --locked python scripts/check_gpu.py`: both GPUs passed matrix multiplication, backpropagation, and finite-gradient checks outside the sandbox. The build includes sm_120 kernels.
- `uv run --locked python scripts/smoke_test.py --device cuda --epochs 10 --stopper early`: all four models passed, including epoch-10 validation, best-state saving, and final testing. Artifacts are under `models/smoke/cuda/`.
- Real YAGO/CST training for one epoch and evaluation of all 4,092 test triples passed on the second physical card. Artifacts are under `models/validation/gpu1/`; numerical details are in `gpu.md`.
- All four models passed two-epoch CPU regression tests under the new PyTorch version. Artifacts are under `models/smoke/cpu/`.

## Validation boundaries

The 16 original 1,000-epoch experiments have not been run. Exact numerical reproduction, full checkpoint continuation, independent category-factory deserialization, multi-GPU training, multiple data-loader workers, and candidate-entity subset evaluation remain unverified. NELL, FB, and DB files were extracted and counted, but not used in full training runs. Ten-epoch tests exercise stopping evaluation and saving, not termination after patience is exhausted.

As documented in the [PyTorch serialization notes](https://docs.pytorch.org/docs/2.7/notes/serialization.html), `torch.load` defaults to `weights_only=True` starting with 2.6. Restoring old PyKEEN full checkpoints containing NumPy RNG state requires separate compatibility work. No global loading-policy override was introduced.

Compatibility fixes, particularly the stage-III negative-score reshape, may change historical results. Original presets are retained in `configs/experiments.json` and `original_experiments.md`; original source is available through Git history. See section 7 of `code_guide.md` for algorithm-review concerns.
