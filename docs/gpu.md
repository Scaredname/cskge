# GPU execution and troubleshooting

## Diagnosis

The tested host has a working NVIDIA driver, version 580.105.08, and two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition cards (approximately 96 GiB each, compute capability 12.0).

Two separate issues caused the initial failures:

1. **Device isolation:** `nvidia-smi` failed inside the restricted execution sandbox and PyTorch saw no GPUs. Running the same checks outside the sandbox exposed both cards. This was not a host driver failure. GPU commands require an execution environment with device access; a sandbox failure alone does not justify reinstalling the driver.
2. **Unsupported GPU architecture:** the original PyTorch 2.0.0+cu117 build returned `is_available=True` on the host but provided native kernels only through sm_86. Actual CUDA operations failed with `no kernel image is available for execution on the device` and an sm_120 compatibility warning. Neither `nvidia-smi` nor `is_available()` alone proves that training will work.

The project now uses the official **PyTorch 2.7.1+cu128** build, including sm_120 kernels. `pyproject.toml` assigns the official cu128 index specifically to torch; `uv.lock` and the compatibility requirements export were updated accordingly. PyKEEN 1.10.2 and the existing core numerical dependencies were retained. See the [PyTorch 2.7 Blackwell/CUDA 12.8 announcement](https://pytorch.org/blog/pytorch-2-7/) for the version rationale.

The CUDA Version 13.0 shown by `nvidia-smi` describes driver support; `torch.version.cuda == "12.8"` identifies the PyTorch build runtime. This combination passed actual computation checks on the tested host.

## Run from a host terminal

```bash
source ~/.local/bin/env
uv sync --locked
uv run --locked python scripts/check_gpu.py
uv run --locked python scripts/smoke_test.py --device cuda --epochs 10 --stopper early
```

The diagnostic reports Python, PyTorch/CUDA, compiled architectures, visible GPUs, and driver information. It performs matrix multiplication and backpropagation on every visible card and exits with a nonzero status on failure. The recorded result is in `gpu_check.json`.

Select the first or second physical GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python scripts/run_experiment.py yago_new/cs-transe --device cuda
CUDA_VISIBLE_DEVICES=1 uv run --locked python scripts/run_experiment.py yago_new/cs-transe --device cuda
```

Each command uses one GPU. After restricting visibility to the second physical card, that card is renumbered to `cuda:0` inside the process. These presets request 1,000 epochs; use `smoke_test.py` for a quick check.

If the host terminal works but a restricted environment fails, use the host environment. Sandbox GPU restrictions were not modified; GPU validation was performed outside the sandbox.

## Completed validation

- Both cards passed CUDA forward/backward operations and finite-gradient checks.
- All four models completed 10 epochs on a synthetic graph on the first card, including validation evaluation, best-state saving, final testing, and model saving.
- CST completed one epoch over 32,993 YAGO training triples and evaluated all 4,092 test triples on the second card. Final loss was 2.0210639702 and realistic, both-side MRR was 0.0012566088. This small, one-epoch configuration verifies execution only.
- All four models passed two-epoch CPU regression checks under the new dependencies, and `uv pip check` found no dependency conflicts.

Real-data validation command:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --locked python train.py -d yago_new -m cs-transe --device cuda \
  -e 1 -b 128 -ed 8 -ced 4 -nen 2 -nenT 2 -eb 32 -stop nop \
  --output-dir models/validation/gpu1
```

These checks establish GPU training and saving, not full numerical reproduction or multi-GPU training. See `validation.md` for checkpoint-restoration limitations.
