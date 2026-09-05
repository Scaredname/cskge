"""Check GPU visibility and execute CUDA forward/backward on each visible GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the diagnostic JSON")
    args = parser.parse_args()
    report = {
        "python": sys.version.split()[0],
        "python_base_prefix": sys.base_prefix,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "gpus": [],
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        report["nvidia_smi"] = (result.stdout + result.stderr).strip()
        report["nvidia_smi_exit_code"] = result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        report["nvidia_smi_error"] = str(error)

    passed = report["cuda_available"]
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(index)),
            "memory_bytes": props.total_memory,
        }
        try:
            x = torch.randn(128, 128, device=f"cuda:{index}", requires_grad=True)
            (x @ x).square().mean().backward()
            torch.cuda.synchronize(index)
            assert torch.isfinite(x.grad).all().item(), "Non-finite CUDA gradient"
            gpu["forward_backward"] = "passed"
            del x
        except (RuntimeError, AssertionError) as error:
            gpu["forward_backward"] = "failed"
            gpu["error"] = str(error)
            passed = False
        report["gpus"].append(gpu)
    report["passed"] = passed
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(output, end="")
    if args.output:
        args.output.write_text(output)
    if not passed:
        print(
            "GPU check failed. Compare this command in a normal host terminal: "
            "sandbox/device isolation can hide a working driver. "
            "If GPUs are visible but kernels fail, check PyTorch architecture support.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
