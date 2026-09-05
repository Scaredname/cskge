"""Run one of the original experiment configurations with optional overrides."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    configs = json.loads((ROOT / "configs/experiments.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(configs))
    args, overrides = parser.parse_known_args()
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    subprocess.run(
        [sys.executable, str(ROOT / "train.py"), *configs[args.experiment], *overrides],
        cwd=ROOT, check=True,
    )


if __name__ == "__main__":
    main()
