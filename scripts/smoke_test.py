"""Exercise all four models on a tiny synthetic categorized graph (CPU or CUDA)."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--stopper", choices=["early", "nop"], default="nop")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    # A temporary local dataset avoids downloads and leaves bundled data intact.
    with tempfile.TemporaryDirectory(prefix="smoke_", dir=ROOT / "data") as tmp:
        dataset = Path(tmp)
        triples = [f"e{i}\tr{j}\te{(i+j+1)%12}\n" for i in range(12) for j in range(2)]
        categories = [f"e{i}\tcategory\tc{i%3}\n" for i in range(12)]
        (dataset / "train_cate.txt").write_text("".join(triples + categories))
        (dataset / "valid.txt").write_text("e0\tr0\te3\ne1\tr1\te5\n")
        (dataset / "test.txt").write_text("e2\tr0\te6\ne3\tr1\te8\n")
        for model in ["transe", "rotate", "cs-transe", "cs-rotate"]:
            output = ROOT / "models/smoke" / args.device / model
            cmd = [sys.executable, str(ROOT / "train.py"), "-d", dataset.name,
                   "-m", model, "--device", args.device, "-e", str(args.epochs), "-b", "4",
                   "-ed", "8", "-ced", "4", "-nen", "2", "-nenT", "2",
                   "-eb", "2", "-stop", args.stopper, "--random_seed", "42",
                   "--output-dir", str(output)]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            results = max(output.rglob("results.json"), key=lambda p: p.stat().st_mtime)
            payload = json.loads(results.read_text())
            assert len(payload["losses"]) == args.epochs, payload
            assert all(math.isfinite(x) for x in payload["losses"]), payload
            assert (results.parent / "trained_model.pkl").is_file()
            print(f"PASS {model}: {results}", flush=True)


if __name__ == "__main__":
    main()
