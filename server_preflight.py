from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    diffcast_root = Path(__file__).resolve().parent
    repo_root = diffcast_root.parents[1]

    print("== Server preflight start ==")
    print(f"DiffCast root: {diffcast_root}")
    print(f"Repo root: {repo_root}")

    print("Checking required python imports...")
    run_checked(
        [
            sys.executable,
            "-c",
            "import torch,accelerate,diffusers,ema_pytorch,einops; print('deps_ok')",
        ]
    )

    print("Checking run.py CLI...")
    run_checked([sys.executable, "run.py", "-h"], cwd=diffcast_root)

    print("Checking SEVIR path placeholder...")
    dataset_file = diffcast_root / "datasets" / "get_datasets.py"
    dataset_text = dataset_file.read_text(encoding="utf-8", errors="ignore")
    if "'sevir' : 'path/to/dataset/sevir'" in dataset_text:
        print("WARNING: DATAPATH['sevir'] still uses placeholder path. Update before training.")
    else:
        print("SEVIR DATAPATH looks customized.")

    print("Running SmaAt + DiffCast contract check...")
    run_checked([sys.executable, str(repo_root / "scripts" / "diffcast_smaat_contract_check.py")])

    print("== Server preflight done ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
