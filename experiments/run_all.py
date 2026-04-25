"""Run all five experiments end-to-end."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

EXPERIMENTS = [
    "exp1_methods.py",
    "exp2_representation.py",
    "exp3_success_failure.py",
    "exp4_topk.py",
    "exp5_weights.py",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).parent
    sys.path.insert(0, str(here))

    for name in EXPERIMENTS:
        path = here / name
        print(f"\n=== running {name} ===")
        argv_backup = sys.argv[:]
        sys.argv = [str(path)]
        if args.quick:
            sys.argv.append("--quick")
        try:
            runpy.run_path(str(path), run_name="__main__")
        finally:
            sys.argv = argv_backup


if __name__ == "__main__":
    main()
