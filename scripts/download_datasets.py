"""Download ACE + MSC, normalize to Episode JSONL.

Output: data/processed/episodes.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from episodic.loaders import ace, msc
from episodic.schema import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/episodes.jsonl")
    ap.add_argument("--max-ace", type=int, default=None)
    ap.add_argument("--max-msc", type=int, default=None)
    ap.add_argument("--no-synthetic", action="store_true",
                    help="If set, do not fall back to synthetic data when HF fails.")
    args = ap.parse_args()

    allow_syn = not args.no_synthetic

    print("[ace] loading ...")
    ace_eps = ace.load(max_records=args.max_ace, allow_synthetic=allow_syn)
    print(f"[ace] {len(ace_eps)} episodes")

    print("[msc] loading ...")
    msc_eps = msc.load(max_records=args.max_msc, allow_synthetic=allow_syn)
    print(f"[msc] {len(msc_eps)} episodes")

    out = Path(args.out)
    n = write_jsonl(ace_eps + msc_eps, out)
    print(f"wrote {n} episodes -> {out}")


if __name__ == "__main__":
    main()
