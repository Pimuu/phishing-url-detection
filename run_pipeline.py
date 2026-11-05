#!/usr/bin/env python3
"""Reproduce every result in results/ (and the data in data/) with one command.

    python run_pipeline.py              # offline: synthetic data, models, figures
    python run_pipeline.py --external   # + recall check on real phishing URLs

Outputs are deterministic for a given --seed.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phishdet.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="random seed for data, splits and models")
    parser.add_argument("--n-per-class", type=int, default=10_000, help="URLs generated per class")
    parser.add_argument("--external", action="store_true",
                        help="also score a sample of real phishing URLs (downloads ~66 MB once)")
    parser.add_argument("--n-external", type=int, default=5000, help="size of the real-URL sample")
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="results directory")
    args = parser.parse_args()
    run(out_dir=args.out, data_dir=ROOT / "data", seed=args.seed, n_per_class=args.n_per_class,
        external=args.external, n_external=args.n_external)


if __name__ == "__main__":
    main()
