"""Run every experiment in order and summarise.

Exit code is the number of experiments with a failing check, so this is usable
as a regression gate.
"""

from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXPERIMENTS = [
    "exp01_linear_base_case",
    "exp02_regrouping_negative_control",
    "exp03_modular_recovery",
    "exp04_perturbation",
    "exp05_cocycle_and_spectra",
    "exp06_partition_lattice",
    "exp07_flat_tangency",
    "exp08_attractor_cocycle",
    "exp09_tier2_nonempty",
    "exp10_behavior_cocycle",
    # exp11, exp12 and exp13 are deliberately NOT registered.
    #
    # exp11/exp12 encode predictions the *dynamics-only* regime refuted, and
    # CLAUDE.md 8 says a result contradicting a theory claim is committed, not
    # tuned away.  Leaving them here would make this gate red forever and train
    # everyone to ignore it.  (Their behavioural readings are also superseded by
    # CLAUDE.md 3.12; both are pinned to `behavior_whiten=False` so they keep
    # reproducing the JSONs on record.)
    #
    # exp13 is excluded on cost, not on outcome: 64 fits at 4000 steps is ~40
    # minutes, far too slow for a gate that is meant to be run often.  Its
    # checks 6-8 are exploratory besides -- a FAIL there is a finding, not a
    # regression.
    #
    # Run all three directly and read results/<name>.json.
]


def main() -> int:
    results = []
    for name in EXPERIMENTS:
        t0 = time.time()
        try:
            runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
            code = 0
        except SystemExit as e:
            code = int(e.code or 0)
        except Exception as exc:  # a crash is a failure, and is reported as one
            print(f"\n!! {name} raised {type(exc).__name__}: {exc}")
            code = 1
        results.append((name, code, time.time() - t0))

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    for name, code, dt in results:
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {name:42s} {dt:6.1f}s")
    n_failed = sum(1 for _, c, _ in results if c != 0)
    print(f"\n  {len(results) - n_failed}/{len(results)} experiments passed all checks")
    print(f"  JSON records in {HERE.parent / 'results'}")
    return n_failed


if __name__ == "__main__":
    raise SystemExit(main())
