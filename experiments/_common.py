"""Shared plumbing for experiment scripts.

Every experiment writes a JSON record containing its seed and every parameter
(CLAUDE.md §8).  The JSON is the artifact; figures, if any, are secondary.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    if isinstance(o, (set, frozenset)):
        return sorted(_jsonable(x) for x in o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    payload = {"experiment": name, "utc": datetime.now(timezone.utc).isoformat(), **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
    return path


def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def verdict(ok: bool, msg: str) -> str:
    """Uniform pass/fail line.  Failures are reported, never tuned away (§8)."""
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {msg}")
    return tag
