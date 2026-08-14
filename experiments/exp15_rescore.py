"""Re-score exp15's saved fingerprints offline -- no refitting.

CLAUDE.md §3.13: "a matching or scoring rule can be re-evaluated offline.
Twenty-five fits is half an hour; a criterion is a one-line change, and the two
should never have been coupled."  exp15 dumps every fingerprint of every arm
into its JSON, so any tolerance, screen or matching rule can be swept here in
seconds.

Run with no arguments to print the sweep; the numbers it emits are the ones a
claim about tolerance-sensitivity should quote.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idyn import metrics as M                # noqa: E402
from idyn import spectra as SP               # noqa: E402

JSON = Path(__file__).resolve().parents[1] / "results" / "exp15_nlb.json"


def load_fp(d: dict) -> M.DynamicalFingerprint:
    return M.DynamicalFingerprint(
        partition=list(d["partition"]),
        spectra=[np.asarray(s, dtype=float) for s in d["spectra"]],
        rotations=list(d["rotations"]),
        coherences=list(d["coherences"]),
    )


def tier1(fp: M.DynamicalFingerprint) -> np.ndarray:
    return np.sort(np.concatenate([np.asarray(s).ravel() for s in fp.spectra]))


def score(pairs, spec_tol: float, rot_tol: float, screen: bool) -> dict:
    rot, spec, agr, t1, lat = [], [], [], [], []
    for fa_list, fb_list in pairs:
        for fa in fa_list:
            for fb in fb_list:
                if screen and (fa.duplicate_modules(spec_tol, rot_tol)
                               or fb.duplicate_modules(spec_tol, rot_tol)):
                    continue
                r = M.invariant_agreement(fa, fb, spec_tol=spec_tol, rot_tol=rot_tol)
                rot.append(r.rotation_error)
                spec.append(r.spectrum_error)
                agr.append(bool(r.agree))
                t1.append(float(np.abs(tier1(fa) - tier1(fb)).max()))
                lat.append(SP.rotation_lattice_margin(
                    [abs(x) for x in fa.rotations], [abs(x) for x in fb.rotations])[0])
    if not rot:
        return {"n": 0}
    return {
        "n": len(rot),
        "tier1": float(np.median(t1)),
        "rot": float(np.median(rot)),
        "spec": float(np.median(spec)),
        "lattice": float(np.median(lat)),
        "frac_agree": float(np.mean(agr)),
    }


def main() -> int:
    if not JSON.exists():
        print(f"no {JSON}; run exp15_nlb.py first")
        return 1
    rec = json.loads(JSON.read_text(encoding="utf-8"))

    treat = [
        ([load_fp(f) for f in blk["half_a"]], [load_fp(f) for f in blk["half_b"]])
        for blk in rec["part3_treatment"]["fingerprints"]
    ]
    arms = {"treatment": treat}
    for name, blk in rec.get("part4_controls", {}).items():
        arms[name] = [
            (treat[i][0], [load_fp(f) for f in fps])
            for i, fps in enumerate(blk["fingerprints_b"])
        ]

    print("Sensitivity of every readout to the tolerances (no refitting).")
    print("The point of the sweep: a conclusion that only holds at one tolerance")
    print("is a conclusion about the tolerance.\n")
    for screen in (False, True):
        print(f"--- duplicate-module screen: {screen} ---")
        hdr = f"{'arm':16} {'spec_tol':>9} {'rot_tol':>8} {'n':>4} " \
              f"{'TIER1':>9} {'rot':>9} {'spec':>9} {'lattice':>9} {'agree':>7}"
        print(hdr)
        for spec_tol, rot_tol in ((0.02, 0.002), (0.05, 0.01), (0.01, 0.001)):
            for name, pairs in arms.items():
                s = score(pairs, spec_tol, rot_tol, screen)
                if not s["n"]:
                    print(f"{name:16} {spec_tol:9.3f} {rot_tol:8.4f}   -- all screened out")
                    continue
                print(
                    f"{name:16} {spec_tol:9.3f} {rot_tol:8.4f} {s['n']:4d} "
                    f"{s['tier1']:9.5f} {s['rot']:9.5f} {s['spec']:9.5f} "
                    f"{s['lattice']:9.5f} {s['frac_agree']:7.2f}"
                )
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
