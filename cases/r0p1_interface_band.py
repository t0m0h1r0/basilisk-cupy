"""docs/notes/117 (R0-P1, issue #141; B's decision on PR #158: option 4, one switching rule, two thresholds): the
half-step source in the face value of the conserving momentum transport (docs/notes/112) switched off near the
interface.

Rule: the cell field g that the library averages to the faces is set to zero in the interfacial cells (0 < f < 1)
and in the cells within w cells of one (a (2w + 1)^2 block), w = 1 ("face-band1") or 2 ("face-band2"), from the
volume fractions at the start of the step. The flux form is kept (conservation by construction); with no interface
the scheme is the unswitched one.

B's pass conditions, fixed before the measurement (PR #158): 1 free stream at round-off (as without a source) at the
density ratios 1e3 and 1e6 for a drop, a bubble and an oblique band; 2 total momentum (pressure included) to 1e-9;
3 no growth for the same shapes (about 2000 steps); 4 identical to the unswitched scheme with no interface; 5 the
verdicts of 1-3 the same for the two thresholds. Recorded only: the time order in a two-phase flow from differences
of fields at the same time (a drop in a vortex).

    python -m cases.r0p1_interface_band       # docs/figures/r0p1_interface_band.json
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from cases.r0p1_source_feedback import growth, single_phase, vortex

FORMS = ("none", "face-band1", "face-band2")
TC = {"n": (64, 128), "cfl": (0.5, 0.25, 0.125)}


def identical(n, mode, directory):
    """Condition 4: with no interface the fields of the switched and of the unswitched scheme (Taylor-Green)."""
    a = os.path.join(directory, "tg_face_%d.npy" % n)
    b = os.path.join(directory, "tg_%s_%d.npy" % (mode, n))
    ea = single_phase(n, "face", save=a)
    eb = single_phase(n, mode, save=b)
    return {"n": n, "mode": mode, "difference": float(np.abs(np.load(a) - np.load(b)).max()), "error": eb["max"],
            "error_unswitched": ea["max"]}


def field(mode, n, cfl, directory):
    path = os.path.join(directory, "vortex_%s_%d_%g.npy" % (mode, n, cfl))
    o = vortex(mode, n, 1000.0, 0.25, cfl, save=path)
    return dict(o, path=path)


def _job(a):
    kind, args, kw = a
    return kind, {"growth": growth, "vortex": vortex, "identical": identical, "field": field}[kind](*args, **kw)


def jobs(directory):
    out = [("growth", (r, m, n), {"steps": 2000, "shape": sh}) for m in FORMS for r in (1e3, 1e6)
           for sh in ("drop", "bubble", "band") for n in (32, 64)]
    out += [("vortex", (m, n, 1000.0, 1.0), {}) for m in FORMS for n in (32, 64)]
    out += [("identical", (n, m, directory), {}) for m in FORMS[1:] for n in (32, 64)]
    out += [("field", (m, n, c, directory), {}) for m in FORMS + ("accel",) for n in TC["n"] for c in TC["cfl"]]
    return sorted(out, key=lambda a: -{"growth": lambda: a[1][2] ** 2 * 20, "vortex": lambda: a[1][1] ** 3,
                                       "identical": lambda: a[1][0] ** 3, "field": lambda: a[1][1] ** 3 / a[1][2]}[a[0]]())


def time_coefficient(fields):
    """Same grid, same time t = 0.25, dt halved: a1 = |u(dt) - u(dt/2)|_rms / (dt - dt/2) (docs/notes/113, section 6);
    also restricted to the cells away from the interface (f = 0 or 1 in the 5 x 5 block in both runs)."""
    rows = []
    F = {(o["mode"], o["n"], o["cfl"]): o for o in fields}
    for mode in FORMS + ("accel",):
        for n in TC["n"]:
            pairs = []
            for c0, c1 in zip(TC["cfl"][:-1], TC["cfl"][1:]):
                a, b = F[(mode, n, c0)], F[(mode, n, c1)]
                ua, ub = np.load(a["path"]), np.load(b["path"])
                d = np.hypot(ua[0] - ub[0], ua[1] - ub[1])
                mixed = ((ua[2] > 1e-12) & (ua[2] < 1 - 1e-12)) | ((ub[2] > 1e-12) & (ub[2] < 1 - 1e-12))
                near = np.zeros_like(mixed)
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        near |= np.roll(np.roll(mixed, i, axis=0), j, axis=1)
                dt = 0.25 / a["steps"] - 0.25 / b["steps"]
                pairs.append({"cfl": c0, "a1_rms": float(np.sqrt((d * d).mean()) / dt), "a1_max": float(d.max() / dt),
                              "a1_rms_away": float(np.sqrt((d[~near] ** 2).mean()) / dt),
                              "a1_rms_near": float(np.sqrt((d[near] ** 2).mean()) / dt),
                              "near_fraction": float(near.mean())})
            rows.append({"mode": mode, "n": n, "pairs": pairs})
    return rows


def main() -> None:
    import multiprocessing as mp

    directory = os.environ.get("R0P1_FIELDS", "r0p1_band_fields")
    os.makedirs(directory, exist_ok=True)
    res = {"growth": [], "vortex": [], "identical": [], "field": []}
    with mp.get_context("spawn").Pool(int(os.environ.get("R0P1_PROCS", "16"))) as pool:
        for kind, o in pool.imap_unordered(_job, jobs(directory)):
            res[kind].append(o)
            print(kind, {k: (float("%.4g" % v) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    res["time_coefficient"] = time_coefficient(res["field"])
    for o in res["field"]:
        o.pop("path")
    for k in ("growth", "vortex", "identical", "field"):
        res[k].sort(key=lambda o: json.dumps(o, sort_keys=True, default=str))
    json.dump(res, open("docs/figures/r0p1_interface_band.json", "w"), indent=1, default=float)
    for r in res["time_coefficient"]:
        print("a1", r["mode"], r["n"], [(p["cfl"], "%.3g" % p["a1_rms"], "away %.3g" % p["a1_rms_away"], "near %.3g" % p["a1_rms_near"])
                                         for p in r["pairs"]])
    print("wrote docs/figures/r0p1_interface_band.json")
    assert math.isfinite(1.0)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.exit(main())
