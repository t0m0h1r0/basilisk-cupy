"""docs/notes/116 (R0-P1, issue #141; B's decision on PR #155: option 1, at most two candidates): can the 2 dt
oscillation of the pressure of the "trapezoid" form (docs/notes/114: dt/2 G added before the transport, nothing taken
back) be removed by a minimal change, keeping the stability and the conservation?

  candidate 1 "trapezoid"    G^{n+1} = 2 g~ of the last projection; the first step is the library's (G^1 from a
                             projection) -- the form of note 114, i.e. "initialise by a projection";
  candidate 2 "trapezoid-p"  the half impulse from the MEAN OF THE LAST TWO PRESSURES, with the present face
                             coefficients and cell average (a force that sums to zero).

B's pass conditions, fixed before the measurement (PR #155): 1 no growth at the density ratios 1e3 and 1e6 for a
drop, a bubble and an oblique band; 2 total momentum (pressure included) to 1e-9; 3 single-phase accuracy kept;
4 step-to-step change of g at the level of the form without a source, not growing with N; 5 compatible with the
viscous step. One failure = rejected, no tuning.

    python -m cases.r0p1_trapezoid        # docs/figures/r0p1_trapezoid.json (about 10 minutes on 16 cores)
"""

from __future__ import annotations

import json
import os
import sys

from cases.r0p1_source_feedback import growth, single_phase, vortex

FORMS = ("none", "trapezoid", "trapezoid-p")


def _job(a):
    kind, args, kw = a
    return kind, {"growth": growth, "single": single_phase, "vortex": vortex}[kind](*args, **kw)


def jobs():
    out = [("growth", (r, m, n), {"steps": 1000, "shape": sh}) for m in FORMS for r in (1e3, 1e6)
           for sh in ("drop", "bubble", "band") for n in (32, 64)]
    out += [("single", (n, m), {}) for m in FORMS + ("face",) for n in (32, 48, 64, 96, 128, 192, 256)]
    out += [("vortex", (m, n), {}) for m in FORMS for n in (32, 64)]
    return sorted(out, key=lambda a: -{"growth": a[1][-1], "single": a[1][0], "vortex": a[1][-1]}[a[0]])


def main() -> None:
    import multiprocessing as mp

    res = {"growth": [], "single": [], "vortex": []}
    with mp.get_context("spawn").Pool(int(os.environ.get("R0P1_PROCS", "16"))) as pool:
        for kind, o in pool.imap_unordered(_job, jobs()):
            res[kind].append(o)
            print(kind, {k: (float("%.4g" % v) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    for k in res:
        res[k].sort(key=lambda o: json.dumps(o, sort_keys=True, default=str))
    json.dump(res, open("docs/figures/r0p1_trapezoid.json", "w"), indent=1, default=float)
    print("wrote docs/figures/r0p1_trapezoid.json")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.exit(main())
