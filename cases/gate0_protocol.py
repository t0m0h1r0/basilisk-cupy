"""docs/notes/103 (Gate 0, the audit issues #114, #115, #121): the convergence protocol applied to method D and to
the start-up flow. 13 grids, 3 interface offsets, max and mass-weighted L2 errors, least-squares orders over all N
and over N >= 96, every property.

    python -m cases.gate0_protocol
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cases.coupled_boost import slopes
from cases.viscous_minenergy import residual_map, single_phase_d
from cases.viscous_walls import startup

NS = (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256)
OFFS = (0.0, 0.0071, 0.0193)
CASES = (("mu", 1e3), ("water-air", 1e3), ("water-air", 1.0), ("nu", 1e3))


def _d(N, kind, ratio, off):
    return residual_map(N, ratio, kind, a=0.2613 + off, b=0.7137 + off)


def _sp(N, off):
    return single_phase_d(N, kind_basis="split")


def _st(N, kind, ratio, off):
    return startup(N, ratio, kind, a=0.3613 + off)


def main() -> None:
    res = {"d": [], "single": [], "startup": []}
    with ProcessPoolExecutor(18) as ex:
        fd = [((N, k, r, o), ex.submit(_d, N, k, r, o)) for N in NS[::-1] for k, r in CASES for o in OFFS]
        fs = [((N,), ex.submit(_sp, N, 0.0)) for N in NS[::-1]]
        ft = [((N, k, r, o), ex.submit(_st, N, k, r, o)) for N in NS[::-1] for k, r in CASES for o in OFFS]
        for key, f in fd:
            res["d"].append({"N": key[0], "kind": key[1], "ratio": key[2], "off": key[3], **f.result()})
        for key, f in fs:
            res["single"].append({"N": key[0], **f.result()})
        for key, f in ft:
            res["startup"].append({"N": key[0], "kind": key[1], "ratio": key[2], "off": key[3], **f.result()})
    orders = []

    def worst(rows, get):
        return [max(get(r) for r in rows if r["N"] == N) for N in NS]
    for k, r in CASES:
        rows = [x for x in res["d"] if x["kind"] == k and x["ratio"] == r]
        for m in ("C", "D"):
            for norm in ("max", "L2m"):
                e = worst(rows, lambda x: x[m][norm])
                po = [slopes(NS, [next(x[m][norm] for x in rows if x["N"] == N and x["off"] == o) for N in NS]) for o in OFFS]
                orders.append({"test": "stokes", "kind": k, "ratio": r, "method": m, "norm": norm, "worst": slopes(NS, e),
                               "per_offset": po, "errors": e})
        e = worst(rows, lambda x: x["r_max_over_qdot"])
        orders.append({"test": "residual", "kind": k, "ratio": r, "worst": slopes(NS, e), "errors": e})
        rows = [x for x in res["startup"] if x["kind"] == k and x["ratio"] == r]
        for m in ("corrected", "uncorrected"):
            e = worst(rows, lambda x: x[m])
            orders.append({"test": "startup", "kind": k, "ratio": r, "method": m, "worst": slopes(NS, e), "errors": e})
    for m in ("C", "D"):
        e = [next(x[m] for x in res["single"] if x["N"] == N) for N in NS]
        orders.append({"test": "single_split", "method": m, "worst": slopes(NS, e), "errors": e})
    res["orders"] = orders
    for o in orders:
        print(o["test"], o.get("kind", ""), o.get("ratio", ""), o.get("method", ""), o.get("norm", ""),
              "order all / N>=96: %.2f / %.2f" % o["worst"], " ".join(f"{v:.2e}" for v in o["errors"]), flush=True)
    json.dump(res, open("docs/figures/gate0_protocol.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
