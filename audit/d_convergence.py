"""監修（独立の検証）：D 方式（G_A q̇ = −K_ν q）の「時間発展 3.2〜4.1 次」（ノート100・101）を、
N を細かく取って測り直す。ノートは N = 32, 64, 128 の 3 点だけで、界面のセル内の位置が N ごとに変わる。

    python -m audit.d_convergence
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

NS = (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256)
CASES = (("mu", 1e3), ("water-air", 1e3), ("water-air", 1.0), ("nu", 1e3), ("single-split", 1.0))


def job(arg):
    N, kind, ratio = arg
    if kind == "single-split":
        from cases.viscous_minenergy import single_phase_d
        r = single_phase_d(N, kind_basis="split")
        return {"N": N, "kind": kind, "ratio": ratio, "C_max": r["C"], "D_max": r["D"]}
    from cases.viscous_minenergy import residual_map
    r = residual_map(N, ratio, kind)
    return {"N": N, "kind": kind, "ratio": ratio, "C_max": r["C"]["max"], "C_L2": r["C"]["L2m"],
            "D_max": r["D"]["max"], "D_L2": r["D"]["L2m"], "frac_a": (N * 0.2613) % 1, "frac_b": (N * 0.7137) % 1}


def main():
    jobs = [(N, k, r) for k, r in CASES for N in NS]
    jobs.sort(key=lambda j: -j[0])
    with ProcessPoolExecutor(14) as ex:
        res = list(ex.map(job, jobs))
    json.dump(res, open("audit/d_convergence.json", "w"), indent=1)
    for k, r in CASES:
        sel = sorted([x for x in res if x["kind"] == k and x["ratio"] == r], key=lambda x: x["N"])
        lN = np.log([x["N"] for x in sel])
        print(f"== {k} ratio {r:g}")
        for key in ("C_max", "D_max", "C_L2", "D_L2"):
            if key not in sel[0]:
                continue
            v = np.array([x[key] for x in sel])
            p_all = -np.polyfit(lN, np.log(v), 1)[0]
            hi = lN >= np.log(96)
            p_hi = -np.polyfit(lN[hi], np.log(v[hi]), 1)[0]
            print(f"  {key}: 次数（全体の最小二乗）{p_all:.2f}、N ≥ 96 だけ {p_hi:.2f} | " + " ".join(f"{x:.2e}" for x in v))


if __name__ == "__main__":
    main()
