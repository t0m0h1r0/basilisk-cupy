"""監修（独立の検証）：相分離 PCCD（部分セル + blend）の Gram の上界 λ_max ≤ 20.5（ノート97）を、
ノートが走査していない配置で調べる。

 (1) 両方の界面の体積分率 (c_a, c_b) を同時に動かす（ノート97 は b = 0.7137 固定で a だけ）。
 (2) 相の列が短い配置（純相のセルが 5〜8 個）。4 個以下は例外になることも確かめる。
 (3) N 依存。

    python -m audit.gram_adversarial
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def gram(N, ratio, a, b, mode="blend_partial"):
    from cases.pccd_floquet import _basis
    from cases.pccd_gram import quadrature

    try:
        phi = _basis(N, ratio, a, b, mode)
    except ValueError as e:
        return {"error": str(e)[:60]}
    X, W, Hv, cell = quadrature(N, ratio, a, b)
    B = np.stack([phi(j, X) for j in range(N)], axis=1)
    G = B.T @ (W[:, None] * B)
    m = np.bincount(cell, weights=W, minlength=N)
    s = 1.0 / np.sqrt(m)
    A = G * s[:, None] * s[None, :]
    ev = np.linalg.eigvalsh(A)
    # Λ: 質量ノルムでの再構築の最大の増幅 sup |u_h|_∞ / |q|_∞ の代わりに ‖B‖∞（点の値の行和の最大）
    return {"lmax": float(ev[-1]), "lmin": float(ev[0]), "Binf": float(np.abs(B).sum(1).max())}


def job(args):
    return args, gram(*args)


def main():
    N = 32
    h = 1.0 / N
    jobs = []
    cs = [1e-6, 1e-4, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-4, 1 - 1e-6]
    # (1) 長い列（重い相 14 セル前後）、(c_a, c_b) の格子
    for r in (1.0, 1e3, 1e6):
        for ca in cs:
            for cb in cs:
                a = (8 + 1 - ca) * h          # 重い相は a から右。混合セル 8 の重い分率 ca
                b = (23 + cb) * h             # 混合セル 23 の重い分率 cb
                jobs.append(("long", N, r, a, b, ca, cb))
    # (2) 短い列：重い相の純相セル n 個（n = 3..8）、軽い相の純相セル n 個
    for r in (1.0, 1e3, 1e6):
        for n in (3, 4, 5, 6, 7, 8):
            for ca in (1e-4, 1e-2, 0.3, 0.5, 0.9, 1 - 1e-4):
                for cb in (1e-4, 1e-2, 0.5, 0.9, 1 - 1e-4):
                    a = (8 + 1 - ca) * h
                    b = (9 + n + cb) * h
                    jobs.append(("short_heavy", N, r, a, b, ca, cb, n))
                    # 軽い相が短い：重い相は b' から a'（周期で回る）
                    a2 = (9 + n + 1 - ca) * h      # 重い相の始まり
                    b2 = (8 + cb) * h              # 重い相の終わり
                    jobs.append(("short_light", N, r, a2, b2, ca, cb, n))
    # (3) N 依存（c = 0.5, 1e-2 で）
    for NN in (16, 64):
        hh = 1.0 / NN
        for r in (1.0, 1e3, 1e6):
            for ca in (1e-2, 0.5):
                jobs.append(("N", NN, r, (NN // 4 + 1 - ca) * hh, (3 * NN // 4 + 0.37) * hh, ca, 0.37))
    args = [(j[1], j[2], j[3], j[4]) for j in jobs]
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(job, args, chunksize=4))
    out = [{"tag": j[0], "N": j[1], "ratio": j[2], "a": j[3], "b": j[4], "ca": j[5], "cb": j[6],
            "n": (j[7] if len(j) > 7 else None), **r[1]} for j, r in zip(jobs, res)]
    json.dump(out, open("audit/gram_adversarial.json", "w"), indent=1)
    for tag in ("long", "N"):
        for r in (1.0, 1e3, 1e6):
            sel = [o for o in out if o["tag"] == tag and o["ratio"] == r and "lmax" in o]
            if sel:
                w = max(sel, key=lambda o: o["lmax"])
                print(tag, r, "max lmax %.2f at ca=%g cb=%g N=%d ; min lmin %.6f" % (w["lmax"], w["ca"], w["cb"], w["N"], min(o["lmin"] for o in sel)))
    for tag in ("short_heavy", "short_light"):
        for n in (3, 4, 5, 6, 7, 8):
            for r in (1.0, 1e3, 1e6):
                sel = [o for o in out if o["tag"] == tag and o["ratio"] == r and o["n"] == n]
                ok = [o for o in sel if "lmax" in o]
                if ok:
                    w = max(ok, key=lambda o: o["lmax"])
                    print(tag, "n=%d r=%g: max lmax %.1f (ca=%g cb=%g), Binf max %.1f, errors %d/%d" % (n, r, w["lmax"], w["ca"], w["cb"], max(o["Binf"] for o in ok), len(sel) - len(ok), len(sel)))
                else:
                    print(tag, "n=%d r=%g: all %d failed: %s" % (n, r, len(sel), sel[0]["error"]))


if __name__ == "__main__":
    main()
