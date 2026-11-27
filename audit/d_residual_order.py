"""監修：D の残差 r = (G_ν − G_A) q̇_C / m の大きさの次数（局所の打切り誤差に相当）。python -m audit.d_residual_order"""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

NS = (32, 48, 64, 96, 128, 192, 256)


def job(a):
    from cases.viscous_minenergy import residual_map
    N, k, r = a
    x = residual_map(N, r, k)
    return {"N": N, "kind": k, "ratio": r, "r_over_qdot": x["r_max_over_qdot"], "r_mixed": x["r_mixed"], "r_near": x["r_near"], "r_far": x["r_far"]}


if __name__ == "__main__":
    jobs = [(N, k, r) for k, r in (("mu", 1e3), ("water-air", 1e3), ("water-air", 1.0), ("nu", 1e3)) for N in NS]
    with ProcessPoolExecutor(14) as ex:
        res = list(ex.map(job, jobs))
    json.dump(res, open("audit/d_residual_order.json", "w"), indent=1)
    for k, r in (("mu", 1e3), ("water-air", 1e3), ("water-air", 1.0), ("nu", 1e3)):
        sel = [x for x in res if x["kind"] == k and x["ratio"] == r]
        v = np.array([x["r_over_qdot"] for x in sel])
        far = np.array([x["r_over_qdot"] * x["r_far"] for x in sel])
        p = -np.polyfit(np.log(NS), np.log(v), 1)[0]
        pf = -np.polyfit(np.log(NS), np.log(far), 1)[0]
        print(k, r, "max|r|/max|q̇|:", " ".join(f"{x:.1e}" for x in v), "次数 %.2f" % p, "| 界面から 4 セル以上：", " ".join(f"{x:.1e}" for x in far), "次数 %.2f" % pf)
