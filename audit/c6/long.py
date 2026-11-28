"""C-6 の続き：長い時間（静止液滴 t = 5、毛管波 t = 4 ≈ 10 周期）と、A-1 の候補 D1+D2lim。"""
import json, sys
import multiprocessing as mp
from audit.c6.surface_tension import run
if __name__ == "__main__":
    J = []
    for v in ("cons", "cons-lim"):
        for n in (32, 64):
            J += [("drop", v, n, 1e3, "numeric", 55.0, 5.0), ("bubble", v, n, 1e3, "numeric", 55.0, 5.0), ("wave", v, n, 1e3, "numeric", 55.0, 4.0)]
        J += [("drop", v, 32, 1e3, "exact", 55.0, 5.0)]
    J += [("wave", "plain", 64, 1e3, "numeric", 55.0, 4.0), ("drop", "plain", 64, 1e3, "numeric", 55.0, 5.0)]
    with mp.get_context("spawn").Pool(10, maxtasksperchild=1) as pool, open("audit/c6/long.jsonl", "w") as fh:
        for r in pool.imap_unordered(run, J, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
