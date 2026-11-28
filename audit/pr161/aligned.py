"""監修（C）：ノート117 の帯は「0 < f < 1 のセル」から作るので、界面が格子の面にちょうど乗る初期状態（f が 0 と 1 だけ）
では帯が空になる。そのとき切り替えは効かない。面に揃った正方形の液滴・帯を、斜め・格子方向の一様流で運んで確かめる。
(U = (1, 1)・CFL 0.5 では、角のセルは 1 ステップで中身が全部抜けうる。)"""
import os, math, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np

def square(grid):
    n = grid.n[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (((i >= 3 * n // 8) & (i < 5 * n // 8)) & ((j >= 3 * n // 8) & (j < 5 * n // 8))).astype(float)

def column(grid):
    n = grid.n[0]
    i = np.arange(n)[:, None] * np.ones((1, n))
    return ((i >= n // 4) & (i < n // 2)).astype(float)

def stairs(grid):           # 面に揃った斜めの階段の帯（全セルが 0 か 1）
    n = grid.n[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (((i + j) % n) < n // 3).astype(float)

def run(a):
    shape, U, mode, ratio, steps = a
    import cases.r0p1_source_feedback as m
    m.SHAPES.update({"square": square, "column": column, "stairs": stairs, "square-bubble": lambda g: 1.0 - square(g)})
    r = m.growth(ratio, mode, 32, 0.5, steps, U, shape=shape)
    return "%-14s U=%-11s %-11s ratio=%-6g  max|u-U| %.2e  unstable %s  rate %.3g  (%d steps)" % (shape, U, mode, ratio, r["max"], r["unstable"], r["rate"], r["steps"])

if __name__ == "__main__":
    J = [(s, U, m, r, 600) for s in ("square", "square-bubble", "column", "stairs") for U in ((1.0, 1.0), (1.0, 0.37), (1.0, 0.0))
         for m in ("none", "face-band1", "face-band2") for r in (1e3, 1e6)]
    with mp.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)
