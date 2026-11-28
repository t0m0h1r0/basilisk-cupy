"""監修（C）C-2：接点が角から遠い配置で、静水圧（reduced）が実際に崩れるかを完全なソルバーで確かめる。
上下の壁に接する斜めの平面（θ < 45°、位置が上下の壁の近く）。対照として、左右の壁だけに接する同じ角度の平面。"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp
def run(a):
    from cases.r0p2_pressure_momentum import hydrostatic
    th, off, n = a
    o = hydrostatic("smooth", th, "reduced", n, t_end=1.0, offset=off, samples=(0.1, 1.0))
    h = o["history"]
    return "theta=%-5g offset=%+.4f N=%-3d: %s %s" % (th, off, n, " ".join("t=%g cells %.1e" % (r["t"], r["cells"]) for r in h), o["failed"] or "")
if __name__ == "__main__":
    J = [(37.5, 0.45, 64), (37.5, -0.45, 64), (42.5, -0.4415, 64), (37.5, 0.45, 32), (37.5, 0.45, 128), (30.0, 0.40, 64), (26.565, 0.40, 64), (37.5, 0.0, 64), (26.565, 0.0, 64)]
    with mp.get_context("spawn").Pool(9, maxtasksperchild=1) as pool:
        for line in pool.imap(run, J, chunksize=1):
            print(line, flush=True)
