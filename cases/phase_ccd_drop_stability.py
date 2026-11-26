"""Why ``faces="phase-ccd"`` blows up on the paper's oscillating drop (water--air, docs/notes/63): the largest
velocity every 10 steps to half a period, with the density ratio, the capillary step, the solver tolerance and the
transport changed one at a time (phi curvature, xi-SDF every 10 steps).

    python -m cases.phase_ccd_drop_stability [--projection | --cell]

``--projection``: after 20 steps, the rho-weighted kinetic energy of a random face velocity before / after the face
projection, and the change made by a second projection (fd2 and phase-ccd, density ratio 1 and 830).
"""


def run(curv, reinit, n, **extra):
    import math
    import warnings

    warnings.filterwarnings("ignore")
    from cases.paper_benchmarks import setup
    from cases.phase_ccd_check import CURVATURES
    ck = CURVATURES[curv]
    extra.setdefault("transport", "ab2")
    ns, info = setup("drop", n, faces="phase-ccd", reinit_every=reinit, reinit_method="xi_sdf",
                     curvature_kw=None if ck is None else dict(ck), **extra)
    ns.start()
    hist = []
    T = info["T"]
    while ns.clock.t < 0.5 * T:
        ns.step(); u = ns.max_velocity()
        if ns.clock.i % 10 == 0: hist.append((ns.clock.i, round(ns.clock.t / T, 3), round(u, 4)))
        if not math.isfinite(u) or u > 3: hist.append((ns.clock.i, round(ns.clock.t / T, 3), u)); break
    return hist

def projection(faces, rho2, steps=20):
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np

    from basilisk_cupy.fields import FaceField
    from basilisk_cupy.projection.approximate import project
    from cases.paper_benchmarks import setup

    ns, _ = setup("drop", 32, faces=faces, transport="ab2", reinit_every=10, reinit_method="xi_sdf",
                  curvature_kw={"source": "phi"}, **({} if rho2 is None else {"rho2": rho2}))
    ns.start()
    for _ in range(steps):
        ns.step()
    rng = np.random.default_rng(0)

    def energy(w):
        return sum(float(np.sum(np.asarray(w[ax]) ** 2 / np.asarray(ns.alpha[ax]))) for ax in range(2))

    out = []
    for _ in range(3):
        uf = FaceField(ns.grid, [rng.standard_normal(np.asarray(ns.uf[ax]).shape) for ax in range(2)], "uf")
        ns._apply_uf_bc(uf)
        e0 = energy(uf)
        ns.dp.data[...] = 0.0
        project(uf, ns.dp, ns.alpha, 1.0, ops=ns.ops, tolerance=1e-12)
        ns._apply_uf_bc(uf)
        e1, u1 = energy(uf), [np.asarray(uf[ax]).copy() for ax in range(2)]
        ns.dp.data[...] = 0.0
        project(uf, ns.dp, ns.alpha, 1.0, ops=ns.ops, tolerance=1e-12)
        ns._apply_uf_bc(uf)
        d = max(float(np.abs(np.asarray(uf[ax]) - u1[ax]).max()) for ax in range(2))
        out.append((round(e1 / e0, 4), round(energy(uf) / e1, 6), d))
    return out


if __name__ == "__main__":
    import sys

    from cases.parallel import pmap

    if "--cell" in sys.argv:  # the cell-centred correction's coefficient (docs/notes/65) with phase-ccd
        items = [dict(curv="phi", reinit=10, n=n, cell_coefficient=c) for n in (32, 64) for c in ("gfm", "cut", "mean")]
        for it, h in zip(items, pmap(run, items, 6)):
            print(it, h[:2], "...", h[-2:], flush=True)
        sys.exit()
    if "--projection" in sys.argv:
        items = [dict(faces=f, rho2=r) for f in ("fd2", "phase-ccd") for r in (None, 998.2)]
        for it, o in zip(items, pmap(projection, items, 4)):
            print(it, o, flush=True)
        sys.exit()
    items = [dict(curv=c, reinit=r, n=32) for c in ("psi", "phi", "heights-fd2") for r in (1, 10)]
    items += [dict(curv="phi", reinit=10, n=32, rho2=998.2), dict(curv="phi", reinit=10, n=32, rho2=99.82),
              dict(curv="phi", reinit=10, n=32, capillary_dt_factor=0.25),
              dict(curv="phi", reinit=10, n=32, tolerance=1e-10), dict(curv="phi", reinit=10, n=32, transport="frozen"),
              dict(curv="phi", reinit=10, n=64, capillary_dt_factor=0.25)]
    for it, h in zip(items, pmap(run, items, 12)):
        print(it, h[:3], "...", h[-3:], flush=True)
