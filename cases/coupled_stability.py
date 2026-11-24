"""Linear model of the capillary loop in CenteredNS and the viscous-only decay test (docs/notes/33).

Model: a capillary mode with interface amplitude ``eta`` and velocity ``v`` (``v`` in units of ``dt``),
force ``F = -omega^2 eta``, ``a = (omega dt)^2``, viscous factor ``b = 1/(1 + d dt)``.

* ``basilisk``: the order of :class:`CenteredNS` with ``vof_first``: ``eta' = eta + v`` (previous velocity),
  the predictor adds the previous step's ``g`` (pressure and force of ``eta``), solves the viscosity and
  removes ``g``, then the new force enters with the projection:
  ``v' = b (v + F(eta)) - F(eta) + F(eta')``;
* ``viscous_first``: ``v' = b v + F(eta)``, ``eta' = eta + v'`` (the order of the model we were sent);
* ``force_inside``: ``v' = b (v + F(eta))``, ``eta' = eta + v'`` (force and viscosity implicit together);
* ``basilisk`` with the transport velocity smoothed (docs/notes/27): ``eta' = eta + S v``, ``S = 1/(1 + theta a)``.

``limit(model, D, theta)`` is the largest ``omega dt`` with spectral radius <= 1 (bisection).
"""

from __future__ import annotations

import numpy as np


def matrix(model: str, x: float, D: float, theta: float = 0.0):
    a, b = x * x, 1.0 / (1.0 + D)
    if model == "basilisk":
        S = 1.0 / (1.0 + theta * a)
        return np.array([[1.0, S], [-a * b, b - a * S]])
    if model == "viscous_first":
        return np.array([[1.0 - a, b], [-a, b]])
    if model == "force_inside":
        return np.array([[1.0 - a * b, b], [-a * b, b]])
    raise ValueError(model)


def limit(model: str, D: float, theta: float = 0.0, top: float = 20.0) -> float:
    lo, hi = 0.0, top
    for _ in range(60):
        x = 0.5 * (lo + hi)
        if max(abs(np.linalg.eigvals(matrix(model, x, D, theta)))) <= 1.0 + 1e-12:
            lo = x
        else:
            hi = x
    return lo


def viscous_decay(visc: str, initial: str = "noise", n: int = 64, steps: int = 40):
    """Kinetic energy ratio after each step: frozen drop interface (equal densities, mu = 0.1), no surface
    tension, the velocity of 30 steps of the drop (``spurious``) or projected white noise (``noise``)."""
    from basilisk_cupy import BoundaryConditions, CartesianGrid
    from basilisk_cupy.interface.cls import CLSInterface
    from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
    from basilisk_cupy.terms.viscous import CCDViscosity
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    def make(v):
        grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
        x, y = grid.meshgrid()
        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=1e-10,
                        viscous=None if v == "fd2" else CCDViscosity(form=v))
        cls = CLSInterface.from_level_set(grid, 0.25 - np.hypot(x, y), BoundaryConditions())
        tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=1.0, mu1=0.1, mu2=0.1, pressure_jump=True, capillary_dt_factor=1.0)
        tp.add_surface_tension(1.0)
        return ns, tp, cls

    ns0, _, _ = make("fd2")
    ns0.start()
    for _ in range(30):
        ns0.step()
    dt = float(ns0.clock.dt)
    if initial == "noise":
        rng = np.random.default_rng(0)
        u0 = [3e-5 * rng.standard_normal((n, n)) for _ in range(2)]
    else:
        u0 = [c.interior.copy() for c in ns0.u]
    ns, tp, cls = make(visc)
    tp.potentials.clear()
    cls.advect = lambda uf, dt: None
    ns.DT = dt
    for c, v0 in zip(ns.u, u0):
        c.interior = v0.copy()
    ns.u.fill_ghosts()
    ns.start()
    e0 = sum(float((c.interior ** 2).sum()) for c in ns.u)
    out = []
    for _ in range(steps):
        ns.step()
        out.append(sum(float((c.interior ** 2).sum()) for c in ns.u) / e0)
    return np.array(out)


#: Measured (static drop, GFM, equal densities, N = 64, 600 steps, bisection): stable / unstable factor of
#: tension.h's step for each viscosity.
MU_SCAN = {0.0: (1.600, 1.650), 0.003: (1.500, 1.550), 0.01: (1.400, 1.450), 0.03: (1.300, 1.350),
           0.1: (1.150, 1.200), 0.3: (1.050, 1.100), 1.0: (1.000, 1.050)}


def main() -> None:
    import argparse

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4), constrained_layout=True)
    Ds = np.logspace(-2, 2, 40)
    for m, lab in (("basilisk", "CenteredNS order"), ("viscous_first", "viscosity, then force"),
                   ("force_inside", "force inside the implicit viscosity")):
        ax[0].semilogx(Ds, [limit(m, D) for D in Ds], label=lab)
    ax[0].set_xlabel("D = d dt")
    ax[0].set_ylabel("stable omega dt")
    ax[0].set_title("model: stability limit (search capped at 20)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    mus = [m for m in MU_SCAN if m > 0]
    ref = np.mean(MU_SCAN[0.0])
    ax[1].semilogx(mus, [np.mean(MU_SCAN[m]) / ref for m in mus], "o-", label="measured (limit / limit at mu = 0)")
    ax[1].axhline(0.5, color="C0", ls=":", label="model, D -> infinity: 0.5")
    ax[1].set_xlabel("mu")
    ax[1].set_title("measured: static drop, N = 64")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    for th in (0.0, 0.25, 1.0):
        ax[2].semilogx(Ds, [limit("basilisk", D, th) for D in Ds], label=f"theta = {th}")
    ax[2].set_xlabel("D = d dt")
    ax[2].set_title("model: transport velocity smoothed (27); capped at 20")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    for v in ("fd2", "split", "layered"):
        ax[3].semilogy(np.arange(1, 41), viscous_decay(v), label=v)
    ax[3].set_xlabel("step")
    ax[3].set_ylabel("E / E0")
    ax[3].set_title("viscous only, projected white noise")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.3)
    fig.savefig(f"{args.out}/coupled_stability.png", dpi=100)
    print("wrote", f"{args.out}/coupled_stability.png")


if __name__ == "__main__":
    main()
