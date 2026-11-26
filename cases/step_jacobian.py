"""Matrix-free linear stability of one CenteredNS + TwoPhaseCLS step about a static drop (docs/notes/35).

The state carried from step to step is ``q = (psi, u_x, u_y, uf_x, uf_y, p, g_x, g_y)``. ``StepMap`` runs one
step from a given ``q`` with a fixed ``dt`` (tension.h's bound times ``factor``) and returns the new ``q``;
``jvp(v)`` is the central difference (perturbation of about ``eps`` per component) ``(F(q0 + eps v) - F(q0 - eps v)) / 2 eps`` about a reference state
``q0`` (the drop after ``settle`` steps, nearly steady). ``arnoldi`` returns the Ritz values of largest
modulus and their vectors (numpy only).
"""

from __future__ import annotations

import math

import numpy as np

from basilisk_cupy import BoundaryConditions, CartesianGrid
from basilisk_cupy.constants import PI
from basilisk_cupy.fields import FaceField
from basilisk_cupy.interface.cls import CLSInterface
from basilisk_cupy.interface.cls_two_phase import TwoPhaseCLS
from basilisk_cupy.time.basilisk_centered import CenteredNS


class StepMap:
    def __init__(self, n=32, rho2=1.0, mu=0.1, factor=1.0, K=0, settle=300, eps=1e-5, settle_factor=0.5,
                 tolerance=1e-14, mu2=None):
        grid = CartesianGrid((n, n), origin=(-0.5, -0.5))
        x, y = grid.meshgrid()
        ns = CenteredNS(grid, uf_normal={w: 0.0 for w in ("left", "right", "bottom", "top")}, tolerance=tolerance,
                        coupled_iterations=K)
        cls = CLSInterface.from_level_set(grid, 0.25 - np.hypot(x, y), BoundaryConditions())
        tp = TwoPhaseCLS(ns, cls, rho1=1.0, rho2=rho2, mu1=mu, mu2=mu if mu2 is None else mu2, pressure_jump=True,
                         capillary_dt_factor=1e9,  # dt fixed through DT below
                         cell_density="diffuse")  # the study's (docs/notes/36) density; sharp by default since 69
        tp.add_surface_tension(1.0)
        rhom = (1.0 + rho2) / 2.0
        h = 1.0 / n
        bound = math.sqrt(rhom * h ** 3 / (PI * 1.0))
        self.dt = factor * bound
        self.grid, self.ns, self.cls, self.tp, self.eps, self.n = grid, ns, cls, tp, eps, n
        ns.DT = settle_factor * bound  # the reference state is reached with a stable step
        ns.start()
        for _ in range(settle):
            ns.step()
        ns.DT = self.dt
        self.q0 = self.get()
        self.clock = (ns.clock.t, ns.clock.i, ns.clock.t + self.dt, self.dt)
        self.residual = float(np.abs(self.F(self.q0) - self.q0).max())

    # -- state ---------------------------------------------------------------------------------
    def get(self):
        ns, cls = self.ns, self.cls
        parts = [cls.psi.interior, ns.u[0].interior, ns.u[1].interior, ns.uf[0], ns.uf[1], ns.p.interior,
                 ns.g[0].interior, ns.g[1].interior]
        self.shapes = [a.shape for a in parts]
        return np.concatenate([np.asarray(a).ravel() for a in parts])

    def set(self, q):
        ns, cls = self.ns, self.cls
        out, k = [], 0
        for s in self.shapes:
            m = int(np.prod(s))
            out.append(q[k:k + m].reshape(s).copy())
            k += m
        cls.psi.interior = out[0]
        cls.psi.fill_ghosts()
        ns.u[0].interior, ns.u[1].interior = out[1], out[2]
        ns.u.fill_ghosts()
        ns.uf = FaceField(self.grid, [out[3], out[4]], "uf")
        ns.p.interior = out[5]
        ns.p.fill_ghosts()
        ns.g[0].interior, ns.g[1].interior = out[6], out[7]
        ns.g.fill_ghosts()

    def F(self, q):
        self.set(q)
        c = self.ns.clock
        if hasattr(self, "clock"):
            c.t, c.i, c.tnext, c.dt = self.clock
        self.ns.step()
        return self.get()

    def deflate(self, v):
        """Remove the neutral directions: a constant pressure (the gauge) and a change of the total psi
        (conserved exactly, so the subspace of zero total is invariant)."""
        v = v.copy()
        k = 0
        for idx, s in enumerate(self.shapes):
            m = int(np.prod(s))
            if idx in (0, 5):
                v[k:k + m] -= v[k:k + m].mean()
            k += m
        return v

    def jvp(self, v):
        v = self.deflate(v)
        nv = float(np.linalg.norm(v)) or 1.0
        e = self.eps * math.sqrt(v.size) / nv  # about eps per component: far above the solvers' residuals
        return self.deflate((self.F(self.q0 + e * v) - self.F(self.q0 - e * v)) / (2 * e))

    def dense(self):
        """The deflated Jacobian as a dense matrix (small grids only)."""
        size = self.q0.size
        J = np.empty((size, size))
        for i in range(size):
            e = np.zeros(size)
            e[i] = 1.0
            J[:, i] = self.jvp(e)
        return J

    def split(self, v):
        names = ["psi", "u", "v", "uf", "vf", "p", "gx", "gy"]
        out, k = {}, 0
        for name, s in zip(names, self.shapes):
            m = int(np.prod(s))
            out[name] = v[k:k + m].reshape(s)
            k += m
        return out


def arnoldi(apply, size, m=60, seed=0, nev=6, power=20):
    """Ritz values of largest modulus of ``apply`` after ``m`` Arnoldi steps (the start vector is a random
    vector after ``power`` applications, to favour the dominant modes), with their vectors."""
    rng = np.random.default_rng(seed)
    V = np.zeros((size, m + 1))
    H = np.zeros((m + 1, m))
    v = rng.standard_normal(size)
    for _ in range(power):
        v = apply(v)
        v = v / np.linalg.norm(v)
    V[:, 0] = v / np.linalg.norm(v)
    for j in range(m):
        w = apply(V[:, j])
        for i in range(j + 1):  # modified Gram-Schmidt, twice
            H[i, j] = V[:, i] @ w
            w = w - H[i, j] * V[:, i]
        for i in range(j + 1):
            c = V[:, i] @ w
            H[i, j] += c
            w = w - c * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-14:
            m = j + 1
            break
        V[:, j + 1] = w / H[j + 1, j]
    lam, Y = np.linalg.eig(H[:m, :m])
    order = np.argsort(-np.abs(lam))[:nev]
    return lam[order], V[:, :m] @ Y[:, order]


def leading(rho2, K, factor, n=32, mu=0.1, m=40, nev=6, settle=300, mu2=None):
    """Ritz values (largest modulus first) and vectors of one step's Jacobian, and the ``StepMap``."""
    sm = StepMap(n, rho2, mu, factor, K, settle=settle, mu2=mu2)
    lam, X = arnoldi(sm.jvp, sm.q0.size, m=m, power=20, nev=nev)
    return lam, X, sm


def main() -> None:
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("rho2", type=float)
    ap.add_argument("K", type=int)
    ap.add_argument("factors", type=float, nargs="+")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    for f in args.factors:
        lam, _, sm = leading(args.rho2, args.K, f, n=args.n)
        print(json.dumps({"rho2": args.rho2, "K": args.K, "f": f, "rho": float(abs(lam[0])),
                          "arg_over_pi": float(np.angle(lam[0]) / np.pi), "residual": sm.residual}), flush=True)


if __name__ == "__main__":
    main()
