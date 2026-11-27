"""docs/notes/106 (Gate 1-B, audit issue #116): short phase runs in the 1-D phase-split PCCD.

The phase-split PCCD (docs/notes/97) needs at least 5 pure cells per run (the explicit end closure); shorter runs
raise. A curved interface in 2-D produces every run length, so the degenerate cases must be defined. The user's
guidance: do not demand 5th order where the data cannot carry it; keep conservation, boundedness and geometric
continuity.

SplitShort (a Split):
  * runs of n >= 5 pure cells: the PCCD run (unchanged);
  * 1 <= n <= 4: the single polynomial of degree n - 1 that matches all n cell averages of the run (conservative,
    exact for polynomials of degree n - 1);
  * n = 0 (a film straddling a face: two adjacent mixed cells and no pure cell of that phase): the film velocity is
    linear between the two interface values u_1, u_2; the other phase in each mixed cell keeps the partial-cell form
    P_ext + (u_G - P_ext(sigma)) b(t); u_1, u_2 from the two mass-weighted means (a 2 x 2 linear system).
Blended with SplitBlend (split_cls=SplitShort) the joins make the run length change continuously as an interface
crosses a face.

    python -m cases.short_runs
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.pccd import WG, XG, _mono
from cases.pccd_density import pieces
from cases.pccd_local import Split, SplitBlend


def _bub(t):
    return 3 * t * t - 2 * t ** 3


class SplitShort(Split):
    def _solve_run(self, run, ubar):
        N = self.N
        n = len(run)
        cv_, cg_ = self.closure[1], self.closure[3]
        nz = [i for i in range(cv_.size) if abs(cv_[i]) > 0 or abs(cg_[i]) > 0]
        need = max(5, nz[-1] + 1)
        if n >= need:
            return super()._solve_run(run, ubar)
        q = ubar[np.array(run) % N]
        d = n                                                   # number of coefficients (degree n - 1)
        A = np.array([[((k + 1) ** (j + 1) - k ** (j + 1)) / (j + 1) for j in range(d)] for k in range(n)])
        c = np.linalg.solve(A, q)
        for k, cell in enumerate(run):
            co = np.zeros(5)                                    # P(k + s) in s-monomials
            for j in range(d):
                for i in range(j + 1):
                    co[i] += c[j] * math.comb(j, i) * k ** (j - i)
            self.coef[cell % N] = co

    def __init__(self, N, ratio, a, b, ubar, closure=None, join=None, mixed_mode="extend"):
        self.film = {}
        super().__init__(N, ratio, a, b, ubar, closure, join, mixed_mode)
        if mixed_mode != "partial":
            return
        P = self.P
        for m in self.mixed:
            m2 = (m + 1) % N
            if m2 not in self.mixed:
                continue
            hv_right = P[m][-1][2]                              # the phase on the right of cell m
            if P[m2][0][2] != hv_right:
                continue
            self._setup_film(m, m2, hv_right, ubar)
        if self.film:
            for i in range(N):
                self.shift[i] = ubar[i] - self._wmean(i)

    def _setup_film(self, m, m2, hv_f, ubar):
        """The film of phase hv_f on [sigma_1 of cell m, sigma_2 of cell m2]: linear between u_1, u_2."""
        N, P, ratio = self.N, self.P, self.ratio
        rho = lambda hv: 1.0 if hv else 1.0 / ratio  # noqa: E731
        s1 = P[m][0][1]                                         # interface in m (film on the right)
        s2 = P[m2][0][1]                                        # interface in m2 (film on the left)
        Lf = (1.0 - s1) + s2
        hv_o = not hv_f
        # the other phase: cell m left part uses cell m-1's polynomial; cell m2 right part uses cell m2+1's
        rows, rhs = np.zeros((2, 2)), np.zeros(2)
        for row, (cell, nb, off, lo, hi, face, sig) in enumerate((
                (m, (m - 1) % N, -1.0, 0.0, s1, 0.0, s1), (m2, (m2 + 1) % N, 1.0, s2, 1.0, 1.0, s2))):
            sq = lo + (hi - lo) * XG
            w = (hi - lo) * WG
            pe = _mono(sq - off) @ self.coef[nb]
            pg = _mono(np.array([sig - off]))[0] @ self.coef[nb]
            bb = _bub(np.abs(sq - face) / abs(sig - face))
            # other phase: rho_o int (pe + (u_row - pg) bb)
            ms = rho(hv_o) * (hi - lo) + rho(hv_f) * (1.0 - (hi - lo))
            rhs[row] = ms * ubar[cell] - rho(hv_o) * (w @ (pe - pg * bb))
            rows[row, row] += rho(hv_o) * (w @ bb)
            # film part in this cell: u_1 + (u_2 - u_1) xi / Lf, xi measured from sigma_1 of cell m
            flo, fhi = (s1, 1.0) if cell == m else (0.0, s2)
            fq = flo + (fhi - flo) * XG
            fw = (fhi - flo) * WG
            xi = (fq - s1) if cell == m else (1.0 - s1 + fq)
            rows[row, 0] += rho(hv_f) * (fw @ (1.0 - xi / Lf))
            rows[row, 1] += rho(hv_f) * (fw @ (xi / Lf))
        u1, u2 = np.linalg.solve(rows, rhs)
        self.film[m] = ("left", s1, u1, u2, Lf, (m - 1) % N)
        self.film[m2] = ("right", s2, u1, u2, Lf, (m2 + 1) % N)

    def _raw(self, i, s):
        if i not in self.film:
            return super()._raw(i, s)
        side, sig, u1, u2, Lf, nb = self.film[i]
        out = np.empty_like(s)
        if side == "left":                                      # other phase on [0, sig), film on [sig, 1]
            k = s < sig
            pe = _mono(s[k] + 1.0) @ self.coef[nb]
            pg = _mono(np.array([sig + 1.0]))[0] @ self.coef[nb]
            out[k] = pe + (u1 - pg) * _bub(s[k] / sig)
            out[~k] = u1 + (u2 - u1) * (s[~k] - sig) / Lf
        else:                                                   # film on [0, sig], other phase on (sig, 1]
            k = s > sig
            pe = _mono(s[k] - 1.0) @ self.coef[nb]
            pg = _mono(np.array([sig - 1.0]))[0] @ self.coef[nb]
            out[k] = pe + (u2 - pg) * _bub((1.0 - s[k]) / (1.0 - sig))
            out[~k] = u1 + (u2 - u1) * (Lf - sig + s[~k]) / Lf
        return out


# --------------------------------------------------------------------------------------------- the experiment
def geometry(N, nL, f1=0.4, f2=0.7, f0=None):
    """Light run of nL pure cells between two mixed cells with light fractions f1 (right part of the first) and f2
    (left part of the last). Returns (a, b) for pieces(N, a, b): heavy on the arc a -> b."""
    f0 = N // 2 if f0 is None else f0
    h = 1.0 / N
    c0 = (f0 + 1 - f1) * h
    c1 = (f0 + 1 + nL + f2) * h
    return c1 % 1.0, c0 % 1.0


def _recon(N, ratio, a, b, q, kind):
    if kind == "blend":
        return SplitBlend(N, ratio, a, b, q, mixed_mode="partial", split_cls=SplitShort)
    if kind == "split":
        return SplitShort(N, ratio, a, b, q, mixed_mode="partial")
    if kind == "base":                                          # docs/notes/97 as it is
        return SplitBlend(N, ratio, a, b, q, mixed_mode="partial")
    raise ValueError(kind)


def _points(N, a, b, sub=4):
    X, W, Hv = [], [], []
    h = 1.0 / N
    for i, ps in enumerate(pieces(N, a, b)):
        for s0, s1, hv in ps:
            for k in range(sub):
                p0, p1 = s0 + (s1 - s0) * k / sub, s0 + (s1 - s0) * (k + 1) / sub
                X.append((i + p0 + (p1 - p0) * XG) * h)
                W.append((p1 - p0) * h * WG)
                Hv.append(np.full(XG.size, hv))
    return np.concatenate(X), np.concatenate(W), np.concatenate(Hv)


def _averages(N, ratio, a, b, u):
    X, W, Hv = _points(N, a, b, sub=8)
    rho = np.where(Hv, 1.0, 1.0 / ratio)
    cell = np.floor(X * N).astype(int) % N
    return np.bincount(cell, weights=rho * W * u(X), minlength=N) / np.bincount(cell, weights=rho * W, minlength=N)


def evaluate(N, ratio, nL, f1=0.4, f2=0.7, kind="blend"):
    """Errors (heavy part / light part, relative to max |u|) for a smooth u and a kinked u, the mean conservation,
    and the Gram lambda_max (M-normalised)."""
    a, b = geometry(N, nL, f1, f2)
    c0, c1 = b, a                                               # the light arc c0 -> c1
    Ll = (c1 - c0) % 1.0

    def in_light(x):
        return ((x - c0) % 1.0) < Ll

    funcs = {"smooth": lambda x: np.sin(2 * np.pi * x) + 0.3 * np.cos(2 * np.pi * x),
             "kink": lambda x: np.sin(2 * np.pi * x) + np.where(in_light(x), ((x - c0) % 1.0) * (Ll - (x - c0) % 1.0) / Ll, 0.0)}
    X, W, Hv = _points(N, a, b)
    out = {"nL": nL, "ratio": ratio, "kind": kind, "f1": f1, "f2": f2}
    try:
        for name, u in funcs.items():
            q = _averages(N, ratio, a, b, u)
            R = _recon(N, ratio, a, b, q, kind)
            e = np.abs(R(X) - u(X))
            scale = np.abs(u(X)).max()
            rho = np.where(Hv, 1.0, 1.0 / ratio)
            cell = np.floor(X * N).astype(int) % N
            means = np.bincount(cell, weights=rho * W * R(X), minlength=N) / np.bincount(cell, weights=rho * W, minlength=N)
            out[name] = {"heavy": float(e[Hv].max() / scale), "light": float(e[~Hv].max() / scale),
                         "mean_err": float(np.abs(means - q).max())}
        # Gram (rho-weighted), M-normalised
        rho = np.where(Hv, 1.0, 1.0 / ratio)
        cell = np.floor(X * N).astype(int) % N
        B = np.stack([_recon(N, ratio, a, b, np.eye(N)[j], kind)(X) for j in range(N)], axis=1)
        G = B.T @ ((rho * W)[:, None] * B)
        m = np.bincount(cell, weights=rho * W, minlength=N)
        s = 1.0 / np.sqrt(m)
        out["lmax"] = float(np.linalg.eigvalsh(G * s[:, None] * s[None, :])[-1])
    except ValueError as err:
        out["undefined"] = str(err)[:80]
    return out


def continuity(N, ratio, l0=0.3, l1=6.7, steps=200, eps=1e-7, kind="blend", f1=0.4):
    """Move the right interface of the light run continuously (the light thickness from l0 to l1 cells); at every
    position compare the reconstruction of the averages of the smooth u at x and x + eps: max |R(l + eps) - R(l)|
    / eps over the fine points (a jump shows up as a huge ratio), and the data-independent operator change for the unit
    averages of the cells near the moving interface."""
    h = 1.0 / N
    f0 = N // 2
    c0 = (f0 + 1 - f1) * h
    u = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x)  # noqa: E731
    xs = (np.arange(N * 16) + 0.5) / (N * 16)
    worst, at = 0.0, None
    rows = []
    for L in np.linspace(l0, l1, steps):
        vals = []
        for dl in (0.0, eps):
            c1 = c0 + (L + dl) * h
            a, b = c1 % 1.0, c0 % 1.0
            q = _averages(N, ratio, a, b, u)
            try:
                vals.append(_recon(N, ratio, a, b, q, kind)(xs))
            except ValueError:
                vals.append(None)
        if vals[0] is None or vals[1] is None:
            rows.append({"L": float(L), "rate": None})
            continue
        r = float(np.abs(vals[1] - vals[0]).max() / eps)
        rows.append({"L": float(L), "rate": r})
        if r > worst:
            worst, at = r, float(L)
    return {"ratio": ratio, "kind": kind, "max_rate": worst, "at_L": at, "rows": rows}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    N = 32
    jobs = [(N, r, nL, f1, f2, k) for r in (1.0, 1e3, 1e6) for nL in range(0, 9)
            for (f1, f2) in ((0.4, 0.7), (0.15, 0.2), (0.9, 0.85)) for k in ("base", "blend")]
    res = {"table": [], "continuity": []}
    with ProcessPoolExecutor(26) as ex:
        ft = [ex.submit(evaluate, *j) for j in jobs]
        fc = [ex.submit(continuity, N, r, kind=k) for r in (1.0, 1e3, 1e6) for k in ("base", "blend")]
        for f in ft:
            o = f.result()
            res["table"].append(o)
            print(o["kind"], o["ratio"], o["nL"], o["f1"], o["f2"], o.get("undefined") or
                  f"smooth {o['smooth']['heavy']:.1e}/{o['smooth']['light']:.1e} kink {o['kink']['heavy']:.1e}/"
                  f"{o['kink']['light']:.1e} mean {max(o['smooth']['mean_err'], o['kink']['mean_err']):.1e} lmax {o['lmax']:.2f}",
                  flush=True)
        for f in fc:
            o = f.result()
            res["continuity"].append(o)
            print("continuity", o["kind"], o["ratio"], "max rate %.3g at L = %s" % (o["max_rate"], o["at_L"]),
                  "undefined at", sum(1 for r in o["rows"] if r["rate"] is None), "of", len(o["rows"]), flush=True)
    json.dump(res, open("docs/figures/short_runs.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
