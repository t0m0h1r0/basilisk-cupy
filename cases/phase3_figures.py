"""Figures of docs/notes/58 and 59 (``docs/figures/phase3_benchmarks.png``): the paper's four benchmarks with
TwoPhaseIMEX (ab2 transport, xi-SDF every 10 steps) against Basilisk's VOF and the linear theories.

    python -m cases.phase3_figures [--jobs N] [--backend-rt cupy]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

RE = dict(transport="ab2", reinit_every=10, reinit_method="xi_sdf")
RUNS = {
    "cw-cls": ("capwave", dict(t_end=5 * 0.046742983863, samples=500, **RE)),
    "cw-vof": ("capwave", dict(t_end=5 * 0.046742983863, samples=500, method="vof")),
    "drop-cls": ("drop", dict(t_end=2 * 0.105460663082, samples=200, **RE)),
    "drop-cls-heights": ("drop", dict(t_end=2 * 0.105460663082, samples=200,
                                      curvature_kw={"method": "heights", "heights_method": "fd2"}, **RE)),
    "drop-vof": ("drop", dict(t_end=2 * 0.105460663082, samples=200, method="vof")),
    "bubble-cls": ("bubble", dict(t_end=0.06, samples=60, snapshots=(0.03, 0.06), **RE)),
    "bubble-vof": ("bubble", dict(t_end=0.06, samples=60, snapshots=(0.03, 0.06), method="vof")),
}


def _run(name: str, backend: str = "numpy"):
    import warnings

    warnings.filterwarnings("ignore")
    from cases.paper_benchmarks import run

    case, kw = RUNS[name] if name in RUNS else ("rt", dict(samples=140, snapshots=(7.0,), **(
        RE if name == "rt-cls" else dict(method="vof"))))
    r = run(case, backend=backend, **kw)
    r.pop("psi", None)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--backend-rt", default="cupy")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cases.parallel import pmap
    from cases.two_fluid_modes import mode

    names = list(RUNS)
    res = dict(zip(names, pmap(_run, [dict(name=n) for n in names], args.jobs)))
    res["rt-cls"] = _run("rt-cls", args.backend_rt)
    res["rt-vof"] = _run("rt-vof", args.backend_rt)
    import json

    for key in ("rt-cls", "rt-vof"):  # linear growth rate: A0 cosh(gamma t) fitted while A < 3 A0
        d = [x for x in res[key]["series"] if "amplitude" in x]
        t = np.array([x["t"] for x in d])
        a = np.array([x["amplitude"] for x in d])
        m = a / a[0] < 3
        gs = np.linspace(1.0, 2.5, 1501)
        err = [float(np.sum((a[0] * np.cosh(gg * t[m]) - a[m]) ** 2)) for gg in gs]
        res[key]["gamma"] = float(gs[int(np.argmin(err))])
        print(key, "gamma", res[key]["gamma"], "A(7)", a[-1], "steps", res[key]["steps"], "wall", round(res[key]["wall"]))
    with open(f"{args.out}/phase3_benchmarks.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "snapshots"} for k, v in res.items()}, f)

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    # capillary wave
    ax = axes[0, 0]
    s = mode(2 * math.pi / 0.01, 0.0728, 998.2, 1.002e-3, 1.204, 1.825e-5)
    for key, lab in (("cw-cls", "TwoPhaseIMEX (ab2, ξ-SDF/10)"), ("cw-vof", "VOF")):
        d = [x for x in res[key]["series"] if "amplitude" in x]
        t = np.array([x["t"] for x in d])
        ax.plot(t, 1e3 * np.array([x["amplitude"] for x in d]), lw=1, label=lab)
    a0 = 1e3 * res["cw-cls"]["series"][0]["amplitude"]
    ax.plot(t, a0 * np.exp(s.real * t) * np.cos(s.imag * t), "k--", lw=0.8, label="normal mode")
    ax.set_title("capillary wave (14a), 32²: amplitude [mm]")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=7)
    # drop
    ax = axes[0, 1]
    w0 = 59.578473371
    for key, lab in (("drop-cls", "CLS, ψ curvature"), ("drop-cls-heights", "CLS, height curvature"),
                     ("drop-vof", "VOF")):
        d = [x for x in res[key]["series"] if "D" in x]
        t = np.array([x["t"] for x in d])
        ax.plot(t, [x["D"] for x in d], lw=1, label=lab)
    ax.plot(t, 0.1 * np.cos(w0 * t), "k--", lw=0.8, label="Rayleigh–Lamb (inviscid)")
    ax.set_title("oscillating drop (14b), 32²: D = (a − b)/(a + b)")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=7)
    # bubble rise
    ax = axes[0, 2]
    for key, lab in (("bubble-cls", "CLS"), ("bubble-vof", "VOF")):
        d = [x for x in res[key]["series"] if "yc" in x]
        ax.plot([x["t"] for x in d], [1e3 * x["yc"] for x in d], lw=1, label=lab)
    ax.plot([0.03], [7.350], "r*", ms=10, label="paper, t = 0.03 s")
    ax.axvline(0.04, color="gray", lw=0.6, ls=":")
    ax.set_title("rising bubble (14c), 32×64: centroid [mm]")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=7)
    # RT amplitude
    ax = axes[0, 3]
    g = mode(4 * math.pi, 0.040, 950.0, 5e-3, 1000.0, 1e-3, 9.81).real
    for key, lab in (("rt-cls", "CLS"), ("rt-vof", "VOF")):
        d = [x for x in res[key]["series"] if "amplitude" in x]
        ax.semilogy([x["t"] for x in d], [abs(x["amplitude"]) for x in d], lw=1, label=lab)
    tt = np.linspace(0, 3, 50)
    ax.semilogy(tt, 0.01 * np.cosh(g * tt), "k--", lw=0.8, label=f"linear, γ = {g:.3f}")
    ax.set_title("Rayleigh–Taylor (14d), 128×512: amplitude")
    ax.set_xlabel("t")
    ax.legend(fontsize=7)
    # snapshots: bubble at 0.03 / 0.06 (CLS, VOF), RT at 7 (CLS, VOF)
    for ax, (key, ts, ext, title) in zip(axes[1], (
            ("bubble-cls", 0.06, (0, 10, 0, 20), "bubble CLS, t = 0.03 (lines) / 0.06 s"),
            ("bubble-vof", 0.06, (0, 10, 0, 20), "bubble VOF, t = 0.03 (lines) / 0.06 s"),
            ("rt-cls", 7.0, (0, 0.5, 0, 2.0), "RT CLS, t = 7"),
            ("rt-vof", 7.0, (0, 0.5, 0, 2.0), "RT VOF, t = 7"))):
        snaps = res[key].get("snapshots", {})
        if snaps:
            k = min(snaps, key=lambda v: abs(float(v) - ts))
            ax.imshow(np.array(snaps[k]).T, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1, aspect="equal")
            k3 = [v for v in snaps if abs(float(v) - 0.03) < 1e-6]
            if k3:
                q = np.array(snaps[k3[0]])
                nx, ny = q.shape
                X, Y = np.meshgrid((np.arange(nx) + 0.5) * ext[1] / nx, (np.arange(ny) + 0.5) * ext[3] / ny, indexing="ij")
                ax.contour(X, Y, q, [0.5], colors="r", linewidths=0.8)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.out}/phase3_benchmarks.png", dpi=130)
    print("wrote", f"{args.out}/phase3_benchmarks.png")


if __name__ == "__main__":
    main()
