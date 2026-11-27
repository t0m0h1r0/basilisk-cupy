"""docs/notes/113: the figure and the tables of cases/r0p1_unsteady_viscous.py (docs/figures/r0p1_unsteady_viscous.json).

    python -m cases.r0p1_unsteady_viscous_plot           # docs/figures/r0p1_unsteady_viscous.png and the tables (stdout)
"""

from __future__ import annotations

import json
import math

import numpy as np

from cases.r0p1_unsteady_viscous import CONFIGS, GRIDS, INVISCID, NU, T_ENDS, k2, slope

JSON = "docs/figures/r0p1_unsteady_viscous.json"
FIXED = {"plain": "plain+visc+cn", "conserving": "cons/ab+visc+cn"}


def load():
    with open(JSON) as fh:
        runs = json.load(fh)["runs"]
    return {(r["config"], r["flow"], r["boost"], r["nu"], r["n"], r["cfl"]): r for r in runs}


def err(r, t, key="max"):
    for e in r["errors"]:
        if abs(e["t"] - t) < 1e-12:
            return e[key]
    return float("nan")


def ladder(R):
    print("\n## ladder (CFL 0.5, t = 0.5, max |u - u_exact|: N = 64 / 128 / 256, last ratio)")
    for flow in ("tg", "asym"):
        for boost in (False, True):
            for nu in (0.0, NU[flow]):
                if nu == 0.0 and not boost:
                    continue
                print("\n### %s, boost %s, nu %g" % (flow, boost, nu))
                for name in (INVISCID if nu == 0.0 else CONFIGS):
                    es = [err(R[(name, flow, boost, nu, n, 0.5)], 0.5) for n in (64, 128, 256)]
                    print("| %-22s | %.2e / %.2e / %.2e | %.2f |" % (name, es[0], es[1], es[2], es[1] / es[2]))


def convention(R):
    print("\n## convention: 13 grids, CFL 0.5, boost, viscous. slopes all / N >= 96")
    out = {}
    for flow in ("tg", "asym"):
        for name in ("plain", "plain+visc+cn", "cons", "cons/ab+visc+cn", "cons/ab+lag+cn", "cons/pred+visc+cn"):
            cells = []
            for t in T_ENDS:
                for key in ("max", "rms"):
                    es = [err(R[(name, flow, True, NU[flow], n, 0.5)], t, key) for n in GRIDS]
                    fine = [i for i, n in enumerate(GRIDS) if n >= 96]
                    a, b = slope(GRIDS, es), slope([GRIDS[i] for i in fine], [es[i] for i in fine])
                    out[(flow, name, t, key)] = (a, b)
                    cells.append("%.2f / %.2f" % (a, b))
            print("| %s | %-18s | %s |" % (flow, name, " | ".join(cells)))
    return out


def time_step(R):
    print("\n## fixed grid (asym, boost, viscous): max error at t = 0.5 for CFL 0.5 / 0.25 / 0.125 / 0.0625")
    for name in ("plain", "plain+visc+cn", "cons", "cons/ab+visc+cn", "cons/ab+lag+cn"):
        for n in (64, 128):
            rs = [R[(name, "asym", True, NU["asym"], n, c)] for c in (0.5, 0.25, 0.125, 0.0625)]
            print("| %-18s | %3d | %s | c1: %s | c2: %s |" % (
                name, n, " / ".join("%.3e" % err(r, 0.5) for r in rs),
                " / ".join("%+.2e" % err(r, 0.5, "c1") for r in rs), " / ".join("%+.2e" % err(r, 0.5, "c2") for r in rs)))


def projections(R):
    print("\n## projections against the fixed configuration, (c - c_fixed) / sum dt^2 at t = 0.5 (boost, viscous)")
    for flow in ("tg", "asym"):
        lam = NU[flow] * k2(flow)
        print("\n### %s: lambda^2 / 2 = %.4f, lambda / 2 = %.4f" % (flow, lam * lam / 2, lam / 2))
        for name, opts in CONFIGS.items():
            if name.endswith("stale") or opts.get("level", "n") not in ("n", "ab") or name in FIXED.values():
                continue
            if opts["variant"] == "conserving" and opts.get("level") != "ab":
                continue                                          # E1 has no closed form: only the /ab rows
            cells = []
            for n in (64, 128, 256):
                r, f = R[(name, flow, True, NU[flow], n, 0.5)], R[(FIXED[opts["variant"]], flow, True, NU[flow], n, 0.5)]
                s = r["errors"][-1]["sum_dt2"]
                cells.append("%+.3f, %+.3f" % ((err(r, 0.5, "c1") - err(f, 0.5, "c1")) / s,
                                               (err(r, 0.5, "c2") - err(f, 0.5, "c2")) / s))
            print("| %-22s | %s |" % (name, " | ".join(cells)))
    print("\n## no boost: (c1 - c1_fixed) / sum dt^2 (prediction lambda^2 / 2 for backward Euler, 0 for cn)")
    for flow in ("tg", "asym"):
        for name in ("plain", "plain+visc", "plain+cn", "cons", "cons+visc", "cons+cn", "cons/ab", "cons/ab+cn"):
            v = CONFIGS[name]["variant"]
            cells = []
            for n in (64, 128, 256):
                r, f = R[(name, flow, False, NU[flow], n, 0.5)], R[(FIXED[v], flow, False, NU[flow], n, 0.5)]
                cells.append("%+.3f" % ((err(r, 0.5, "c1") - err(f, 0.5, "c1")) / r["errors"][-1]["sum_dt2"]))
            print("| %s | %-12s | %s |" % (flow, name, " | ".join(cells)))


def plot(R):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    style = {"plain": ("C7", "o", "--"), "plain+visc+cn": ("C7", "o", "-"), "cons": ("C3", "s", "--"),
             "cons/ab+visc+cn": ("C0", "s", "-"), "cons/ab+lag+cn": ("C2", "^", "-"), "cons/pred+visc+cn": ("C1", "v", "-")}
    for ax, flow in zip(axes[:2], ("tg", "asym")):
        for name, (c, m, ls) in style.items():
            es = [err(R[(name, flow, True, NU[flow], n, 0.5)], 0.5) for n in GRIDS]
            ax.loglog(GRIDS, es, color=c, marker=m, ls=ls, ms=4, label=name)
        g = np.array(GRIDS, float)
        e0 = err(R[("cons/ab+visc+cn", flow, True, NU[flow], 32, 0.5)], 0.5)
        ax.loglog(g, 0.5 * e0 * (32 / g) ** 2, "k:", lw=0.8, label="slope 2")
        ax.loglog(g, 2.0 * e0 * (32 / g), "k-.", lw=0.8, label="slope 1")
        ax.set_title("%s + boost, nu k^2 = 0.79: max |u - u_exact| at t = 0.5, CFL 0.5" % flow, fontsize=9)
        ax.set_xlabel("N")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    ax = axes[2]
    for name, (c, m, ls) in style.items():
        if name == "cons/pred+visc+cn":
            continue
        for n, alpha in ((64, 0.45), (128, 1.0)):
            cf = (0.5, 0.25, 0.125, 0.0625)
            es = [err(R[(name, "asym", True, NU["asym"], n, x)], 0.5) for x in cf]
            ax.loglog(cf, es, color=c, marker=m, ls=ls, ms=4, alpha=alpha, label="%s, N = %d" % (name, n))
    ax.set_title("asym + boost: fixed grid, the time step alone", fontsize=9)
    ax.set_xlabel("CFL")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig("docs/figures/r0p1_unsteady_viscous.png", dpi=130)


def main() -> None:
    R = load()
    ladder(R)
    convention(R)
    time_step(R)
    projections(R)
    plot(R)
    assert math.isfinite(1.0)


if __name__ == "__main__":
    main()
