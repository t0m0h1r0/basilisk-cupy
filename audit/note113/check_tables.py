"""監修（C）：ノート113 の表を、保存された JSON から A の作図コードを使わずに計算し直す。厳密解の記号計算による確認も。"""
import json, math
import numpy as np
import sympy as sp

# --- A 節：厳密解と圧力
x, y, t, nu, U1, U2 = sp.symbols("x y t nu U1 U2")
for psi0, kk in ((sp.sin(2 * sp.pi * x) * sp.sin(2 * sp.pi * y) / (2 * sp.pi), 8 * sp.pi ** 2),
                 (sp.sin(2 * sp.pi * (x + 2 * y)) + sp.Rational(3, 5) * sp.cos(2 * sp.pi * (2 * x - y)) + sp.Rational(2, 5) * sp.sin(2 * sp.pi * (2 * x + y) + 1), 20 * sp.pi ** 2)):
    ps = sp.exp(-nu * kk * t) * psi0.subs({x: x - U1 * t, y: y - U2 * t}, simultaneous=True)
    u, v = U1 + sp.diff(ps, y), U2 - sp.diff(ps, x)
    p = -((u - U1) ** 2 + (v - U2) ** 2 + kk * ps ** 2) / 2
    rx = sp.diff(u, t) + u * sp.diff(u, x) + v * sp.diff(u, y) + sp.diff(p, x) - nu * (sp.diff(u, x, 2) + sp.diff(u, y, 2))
    ry = sp.diff(v, t) + u * sp.diff(v, x) + v * sp.diff(v, y) + sp.diff(p, y) - nu * (sp.diff(v, x, 2) + sp.diff(v, y, 2))
    f = sp.lambdify((x, y, t, nu, U1, U2), [rx, ry, sp.diff(u, x) + sp.diff(v, y)])
    rng = np.random.default_rng(0)
    print("NS residual (random points):", max(abs(c) for _ in range(20) for c in f(*rng.uniform(-1, 1, 3), 0.01, 1.0, 0.37)))

# --- 表
R = json.load(open("docs/figures/r0p1_unsteady_viscous.json"))["runs"]
def get(cfg, flow, nu, boost, cfl=0.5):
    return sorted([r for r in R if r["config"] == cfg and r["flow"] == flow and abs(r["nu"] - nu) < 1e-12 and r["boost"] == boost and r["cfl"] == cfl], key=lambda r: r["n"])
NU = {"tg": 0.01, "asym": 0.004}
sl = lambda ns, es: -np.polyfit(np.log(ns), np.log(es), 1)[0]
print("\n3 節：13 格子の傾き（全体 / N>=96）、t = 0.125, 0.25, 0.5 の max と rms")
for flow in ("tg", "asym"):
    for cfg in ("plain", "plain+visc+cn", "cons", "cons/ab+visc+cn", "cons/ab+lag+cn", "cons/pred+visc+cn"):
        rs = get(cfg, flow, NU[flow], True)
        ns = np.array([r["n"] for r in rs], float)
        out = []
        for k in range(3):
            for key in ("max", "rms"):
                es = np.array([r["errors"][k][key] for r in rs])
                out.append("%.2f/%.2f" % (sl(ns, es), sl(ns[ns >= 96], es[ns >= 96])))
        print("%-5s %-18s n=%d  %s" % (flow, cfg, len(rs), "  ".join(out)))
print("\n2 節：全てを直した構成（cons/ab+visc+cn、plain は plain+visc+cn）との差の射影 / ΣΔt²（N = 64/128/256）")
for flow in ("tg", "asym"):
    for cfg, ref in (("cons/ab", "cons/ab+visc+cn"), ("cons/ab+cn", "cons/ab+visc+cn"), ("cons/ab+visc", "cons/ab+visc+cn"),
                     ("cons/ab+visc+cn-naive", "cons/ab+visc+cn"), ("cons/ab+lag+cn", "cons/ab+visc+cn"),
                     ("plain", "plain+visc+cn"), ("plain+cn", "plain+visc+cn"), ("plain+visc", "plain+visc+cn")):
        a = {r["n"]: r for r in get(cfg, flow, NU[flow], True)}; b = {r["n"]: r for r in get(ref, flow, NU[flow], True)}
        s = []
        for n in (64, 128, 256):
            ea, eb = a[n]["errors"][-1], b[n]["errors"][-1]
            s.append("%+.3f,%+.3f" % ((ea["c1"] - eb["c1"]) / ea["sum_dt2"], (ea["c2"] - eb["c2"]) / ea["sum_dt2"]))
        print("%-5s %-22s %s" % (flow, cfg, " / ".join(s)))
