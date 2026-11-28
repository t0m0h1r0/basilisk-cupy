"""A-8 (#173), docs/notes/122: a verifier of the LOCAL budgets of the transport, from flux arrays only.

``verify(sweep)`` takes one directional sweep as plain arrays (no knowledge of the scheme that made them):
    axis, periodic         the sweep direction, the periodic axes
    un                     the face CFL numbers u dt / h on the n + 1 faces along ``axis``
    F                      phase 1's volume moved through each face (cell volumes; phase 2's is un - F)
    Fq[k]                  the momentum moved through each face by tracer k (q / h^2 per cell volume)
    ff[k], phase[k]        tracer k's face value (q / c of its phase) and its phase (0: phase 1, 1: phase 2)
    c0, c1                 phase 1's volume fraction before / after; q0[k], q1[k] the tracers before / after
    Sc, Sq[k]              the non-flux (dilation) terms of the update, per cell
and checks
    L1  the phase volume (mass / rho_k) budget per cell:  c1 - c0 - (F_lo - F_hi) - Sc = 0
    L2  the momentum budget per cell (transport only):    q1 - q0 - (Fq_lo - Fq_hi) - Sq = 0
    L3  shared faces: a periodic sweep's first and last faces carry the same fluxes; wall faces carry none
    L4  common fluxes: Fq = ff x (the phase's volume flux), on every face where the phase's volume moves
    L5  the dilation terms are consistent: Sq = (q0 / c_phase) Sc_phase (the momentum dilates with the volume)
and returns the largest residuals with the cell / face where each occurs. ``capture()`` collects the sweeps of
the library (or of a patched scheme) from cases/r0_aligned_free_stream.py's hooks; R2's CCD fluxes can be passed
to ``verify`` the same way.

    python -m cases.r0_local_budget      # docs/figures/r0_local_budget.json (a few minutes)
"""

from __future__ import annotations

import contextlib
import json
import os

import numpy as np


def _faces(a, axis, lo):
    """Faces 0..n-1 (lo) or 1..n (hi) along axis of a face array with n + 1 faces."""
    n = a.shape[axis] - 1
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(0, n) if lo else slice(1, n + 1)
    return a[tuple(idx)]


def _worst(r):
    r = np.asarray(r)
    if r.size == 0:
        return {"max": 0.0, "at": None}
    k = int(np.argmax(np.abs(r)))
    return {"max": float(np.abs(r).max()), "at": [int(v) for v in np.unravel_index(k, r.shape)]}


def verify(sw, scale_q=1.0):
    """The residuals of L1-L5 for one sweep (dict of arrays, see the module docstring). ``scale_q``: a momentum scale
    for relative residuals of L2 / L4 / L5 (e.g. rho1 U)."""
    ax, un, F = sw["axis"], np.asarray(sw["un"]), np.asarray(sw["F"])
    out = {}
    out["L1"] = _worst(sw["c1"] - sw["c0"] - (_faces(F, ax, True) - _faces(F, ax, False)) - sw["Sc"])
    out["L2"] = {}
    out["L4"] = {}
    out["L5"] = {}
    for k, Fq in enumerate(sw["Fq"]):
        Fq = np.asarray(Fq)
        out["L2"][k] = _worst((sw["q1"][k] - sw["q0"][k] - (_faces(Fq, ax, True) - _faces(Fq, ax, False))
                               - sw["Sq"][k]) / scale_q)
        Fk = F if sw["phase"][k] == 0 else un - F
        moved = np.abs(Fk) > 0
        out["L4"][k] = _worst(np.where(moved, Fq - np.asarray(sw["ff"][k]) * Fk, 0.0) / scale_q)
        out["L5"][k] = _worst(_l5(sw, k) / scale_q)
    l3 = []
    for a in [F] + [np.asarray(x) for x in sw["Fq"]]:
        first, last = np.take(a, 0, axis=ax), np.take(a, -1, axis=ax)
        l3.append(np.abs(first - last).max() if sw["periodic"][ax] else max(np.abs(first).max(), np.abs(last).max()))
    out["L3"] = {"max": float(max(l3)), "at": None}
    return out


def _l5(sw, k, mask=None):
    """Momentum dilation - (concentration) x (the phase's volume dilation). The concentration is the one the scheme
    declares for the step (``conc``: vof.h keeps the step's start in both sweeps), in the cells holding the phase."""
    Sck = sw["Sc"] if sw["phase"][k] == 0 else sw["Sdiv"] - sw["Sc"]
    r = sw["Sq"][k] - sw["conc"][k] * Sck
    return r if mask is None else np.where(mask, r, 0.0)


def from_library(rec, rho, start=None):
    """A captured library sweep (cases/r0_aligned_free_stream.CAPTURE) -> the verifier's arrays. Tracers 2 k and
    2 k + 1 are q1 and q2 of the k-th velocity component (TwoPhase's order)."""
    ax, dt, h = rec["axis"], rec["dt"], rec["h"]
    u = np.asarray(rec["u"])
    n = rec["c0"].shape[ax]
    div = _faces(u, ax, False) - _faces(u, ax, True)
    sw = {"axis": ax, "periodic": rec["periodic"], "un": u * dt / h, "F": np.asarray(rec["flux"]) * dt / h,
          "c0": rec["c0"], "c1": rec["c1"], "Sc": rec["cc"] * div * dt / h, "Sdiv": div * dt / h,
          "Fq": [np.asarray(t) * dt / h for t in rec["tfluxes"]],
          "ff": [np.asarray(f[0]) for f in rec["ffs"]], "phase": [1 if inv else 0 for inv in rec["inverse"]],
          "q0": rec["t0"], "q1": rec["t1"], "Sq": [np.asarray(t) * div * dt / h for t in rec["tcs"]]}
    # the step's start (the first sweep of the step): the concentration of each phase where it is present
    c_s, t_s = (rec["c0"], rec["t0"]) if start is None else (start["c0"], start["t0"])
    sw["conc"] = []
    for k, inv in enumerate(rec["inverse"]):
        ck = 1.0 - c_s if inv else c_s
        sw["conc"].append(np.where(ck > 1e-10, t_s[k] / np.where(ck > 1e-10, ck, 1.0), 0.0))
    assert n + 1 == u.shape[ax]
    return sw


@contextlib.contextmanager
def capture():
    from cases import r0_aligned_free_stream as m

    m.CAPTURE[0] = []
    try:
        yield m.CAPTURE[0]
    finally:
        m.CAPTURE[0] = None


def categories(rec):
    """Cell masks of the special cases (L5 of #173): dilation cells (cc = 1 and div != 0), c = 1/2 exactly, phases
    absent (0 < c <= 1e-10 or < 0, or the same for 1 - c), interfacial cells."""
    c = rec["c0"]
    return {"c=1/2": c == 0.5, "absent": ((c <= 1e-10) & (c != 0.0)) | ((1 - c <= 1e-10) & (c != 1.0)),
            "interfacial": (c > 0) & (c < 1)}


def summarise(sweeps, rho=(1.0, 1e-3), scale_q=1.0, lin=None):
    """The worst residuals of L1-L5 over the captured sweeps, L4 / L5 restricted to the special cells, D1's
    signature (faces where the phase's volume moves but its momentum flux is exactly zero), and L6: the mass the
    transport implies, rho1 c + rho2 (1 - c), against the density the velocity is recovered with (clamped unless
    measured from the density the solver used at the end of each step's transport, over rho2 (``lin`` is ignored:
    kept for old calls). Budgets at round-off do not mean that the face values are right or that a free stream is
    kept (audit on #173: a face value changed consistently in the flux and the update passes L1-L5; D2 is not seen
    either): L1-L6 check the bookkeeping of the fluxes, not their accuracy."""
    worst = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "L4": 0.0, "L5": 0.0, "L6": 0.0}
    zero_flux_faces = 0
    where = {}
    special = {}
    for s, rec in enumerate(sweeps):
        sw = from_library(rec, rho, start=sweeps[s - 1] if s % 2 else None)   # two sweeps per step
        v = verify(sw, scale_q)
        for key in ("L1", "L3"):
            if v[key]["max"] > worst[key]:
                worst[key], where[key] = v[key]["max"], {"sweep": s, "at": v[key]["at"]}
        for key in ("L2", "L4", "L5"):
            for k, r in v[key].items():
                if r["max"] > worst[key]:
                    worst[key], where[key] = r["max"], {"sweep": s, "tracer": k, "at": r["at"]}
        for k, Fq in enumerate(sw["Fq"]):
            Fk = sw["F"] if sw["phase"][k] == 0 else sw["un"] - sw["F"]
            zero_flux_faces += int(((np.abs(Fk) > 0) & (np.asarray(Fq) == 0.0) & (np.asarray(sw["ff"][k]) != 0.0)).sum())
        if s % 2 and "rho_used" in rec:
            # L6 (measured, audit on #173): the transported mass rho1 c + rho2 (1 - c) against the density the
            # solver actually recovered the velocity with: (q1 + q2) - rho_lin(c) u = (rho_used - rho_lin(c)) u
            c = rec["c1"]
            rl = rho[0] * c + rho[1] * (1 - c)
            worst["L6"] = max(worst["L6"], float(np.abs(rl - np.asarray(rec["rho_used"])).max() / rho[1]))
        # L5 on the special cells: the dilation consistency at c = 1/2 and in the absent cells
        for name, mask in categories(rec).items():
            for k in range(len(sw["Fq"])):
                r = np.abs(_l5(sw, k, mask)).max() / scale_q
                special[name] = max(special.get(name, 0.0), float(r))
    return {"worst": worst, "where": where, "L5_special": special, "sweeps": len(sweeps),
            "zero_momentum_flux_faces": zero_flux_faces}


# --- the configurations (#173) ------------------------------------------------------------------------------------
def _flow(f0, conf, steps=6, ratio=1e3, n=32, **tpkw):
    """A Taylor-Green flow + (0.5, 0.2) carrying f0 (periodic): the per-sweep dilation is not zero, so the dilation
    terms are exercised (a uniform flow cannot show #156)."""
    from basilisk_cupy import BoundaryConditions, CartesianGrid, CellField
    from basilisk_cupy.interface.two_phase import TwoPhase
    from basilisk_cupy.time.basilisk_centered import CenteredNS

    from cases.r0_combined import configuration

    with configuration("base-hooked" if conf == "base" else "full"), capture() as cap:
        grid = CartesianGrid((n, n))
        ns = CenteredNS(grid, periodic=(0, 1), tolerance=1e-10)
        TwoPhase(ns, CellField.from_interior(grid, f0(grid), BoundaryConditions.periodic(2)), rho1=1.0,
                 rho2=1.0 / ratio, conserving=True, **tpkw)
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        ns.u[0].interior = 0.5 + np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        ns.u[1].interior = 0.2 - np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        ns.start()
        for _ in range(steps):
            ns.step()
    return cap


def _run(config, conf):
    """The sweeps of one configuration; conf "base" (the library's code, through the hooks with every fix off:
    bit-identical) or "full" (A-3's configuration)."""
    from cases import r0_aligned_free_stream as m

    if config == "vortex":
        from cases.r0_aligned_free_stream import shape

        return _flow(lambda g: shape("drop", g), conf), 2.5
    if config == "aligned":
        from cases.r0_aligned_free_stream import shape

        return _flow(lambda g: shape("column", g), conf), 2.5
    if config == "half":
        from cases.r0_aligned_free_stream import shape

        return _flow(lambda g: shape("half", g), conf, steps=1), 2.5      # c = 1/2 exactly at the first step
    if config == "half, rule before #160":
        from cases.r0_aligned_free_stream import shape

        return _flow(lambda g: shape("half", g), conf, steps=1, conserving_vof_half=True), 2.5
    if config == "free-stream":
        sch = "base" if conf == "base" else "d1+d2lim"
        with capture() as cap:
            m.free_stream("column", sch, 1e3, U=(1.0, 0.37), steps=30)
        return cap, 1.0
    if config == "wall":
        from cases.r0_combined import configuration
        from cases.r0p2_pressure_momentum import hydrostatic

        with configuration("base-hooked" if conf == "base" else "full"), capture() as cap:
            hydrostatic("smooth", 26.565, "plain", 32, offset=0.1171, t_end=0.05, samples=(0.05,))   # it moves
        return cap, 1.0
    raise ValueError(config)


def main() -> None:
    res = {}
    for config in ("vortex", "aligned", "half", "half, rule before #160", "free-stream", "wall"):
        for conf in ("base", "full"):
            cap, scale = _run(config, conf)
            res["%s/%s" % (config, conf)] = summarise(cap, scale_q=scale, lin=conf == "full")
            r = res["%s/%s" % (config, conf)]
            print(config, conf, {k: "%.1e" % v for k, v in r["worst"].items()},
                  {k: "%.1e" % v for k, v in r["L5_special"].items()}, "zero-flux faces", r["zero_momentum_flux_faces"],
                  flush=True)
    with open(os.path.join("docs", "figures", "r0_local_budget.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
