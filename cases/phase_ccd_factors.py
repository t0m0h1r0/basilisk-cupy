"""Factors of the capillary-wave damping with ``faces="phase-ccd"`` (docs/notes/63): face operator, CLS divergence
and crossing-jump interpolation, on the paper's capillary wave (water--air, 32^2, ab2 transport, xi-SDF every 10).

    python -m cases.phase_ccd_factors
"""


def run(faces, div, interp):
    import warnings
    warnings.filterwarnings("ignore")
    from cases.phase3_capwave import T, fit
    from cases.paper_benchmarks import run as bench
    r = bench("capwave", n=32, t_end=5 * T, samples=500, faces=faces, transport="ab2", reinit_every=10,
              reinit_method="xi_sdf", cls_divergence=div, jump_interp=interp)
    s = [d for d in r["series"] if "amplitude" in d]
    return fit([d["t"] for d in s], [d["amplitude"] for d in s])


if __name__ == "__main__":
    from cases.parallel import pmap
    items = [dict(faces=f, div=d, interp=i) for f in ("fd2", "phase-ccd") for d in ("fccd", "full")
             for i in ("linear", "hermite")]
    for it, (b, om, res) in zip(items, pmap(run, items, 8)):
        print(it, "beta %+.2f (%.2f x) omega %+.2f%%" % (b, b / -0.803, 100 * (om / 134.3429 - 1)), flush=True)
