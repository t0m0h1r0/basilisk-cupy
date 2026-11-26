"""Factors of the oscillating drop's excess damping at N = 64 (docs/notes/63): transport, height differences,
re-initialisation.

    python -m cases.phase3_drop_factors
"""


def run(transport, heights, reinit):
    import warnings
    warnings.filterwarnings("ignore")
    import cases.phase3_drop_bubble as m
    opts = dict(transport=transport, curvature_kw={"method": "heights", "heights_method": heights})
    if reinit:
        opts.update(reinit_every=10, reinit_method="xi_sdf")
    m.OPTS["_x"] = opts
    r = m.run("drop", "_x", 64)
    if r["series"][-1].get("failed"):
        return None
    return m.summarise_drop(r)


if __name__ == "__main__":
    from cases.parallel import pmap
    items = [dict(transport=t, heights=h, reinit=r) for t in ("ab2", "frozen") for h in ("fd2", "ccd") for r in (True, False)]

    def show(k, it, res):
        if res is None:
            print(it, "FAILED", flush=True)
        else:
            b, om, fr, vol, umax = res
            print(it, "beta %+.2f (%.2f x) omega %+.2f%% max|u| %.3f" % (b, b / -0.179428, 100 * (om / 59.554813 - 1), umax), flush=True)
    pmap(run, items, 8, on_result=show)
