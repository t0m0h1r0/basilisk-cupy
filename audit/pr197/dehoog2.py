import importlib.util, sys
import mpmath as mp
spec = importlib.util.spec_from_file_location("c", "audit/pr197/capwave_exact.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
for dps in (50, 100):
    for t in (1.2, 2.5, 3.0, 4.0):
        out = []
        for name, kw in (("dh24", dict(method="dehoog", degree=24)), ("dh48", dict(method="dehoog", degree=48)), ("dh96", dict(method="dehoog", degree=96)), ("talbot", dict(method="talbot")), ("talbot%d" % dps, dict(method="talbot", degree=dps))):
            mp.mp.dps = dps
            out.append((name, float(mp.invertlaplace(c.eta_hat, mp.mpf(t), **kw).real)))
        print("dps=%d t=%.1f " % (dps, t) + "  ".join("%s %.6e" % x for x in out), flush=True)
