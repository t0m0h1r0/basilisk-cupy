import importlib.util
import mpmath as mp
spec = importlib.util.spec_from_file_location("c", "audit/pr197/capwave_exact.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
for t in (0.5, 1.2, 2.0, 2.5, 3.0, 4.0):
    out = []
    for name, kw in (("dehoog24", dict(method="dehoog", degree=24)), ("dehoog48", dict(method="dehoog", degree=48)), ("talbot", dict(method="talbot")), ("talbot60", dict(method="talbot", degree=60))):
        mp.mp.dps = 30 if name != "talbot60" else 60
        out.append((name, float(mp.invertlaplace(c.eta_hat, mp.mpf(t), **kw).real)))
    print("t=%.1f " % t + "  ".join("%s %.6e" % x for x in out), flush=True)
