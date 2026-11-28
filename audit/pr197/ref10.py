import importlib.util, json, subprocess
import numpy as np, mpmath as mp
spec = importlib.util.spec_from_file_location("c", "audit/pr197/capwave_exact.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
d = json.loads(subprocess.check_output(["git", "show", "origin/r0-assets:docs/figures/capwave_exact_10periods.json"]))
t, a = np.array(d["t"]), np.array(d["a"])
worst = 0.0
for k in list(range(0, len(t), 41)) + [len(t) - 1]:
    mp.mp.dps = 50
    v = float(mp.invertlaplace(c.eta_hat, mp.mpf(float(t[k])), method="dehoog", degree=48).real)
    worst = max(worst, abs(v - a[k]))
print("A table vs de Hoog 48 at %d times: max diff %.1e (a0 = 0.01)" % (len(range(0, len(t), 41)) + 1, worst))
p, damp, ext = c.measure(t, a)
print("read from A's table: period %.5f damping %.5f  (A: %.5f %.5f), points %d, t_end %.3f" % (p, damp, d["period"], d["damping"], len(t), t[-1]))
