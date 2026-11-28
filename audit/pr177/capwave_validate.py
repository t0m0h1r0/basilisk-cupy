"""参照解の検証：capwave.c の物性（ρ = 1 と 1、μ = 0.0182571749236、σ = 1、深さ 1）で、Basilisk の Prosperetti の表と比べる。"""
import numpy as np, mpmath as mp
import audit.pr177.capwave_exact as e
e.h, e.rl, e.ru = mp.mpf(1), mp.mpf(1), mp.mpf(1)
e.ml = e.mu_ = mp.mpf("0.0182571749236")
tab = [tuple(map(float, l.split())) for l in open("cases/prosperetti_capillary.txt").read().splitlines()[1:]]
w0 = float(mp.sqrt(e.sig * e.k ** 3 / (e.rl + e.ru)))
err = 0.0
for tau, a in tab[5::40]:
    t = tau / w0
    v = float(mp.invertlaplace(e.eta_hat, mp.mpf(t), method="dehoog", degree=24).real)
    err = max(err, abs(abs(v) - abs(a)))
    print("tau %7.3f  table %+.6f  this %+.6f" % (tau, a, v))
print("max |difference| / a0 = %.2e" % (err / 0.01))
