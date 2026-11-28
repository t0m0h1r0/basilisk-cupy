import cases.r0_aligned_free_stream as m
for th in (3.0, 1e30):
    m.THETA[0] = th
    print("theta", th, "column ratio1e3 max_err %.2e" % m.free_stream("column", "d1+d2lim", 1e3, U=(1.0, 0.37))["max_err"], flush=True)
