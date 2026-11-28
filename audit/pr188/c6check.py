import cases.r0_aligned_free_stream as m
from cases.r0_combined import shear_inviscid
for th in (3.0, 1.0):
    m.THETA[0] = th
    for N in (64, 128):
        a = shear_inviscid("d1", N=N); b = shear_inviscid("d1+d2lim", N=N)
        print("theta %g N=%d  L1 %.3f  L2 %.3f  Linf %.3f  cut %d" % (th, N, b["L1"]/a["L1"], b["L2"]/a["L2"], b["Linf"]/a["Linf"], b["cut"]), flush=True)
