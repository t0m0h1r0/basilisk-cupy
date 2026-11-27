import sys, time
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("gs", "audit/r0p1/ghost_scope.py")
src = open("audit/r0p1/ghost_scope.py").read().split("NS = (32, 64, 128)")[0]
exec(src)
for variant, ghosts in (("plain", "library"), ("conserving-src", "library"), ("conserving-src", "self")):
    for n in (32, 64):
        t0 = time.time()
        print(variant, ghosts, n, "%.4e" % tg(variant, n, 0.5, False, ghosts)[0], "%.0fs" % (time.time() - t0), flush=True)
