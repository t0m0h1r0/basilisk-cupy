"""発散の境目と、予測の影響の切り分け。c-cn：現行の運ぶ速度 + CN（源の項に粘性）。d1-save：現行の d1。
d1-noL：予測の L u^n だけ外す（d1v2-nopredL-addold）。pred：conserving_predict。"""
import sys; sys.path.insert(0, "audit/d179c")
from run import *
CFG["d1-noL"] = dict(level="d1v2-nopredL-addold", source="g+visc", visc="cn")

def k2(a):
    name, n, nu, cfl = a
    try:
        o = run(n, cfl, flow="mix", nu=nu, boost=True, t_ends=(0.5,), variant="conserving", **CFG[name])
    except FloatingPointError as ex:
        return "%-7s N=%-3d nu=%-4g cfl=%-5g: %s" % (name, n, nu, cfl, ex)
    if not o["errors"]:
        return "%-7s N=%-3d nu=%-4g cfl=%-5g: BLOW-UP %s" % (name, n, nu, cfl, str(o.get("failed", ""))[:80])
    e = o["errors"][-1]
    return "%-7s N=%-3d nu=%-4g cfl=%-5g kappa(nominal %.3g) steps %d: max err %.3e" % (name, n, nu, cfl, nu * cfl / n / 1.9 * n * n, e["steps"], e["max"])

if __name__ == "__main__":
    J = [(c, n, nu, 0.5) for c in ("d1", "d1-noL", "pred", "c-cn") for n, nu in ((64, 4.0), (128, 1.0), (128, 2.0), (128, 4.0), (64, 8.0))]
    J += [("d1", 128, 2.0, 0.25), ("d1", 128, 2.0, 0.125)]
    with mp.get_context("spawn").Pool(18, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(k2, J, chunksize=1):
            print(line, flush=True)
