"""Rising bubble (14c) at N = 128 (128 x 256) on the GPU: CLS (ab2, xi-SDF every 10) and VOF, to t = 0.03 s
(docs/notes/63); ``--variants imex-vof imex-vof-gfm``: the VOF on the IMEX stepper (docs/notes/64, 65).

    python -m cases.phase3_bubble128 [--backend cupy]
"""

from __future__ import annotations

import argparse
import time
import warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cupy")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--variants", nargs="*", default=["cls", "vof"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from cases.paper_benchmarks import run

    ap_variants = {"cls": dict(transport="ab2", reinit_every=10, reinit_method="xi_sdf"), "vof": dict(method="vof"),
                   "imex-vof": dict(interface="vof", transport="ab2"),
                   "imex-vof-gfm": dict(interface="vof-gfm", transport="ab2")}
    for name in args.variants:
        kw = ap_variants[name]
        t0 = time.time()
        r = run("bubble", n=args.n, t_end=0.03, samples=30, backend=args.backend, **kw)
        s = r["series"]
        at = s[-1]
        print(f"{name} N {args.n}: t {at['t']:.4f} yc {1e3 * at['yc']:.3f} mm mean rise "
              f"{(at['yc'] - s[0]['yc']) / at['t']:.4f} vc {at['vc']:.4f} D {at['D']:.3f} steps {r['steps']} "
              f"wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
