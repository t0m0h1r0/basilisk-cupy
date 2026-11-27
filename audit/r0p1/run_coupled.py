"""監修（C）：ノート112 の 4 節の主要な行を A のコードで再現する。"""
import sys
from cases.r0p1_local_error import coupled, periodic_momentum_ghosts
import basilisk_cupy; print(basilisk_cupy.__file__)
for variant in ("conserving", "conserving-src"):
    for fix in (False, True):
        row = []
        for n in (32, 64, 128):
            if fix:
                with periodic_momentum_ghosts():
                    row.append(coupled(n, 0.5, variant)["max"])
            else:
                row.append(coupled(n, 0.5, variant)["max"])
        print(variant, "ghost-fix" if fix else "library", " ".join("%.3e" % r for r in row), flush=True)
