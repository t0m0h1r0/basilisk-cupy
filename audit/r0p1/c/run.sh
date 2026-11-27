#!/bin/bash
# 監修（C）：C 版の Basilisk で、conserving.h の q1・q2 の境界条件の上書き（boundary_q1_x）が周期の方向でも働くかを確かめる。
#   原典の conserving.h と、上書きの行を除いた版（q1・q2 は既定の周期の境界条件のまま）を、quadtree と multigrid で比べる。
#   BASILISK=<basilisk の src> PATH=$PATH:$BASILISK bash audit/r0p1/c/run.sh
here=$(cd "$(dirname "$0")" && pwd); tmp=$(mktemp -d); cd "$tmp"
cp "$here/../../../tools/basilisk/tg_conserving.c" tg_orig.c
python3 - "$BASILISK" <<'PY'
import sys
src = open(sys.argv[1] + "/navier-stokes/conserving.h").read()
old = """  for (int i = 0; i < nboundary; i++)
    foreach_dimension() {
      q1.x.boundary[i] = boundary_q1_x;
      q2.x.boundary[i] = boundary_q2_x;
    }
"""
assert old in src
open("conserving_periodic.h", "w").write(src.replace(old, "  /* audit: q1, q2 keep the default (periodic) boundary conditions */\n"))
t = open("tg_orig.c").read()
open("tg_patch.c", "w").write(t.replace('#include "navier-stokes/conserving.h"', '#include "conserving_periodic.h"'))
PY
for b in tg_orig tg_patch; do
  (echo '#include "grid/multigrid.h"'; cat $b.c) > ${b}_mg.c
  qcc -O2 -DCONSERVING $b.c -o $b -lm > /dev/null 2>&1
  qcc -O2 -DCONSERVING ${b}_mg.c -o ${b}_mg -lm > /dev/null 2>&1
done
for n in 32 64 128; do for b in tg_orig tg_patch tg_orig_mg tg_patch_mg; do echo -n "$b "; ./$b $n 2>&1 | grep "max|u"; done; done
