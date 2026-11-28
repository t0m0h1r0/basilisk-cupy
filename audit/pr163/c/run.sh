#!/bin/bash
# 監修（C）：ノート118 の D1・D2 を、A のコードを使わず C 版の vof.h / conserving.h に独立に実装して確かめる。
#   BASILISK=<basilisk の src> PATH=$PATH:$BASILISK bash audit/pr163/c/run.sh
# 出力の列 00 / 10 / 01 / 11 は (D1, D2) の有無。格子 32^2、CFL 0.5、600 ステップの max|u - U|。
here=$(cd "$(dirname "$0")" && pwd); tmp=$(mktemp -d); cp "$here/a.c" "$tmp"; cd "$tmp"
cp $BASILISK/two-phase.h $BASILISK/two-phase-generic.h .
python3 - "$BASILISK" <<'PY'
import sys
B = sys.argv[1]
def sub(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
v = open(B + "/vof.h").read()
v = sub(v, "\tgf[] = vof_concentration_gradient_x (point, c, t);\n",
        "#if FIXD2\n\tgf[] = (t.inverse ? 1. - c[] : c[]) < 1. ? 0. : vof_concentration_gradient_x (point, c, t);\n"
        "#else\n\tgf[] = vof_concentration_gradient_x (point, c, t);\n#endif\n")
v = sub(v, "      else\n\ttflux[] = 0.;\n",
        "      else {\n#if FIXD1\n\tscalar pt = t.partner;\n\ttflux[] = t.rhok*(t[i] + pt[i])/(c[i]*(rho1 - rho2) + rho2)*cf1*uf.x[];\n"
        "#else\n\ttflux[] = 0.;\n#endif\n      }\n")
open("vof.h", "w").write("attribute {\n  scalar partner;\n  double rhok;\n}\nextern double rho1, rho2;\n" + v)
k = open(B + "/navier-stokes/conserving.h").read()
k = sub(k, "      double fc = clamp(f[],0,1);\n", "#if FIXD1\n      double fc = f[];\n#else\n      double fc = clamp(f[],0,1);\n#endif\n")
k = sub(k, "  foreach_dimension() {\n    q2.x.inverse = true;\n",
        "  foreach_dimension() {\n    q1.x.partner = q2.x; q2.x.partner = q1.x;\n    q1.x.rhok = rho1; q2.x.rhok = rho2;\n    q2.x.inverse = true;\n")
k = sub(k, "      u.x[] = (q1.x[] + q2.x[])/rho(f[]);\n",
        "#if FIXD1\n      u.x[] = (q1.x[] + q2.x[])/(f[]*(rho1 - rho2) + rho2);\n#else\n      u.x[] = (q1.x[] + q2.x[])/rho(f[]);\n#endif\n")
open("conserving_fix.h", "w").write(k)
PY
for v in "0 0" "1 0" "0 1" "1 1"; do set -- $v; qcc -O2 -disable-dimensions -I. -DFIXD1=$1 -DFIXD2=$2 a.c -o a_$1$2 -lm > qcc_$1$2.log 2>&1; done
for s in "0 1 0" "0 1 0.37" "1 1 0" "1 1 0.37"; do set -- $s; for r in 1e-3 1e-6; do
  printf "shape=%s rho2=%s U=(%s,%s): " $([ $1 = 0 ] && echo column || echo square) $r $2 $3
  for b in a_00 a_10 a_01 a_11; do printf "%s %s | " ${b#a_} "$(./$b $1 $r $2 $3 2>&1 | grep RESULT | awk '{print $3}')"; done; echo
done; done
