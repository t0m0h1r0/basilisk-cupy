#!/bin/bash
# 監修（C）PR #188：D1 + D2 限定（利得 > θ で切る）を A のコードを使わず C 版に独立に実装。θ = 1・3、D1 だけ（θ = 無限大）。
#   BASILISK=<basilisk の src> PATH=$PATH:$BASILISK bash audit/pr188/c/run.sh <作業ディレクトリ>
here=$(cd "$(dirname "$0")" && pwd); tmp=$1; mkdir -p "$tmp"; cp "$here/a.c" "$tmp"; cd "$tmp"
cp $BASILISK/two-phase.h $BASILISK/two-phase-generic.h .
python3 - "$BASILISK" <<'PY'
import sys
B = sys.argv[1]
def sub(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
v = open(B + "/vof.h").read()
v = sub(v, "      else\n\ttflux[] = 0.;\n",
        "      else {\n\tscalar pt = t.partner;\n\ttflux[] = t.rhok*(t[i] + pt[i])/(c[i]*(rho1 - rho2) + rho2)*cf1*uf.x[];\n      }\n")
v = sub(v, "\tdouble ff = t[i]/ci + s*min(1., 1. - s*un)*gf[i]*Delta/2.;\n",
        "\tdouble gi_ = gf[i];\n\t{ double a_ = fabs (un), out_ = min (cf1*a_, max (ci, 0.)), ro_ = t.inverse ? rho1 : rho2;\n"
        "\t  double rem_ = t.rhok*max (ci - out_, 0.) + ro_*max (1. - ci, 0.);\n"
        "\t  if (t.rhok*out_*(1. - a_)/2. > THETA*rem_) gi_ = 0.; }\n"
        "\tdouble ff = t[i]/ci + s*min(1., 1. - s*un)*gi_*Delta/2.;\n")
open("vof.h", "w").write("attribute {\n  scalar partner;\n  double rhok;\n}\nextern double rho1, rho2;\n" + v)
k = open(B + "/navier-stokes/conserving.h").read()
k = sub(k, "      double fc = clamp(f[],0,1);\n", "      double fc = f[];\n")
k = sub(k, "  foreach_dimension() {\n    q2.x.inverse = true;\n",
        "  foreach_dimension() {\n    q1.x.partner = q2.x; q2.x.partner = q1.x;\n    q1.x.rhok = rho1; q2.x.rhok = rho2;\n    q2.x.inverse = true;\n")
k = sub(k, "      u.x[] = (q1.x[] + q2.x[])/rho(f[]);\n", "      u.x[] = (q1.x[] + q2.x[])/(f[]*(rho1 - rho2) + rho2);\n")
open("conserving_fix.h", "w").write(k)
PY
for t in 1 3 1e30; do qcc -O2 -disable-dimensions -I. -DTHETA=$t a.c -o a_$t -lm > qcc_$t.log 2>&1; ls a_$t >/dev/null || { tail qcc_$t.log; exit 1; }; done
