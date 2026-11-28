# 監修 C-1：PR #164（ノート119、D2 の形の比較）の独立の検証
- `shear.py`：密度比 1000 の厳密解（`oblique_shear_reference.py`、C-3）での精度と、限定の適用が切った面の数。
- `threshold.py`：固定の CFL と界面の位置の走査。`theta.py`：利得の閾値 θ への依存。
- C 版への独立の実装：`../pr163/c/run.sh` の vof.h に、濃度の流束の直前で「利得 > 残る質量なら勾配を 0」を足した版（下の差分）。
```
double gi_ = gf[i];
{ double a_ = fabs (un), out_ = min (cf1*a_, max (ci, 0.)), ro_ = t.inverse ? rho1 : rho2;
  double rem_ = t.rhok*max (ci - out_, 0.) + ro_*max (1. - ci, 0.);
  if (t.rhok*out_*(1. - a_)/2. > rem_) gi_ = 0.; }
double ff = t[i]/ci + s*min(1., 1. - s*un)*gi_*Delta/2.;     // D1（相が無いセルは混合の値）と併用
```
実行は `PYTHONPATH=. python audit/pr164/<file>`（PR #164 の版の cases が要る）。
