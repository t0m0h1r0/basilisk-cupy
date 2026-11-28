/* #199：C Basilisk の fraction()。直線 y < x + c（格子点を通る／通らない）と、正方形の隅。N=32。 */
#include "grid/multigrid.h"
#include "fractions.h"
int main () {
  init_grid (32);
  scalar f[];
  double cs[] = {0., -5.551115123125783e-17, 0.0137};
  for (int k = 0; k < 3; k++) {
    double c = cs[k];
    fraction (f, x + c - y);
    double emax = 0; int nbad = 0;
    foreach (serial) {
      /* 厳密：{ y > x + c } の面積（x + c - y < 0 が相の外） */
      double h = Delta, x0 = x - h/2, y0 = y - h/2, a = 0; int M = 4000;
      for (int m = 0; m < M; m++) { double xx = x0 + (m + 0.5)*h/M; double lo = xx + c; double len = lo - y0; a += (len < 0 ? 0 : len > h ? h : len)/M; }
      double ex = a/h;
      double d = fabs (f[] - ex); if (d > emax) emax = d; if (d > 1e-3) nbad++;
    }
    fprintf (stderr, "line c=%g: C max |f - exact| %.2e  cells>1e-3 %d\n", c, emax, nbad);
  }
  fraction (f, min (0.2 - fabs(x - 0.5123), 0.2 - fabs(y - 0.4871)));
  double emax = 0;
  foreach (serial) {
    double h = Delta;
    double ox = (min(x + h/2, 0.7123) - max(x - h/2, 0.3123))/h, oy = (min(y + h/2, 0.6871) - max(y - h/2, 0.2871))/h;
    double ex = (ox > 0 ? ox : 0)*(oy > 0 ? oy : 0);
    if (fabs (f[] - ex) > emax) emax = fabs (f[] - ex);
  }
  fprintf (stderr, "square: C max |f - exact| %.2e\n", emax);
}
