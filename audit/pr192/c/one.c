/* D2 が切る 1 ステップの比較：重い層の下の界面のセルの割合を 0.033、一様 u_y=1、せん断 u_x、Δt=5e-4（C=0.032）。 */
#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "conserving_fix.h"
int main () {
  periodic (right); periodic (top); init_grid (64);
  rho1 = 1.; rho2 = 1e-3; TOLERANCE = 1e-12; CFL = 0.5; DT = 0.007;
  run();
}
event init (t = 0) {
  double k = 4.*pi, h = 1./64, lo = 0.25 + (1. - 0.033)*h, hi = 0.75 + (1. - 0.033)*h;
  foreach() {
    f[] = clamp ((min (y + Delta/2., hi) - max (y - Delta/2., lo))/Delta, 0., 1.);
    double y0 = y - Delta/2.;
    u.x[] = 0.5*(cos (k*(y0 - 0.25)) - cos (k*(y0 + Delta - 0.25)))/(k*Delta);
    u.y[] = 1.;
  }
}
event end (i = NSTEP) {
  fprintf (stderr, "i=%d t=%.17g dt=%.17g\n", i, t, dt);
  foreach() fprintf (stdout, "%d %d %.17g %.17g %.17g\n", point.i - 2, point.j - 2, u.x[], u.y[], f[]);
  exit (0);
}
