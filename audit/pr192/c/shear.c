/* 監修 PR #192：条件 6 の配置（非粘性の格子に平行なせん断流、重い層 1/4<=y<3/4、U=(0,1)、密度比 1000、N=64、CFL 0.5、t=1）。
   D1 + D2 限定（θ）は C 版の vof.h / conserving.h に監修が独立に入れたもの（../pr188/c/run.sh と同じ書き換え）。 */
#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "conserving_fix.h"
int NN = 64;
int main (int argc, char * argv[]) {
  if (argc > 1) NN = atoi (argv[1]);
  periodic (right); periodic (top); init_grid (NN);
  rho1 = 1.; rho2 = 1e-3; TOLERANCE = 1e-10; CFL = 0.5;
  run();
}
event init (t = 0) {
  double k = 4.*pi;
  foreach() {
    f[] = (y > 0.25 && y < 0.75);
    double y0 = y - Delta/2.;
    u.x[] = 0.5*(cos (k*(y0 - 0.25)) - cos (k*(y0 + Delta - 0.25)))/(k*Delta);
    u.y[] = 1.;
  }
}
event end (t = 1) {
  fprintf (stderr, "i=%d t=%.17g dt=%g\n", i, t, dt);
  foreach() fprintf (stdout, "%d %d %.17g %.17g %.17g\n", point.i - 2, point.j - 2, u.x[], u.y[], f[]);
  exit (0);
}
event hist (i++) {
  if (i % 20 == 0 || i < 4) {
    double ux = 0, uy = 0;
    foreach_face(x, reduction(max:ux)) ux = max (ux, fabs (uf.x[]));
    foreach_face(y, reduction(max:uy)) uy = max (uy, fabs (uf.y[]));
    fprintf (stderr, "hist i=%d t=%.6g dt=%.8g max|ufx|=%.6g max|ufy|=%.6g\n", i, t, dt, ux, uy);
  }
}
