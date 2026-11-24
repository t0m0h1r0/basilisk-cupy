/* Static drop for cases/static_drop_compare.py: R = 0.25, sigma = 1, mu = 0.1, rho 1 / 0.1, 64^2, walls.
   Writes fields-0.2.txt (x y f u.x u.y p) and the log (i t dt max|u|) on stderr.
     qcc -O2 -Wall static_drop.c -o static_drop -lm && ./static_drop 2> log */

#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "tension.h"

int main() {
  size (1.);
  origin (-0.5, -0.5);
  init_grid (64);
  rho1 = 1., rho2 = 0.1;
  mu1 = 0.1, mu2 = 0.1;
  f.sigma = 1.;
  TOLERANCE = 1e-8;
  run();
}

event init (t = 0) {
  fraction (f, sq(0.25) - sq(x) - sq(y));
}

event logfile (i++) {
  double umax = 0.;
  foreach (reduction(max:umax))
    if (fabs(u.x[]) > umax) umax = fabs(u.x[]);
  foreach (reduction(max:umax))
    if (fabs(u.y[]) > umax) umax = fabs(u.y[]);
  fprintf (stderr, "%d %g %g %g\n", i, t, dt, umax);
}

event fields (t = 0.2) {
  FILE * fp = fopen ("fields-0.2.txt", "w");
  foreach()
    fprintf (fp, "%.12g %.12g %.12g %.12g %.12g %.12g\n", x, y, f[], u.x[], u.y[], p[]);
  fclose (fp);
}

event end (t = 0.2) {}
