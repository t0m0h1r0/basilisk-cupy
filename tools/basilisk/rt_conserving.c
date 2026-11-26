/* Rayleigh--Taylor with momentum-conserving VOF (Basilisk src/test/rt.c on a uniform 64^2 grid, no adaptivity) for
   cases/conserving_compare.py. Writes fields-<t>.txt (x y f u.x u.y p) and the log (i t dt) on stderr.
     qcc -O2 -Wall rt_conserving.c -o rt_conserving -lm && ./rt_conserving 2> log */

#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "navier-stokes/conserving.h"

uf.n[left]   = 0;
p[left] = neumann(0.);
uf.n[right]  = 0;
p[right] = neumann(0.);

int main() {
  size (4);
  origin (-2, -2);
  init_grid (64);
  mu1 = mu2 = 0.00313;
  rho1 = 1.225, rho2 = 0.1694;
  a[] = {0,-9.81};
  DT = 5e-3;
  TOLERANCE = 1e-6;
  run();
}

event init (t = 0) {
  fraction (f, 0.05*cos (2.*pi*x) + y);
}

event logfile (i++) {
  fprintf (stderr, "%d %.12g %.12g\n", i, t, dt);
}

event fields (t = {0.1, 0.4}) {
  char name[80];
  sprintf (name, "fields-%g.txt", t);
  FILE * fp = fopen (name, "w");
  foreach()
    fprintf (fp, "%.12g %.12g %.12g %.12g %.12g %.12g\n", x, y, f[], u.x[], u.y[], p[]);
  fclose (fp);
}

event end (t = 0.4) {}
