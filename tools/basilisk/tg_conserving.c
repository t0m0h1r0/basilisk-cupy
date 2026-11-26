/* Inviscid Taylor--Green (a steady Euler flow) on the periodic unit square, uniform N^2 grid, with Basilisk's centred
   solver: plain (BCG advection) or with two-phase.h + conserving.h and f = 0 (one phase everywhere, no interface).
   Prints the max |u - u0| at t = 0.5 (cases/pure_advection.py, docs/notes/77).
     qcc -O2 -Wall [-DCONSERVING] tg_conserving.c -o tg -lm && ./tg 64 */

#include "navier-stokes/centered.h"
#if CONSERVING
#include "two-phase.h"
#include "navier-stokes/conserving.h"
#endif

int main (int argc, char * argv[]) {
  periodic (right);
  periodic (top);
  init_grid (argc > 1 ? atoi (argv[1]) : 64);
  TOLERANCE = 1e-8;
#if CONSERVING
  rho1 = rho2 = 1.;
#endif
  run();
}

event init (t = 0) {
#if CONSERVING
  foreach()
    f[] = 0.;
#endif
  foreach() {
    u.x[] = sin (2.*pi*x)*cos (2.*pi*y);
    u.y[] = - cos (2.*pi*x)*sin (2.*pi*y);
  }
}

event end (t = 0.5) {
  double e = 0.;
  foreach (reduction(max:e)) {
    double ex = u.x[] - sin (2.*pi*x)*cos (2.*pi*y), ey = u.y[] + cos (2.*pi*x)*sin (2.*pi*y);
    double d = sqrt (ex*ex + ey*ey);
    if (d > e) e = d;
  }
  fprintf (stderr, "N %d steps %d max|u - u0| %.6e\n", N, i, e);
}
