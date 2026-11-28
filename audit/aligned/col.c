/* audit: a face-aligned column (0.25 < x < 0.5) carried by the uniform flow (1, 0), two-phase.h + conserving.h,
   density ratio 1/RHO2, periodic. Prints max|u - U| (the exact solution is the uniform flow). */
#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "navier-stokes/conserving.h"
double RHO2 = 1e-3, tol = 1e-4, emax = 0., cfl_ = 0.5; int argc_ = 0;
int main (int argc, char * argv[]) {
  periodic (right); periodic (top);
  init_grid (32);
  if (argc > 1) RHO2 = atof (argv[1]);
  if (argc > 2) tol = atof (argv[2]);
  argc_ = argc; if (argc > 3) cfl_ = atof (argv[3]);
  rho1 = 1.; rho2 = RHO2; TOLERANCE = tol; CFL = argc > 3 ? atof (argv[3]) : 0.5;
  run();
}
event init (t = 0) {
  if (argc_ > 3) CFL = cfl_;
  foreach() { f[] = (x > 0.25 && x < 0.5); u.x[] = 1.; u.y[] = 0.; }
}
event check (i++; i <= 600) {
  double e = 0.;
  foreach (reduction(max:e)) {
    double d = fabs (u.x[] - 1.); if (d > e) e = d;
    d = fabs (u.y[]); if (d > e) e = d;
  }
  if (e > emax) emax = e;
  if (i % 100 == 0 || e > 10.)
    fprintf (stderr, "RHO2 %g tol %g i %d dt %g err %.3e max-so-far %.3e mgp %d resa %.2e\n", RHO2, tol, i, dt, e, emax, mgp.i, mgp.resa);
  if (e > 10.) exit (0);
}
