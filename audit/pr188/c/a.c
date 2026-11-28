/* audit: D1 / D2 of note 118 written independently into the C vof.h / conserving.h. Aligned shapes in a uniform flow. */
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "conserving_fix.h"
double RHO2 = 1e-3, Ux = 1., Uy = 0., emax = 0., CFLX = 0.5, OFF = 0.; int SHAPE = 0;
int main (int argc, char * argv[]) {
  periodic (right); periodic (top); init_grid (32);
  SHAPE = atoi (argv[1]); RHO2 = atof (argv[2]); Ux = atof (argv[3]); Uy = atof (argv[4]);
  if (argc > 5) CFLX = atof (argv[5]);
  if (argc > 6) OFF = atof (argv[6]);
  rho1 = 1.; rho2 = RHO2; TOLERANCE = 1e-10; CFL = CFLX;
  run();
}
event init (t = 0) {
  foreach() {
    if (SHAPE == 0)   // 帯。OFF セル分ずらす（部分セルを含む）
      f[] = clamp ((min (x + Delta/2., 0.5 + OFF*Delta) - max (x - Delta/2., 0.25 + OFF*Delta))/Delta, 0., 1.);
    else
      f[] = (x > 0.375 && x < 0.625 && y > 0.375 && y < 0.625);
    u.x[] = Ux; u.y[] = Uy;
  }
}
event check (i++; i <= 600) {
  double e = 0.;
  foreach (reduction(max:e)) {
    double d = max (fabs (u.x[] - Ux), fabs (u.y[] - Uy)); if (d > e) e = d;
  }
  if (e > emax) emax = e;
  if (i == 600 || e > 10.) { fprintf (stderr, "RESULT max %.2e last %.2e\n", emax, e); exit (0); }
}
