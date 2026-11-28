/* #195: the audit's one.c with the unmodified conserving.h; dumps after the transport and at the end of each step */
#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "navier-stokes/conserving.h"
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
static void mydump (const char * tag, int it, double tt, double dtt) {
  char name[80]; sprintf (name, "%s_%d.txt", tag, it);
  FILE * fp = fopen (name, "w");
  foreach() fprintf (fp, "%d %d %.17g %.17g %.17g %.17g %.17g %.17g\n", point.i - 2, point.j - 2, u.x[], u.y[], f[], p[], g.x[], g.y[]);
  fclose (fp);
  sprintf (name, "%s_%d_uf.txt", tag, it);
  fp = fopen (name, "w");
  foreach_face(x) fprintf (fp, "x %d %d %.17g\n", point.i - 2, point.j - 2, uf.x[]);
  foreach_face(y) fprintf (fp, "y %d %d %.17g\n", point.i - 2, point.j - 2, uf.y[]);
  fclose (fp);
  fprintf (stderr, "%s i=%d t=%.17g dt=%.17g\n", tag, it, tt, dtt);
}
event tracer_advection (i++) { if (i < 3) mydump ("vof", i, t, dt); }
event end_timestep (i++) { if (i < 3) mydump ("end", i, t, dt); }
event stop (i = 3) { exit (0); }
