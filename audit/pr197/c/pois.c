/* 監修 PR #197：面の上の界面の二層 Poiseuille（ρ 1/1e-3、μ 1/(1/55)、体積力 F=1、x 周期、粘着壁、N=32、DT=0.5/N）。 */
#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
face vector av[];
u.t[bottom] = dirichlet(0); u.t[top] = dirichlet(0);
double OFF = 0., DTX = 0.5;
int main (int argc, char * argv[]) {
  if (argc > 1) OFF = atof (argv[1]);
  if (argc > 2) DTX = atof (argv[2]);
  periodic (right); init_grid (32);
  rho1 = 1.; rho2 = 1e-3; mu1 = 1.; mu2 = 1./55.; TOLERANCE = 1e-10; DT = DTX/32;
  run();
}
event init (t = 0) {
  a = av;
  double h = 1./32;
  foreach() f[] = clamp ((0.5 + OFF*h - (y - Delta/2.))/Delta, 0., 1.);
}
event acceleration (i++) {
  foreach_face(x) av.x[] = alpha.x[]/(fm.x[] + SEPS);
  foreach_face(y) av.y[] = 0.;
}
event mon (i++) {
  static double vmax = 0.; static int first = -1;
  double v = 0.; foreach (reduction(max:v)) v = max (v, fabs (u.y[]));
  if (v > vmax) vmax = v;
  if (first < 0 && v > 1e-8) first = i;
  if (t >= 2. - 1e-9) {
    double h = 1./32, df = 0;
    foreach (reduction(max:df)) df = max (df, fabs (f[] - clamp ((0.5 + OFF*h - (y - Delta/2.))/Delta, 0., 1.)));
    fprintf (stderr, "off=%g DT=%g/N: steps %d  uy max %.2e (first > 1e-8 at step %d)  f change %.2e\n", OFF, DTX, i, vmax, first, df);
    exit (0);
  }
}
event end (t = 2.) {}
