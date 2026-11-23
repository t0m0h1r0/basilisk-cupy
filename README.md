# basilisk-cupy

Python/CuPy port of the [Basilisk](http://basilisk.fr) flow solver, structured so
that every numerical ingredient (spatial operators, linear solvers, equation
terms, pressure projection, interface method, time stepping) can be swapped
for an alternative and validated against the Basilisk-compatible baseline.
