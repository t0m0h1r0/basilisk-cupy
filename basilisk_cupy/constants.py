"""Constants with Basilisk's values.

Basilisk's ``common.h`` defines ``pi`` as ``3.14159265358979``, about 3.2e-15 below the
double closest to pi. Ported test cases use it where Basilisk's source writes ``pi``:
the round-off it causes decides, for instance, whether a cell crossed near a vertex gets a
volume fraction of exactly 1 or 1 - 1.4e-15 (and so whether it is an interfacial cell).
"""

PI = 3.14159265358979
