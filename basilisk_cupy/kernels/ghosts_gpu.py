"""Ghost-layer filling as one cupy kernel per axis (both sides, all layers).

Same arithmetic as the ``fill`` methods of :mod:`basilisk_cupy.boundary`
(array path), for conditions whose value is a constant:

* Dirichlet ``v``:  ``ghost_k = 2 v - interior_k``
* Neumann ``q``:    ``ghost_k = interior_k + ((2k - 1) h) q``
* Periodic:         copy from the opposite side

Axes are filled one after another over the full padded extent of the other
axes, so corner ghosts come out exactly as in the array path.
"""

from __future__ import annotations

from functools import lru_cache

from .relax_gpu import THREADS

DIRICHLET, NEUMANN, PERIODIC = 0, 1, 2

_SOURCE = r"""
extern "C" __global__ void fill_axis(double* __restrict__ d, const long long outer, const long long len,
                                     const long long inner, const long long n, const long long g,
                                     const int klo, const double vlo, const int khi, const double vhi,
                                     const double h) {
    // one thread per (outer, inner) position of the boundary plane; loops over sides and layers
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= outer * inner) return;
    const long long o = t / inner, i = t % inner;
    const long long base = o * len * inner + i;
    for (int side = 0; side < 2; ++side) {
        const int kind = side == 0 ? klo : khi;
        const double v = side == 0 ? vlo : vhi;
        for (long long k = 1; k <= g; ++k) {
            long long ghost, src;
            if (side == 0) {
                ghost = g - k;
                src = kind == 2 ? g + n - k : g + k - 1;
            } else {
                ghost = g + n + k - 1;
                src = kind == 2 ? g + k - 1 : g + n - k;
            }
            const double a = d[base + src * inner];
            double out;
            if (kind == 0) out = 2.0 * v - a;
            else if (kind == 1) out = a + ((double)(2 * k - 1) * h) * v;
            else out = a;
            d[base + ghost * inner] = out;
        }
    }
}
"""


@lru_cache(maxsize=None)
def _kernel():
    import cupy as cp  # noqa: PLC0415

    return cp.RawKernel(_SOURCE, "fill_axis", options=("--fmad=false",))


def fill_ghosts(data, n, h, ghosts: int, sides) -> None:
    """``sides[axis] = ((kind_lo, value_lo), (kind_hi, value_hi))``; fills ``data`` in place.

    Periodic sides read the opposite side before writing their own layers, as
    the array path does (lower side first, then upper side).
    """
    import cupy as cp  # noqa: PLC0415

    if data.dtype != cp.float64 or not data.flags.c_contiguous:
        raise TypeError("ghost kernel needs a C-contiguous float64 array")
    shape = data.shape
    kern = _kernel()
    for axis, ((klo, vlo), (khi, vhi)) in enumerate(sides):
        outer = 1
        for v in shape[:axis]:
            outer *= v
        inner = 1
        for v in shape[axis + 1:]:
            inner *= v
        total = outer * inner
        kern(((total + THREADS - 1) // THREADS,), (THREADS,),
             (data, cp.int64(outer), cp.int64(shape[axis]), cp.int64(inner), cp.int64(n[axis]),
              cp.int64(ghosts), cp.int32(klo), cp.float64(vlo), cp.int32(khi), cp.float64(vhi),
              cp.float64(h[axis])))
