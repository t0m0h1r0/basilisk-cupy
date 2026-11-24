"""Column propagation of height functions (Basilisk ``column_propagation``) as one cupy kernel.

One thread per column; the column is updated in place, in order, as Basilisk's ``foreach()``
does (cells already visited are used with their new values), so the result is the array
path's to the bit.
"""

from __future__ import annotations

from functools import lru_cache

THREADS = 128

_SOURCE = r"""
__device__ __forceinline__ double hgt(double H) {
    return H > 10.0 ? H - 20.0 : (H < -10.0 ? H + 20.0 : H);
}
extern "C" __global__ void column_propagation(double* __restrict__ h, const long long n,
                                              const long long stride, const long long other,
                                              const long long ostride_hi, const long long ostride_lo,
                                              const long long nlo) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= other) return;
    const long long base = (t / nlo) * ostride_hi + (t % nlo) * ostride_lo;
    for (long long idx = 0; idx < n; idx++) {
        double cur = h[base + idx * stride];
        for (int i = -2; i <= 2; i++) {
            const long long k = idx + i;
            if (k < 0 || k >= n) continue;
            const double nb = h[base + k * stride];
            if (fabs(hgt(nb)) <= 3.5 && fabs(hgt(nb) + i) < fabs(hgt(cur)))
                cur = nb + i;
        }
        h[base + idx * stride] = cur;
    }
}
"""


@lru_cache(maxsize=None)
def _kernel():
    import cupy as cp  # noqa: PLC0415

    return cp.RawKernel(_SOURCE, "column_propagation", options=("--fmad=false",))


def column_propagation(h, axis: int) -> None:
    """In place, ``h`` a C-contiguous 2-D float64 cupy array; columns along ``axis``."""
    n0, n1 = h.shape
    item = h.itemsize
    s0, s1 = h.strides[0] // item, h.strides[1] // item
    if axis == 0:
        n, stride, other, hi, lo, nlo = n0, s0, n1, 0, s1, n1
    else:
        n, stride, other, hi, lo, nlo = n1, s1, n0, 0, s0, n0
    blocks = (other + THREADS - 1) // THREADS
    _kernel()((blocks,), (THREADS,), (h, n, stride, other, hi, lo, nlo))
