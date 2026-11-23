"""Multigrid prolongation and restriction as one cupy kernel each.

Same arithmetic, in the same order, as ``multigrid.prolong_bilinear`` (array
path), which applies ``fine = 0.75 c[0] + 0.25 c[+-1]`` axis by axis (axis 0
first) over the padded extent of the axes not yet refined. Unrolling that
recursion gives 2^d coarse reads per fine cell, nested exactly as the array
path evaluates them. Restriction sums the 2^d children in C order and divides
by 2^d, as ``multigrid.restrict_cell``.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

from .relax_gpu import THREADS, padded_strides


def _source(ndim: int) -> str:
    ps = padded_strides(ndim)

    def value(k: int, pos: list[str]) -> str:
        if k < 0:
            return "c[" + " + ".join(f"({pos[a]}) * ({ps[a]})" for a in range(ndim)) + "]"
        centre = pos.copy()
        centre[k] = f"c{k} + g"
        side = pos.copy()
        side[k] = f"c{k} + g + s{k}"
        return f"(0.75 * {value(k - 1, centre)} + 0.25 * {value(k - 1, side)})"

    # fine index I_a over 2 n_a, coarse cell c_a = I_a / 2, neighbour s_a = -1 (even child) or +1 (odd)
    unflat, rem = [], "t"
    for a in range(ndim - 1, -1, -1):
        unflat.append(f"const long long I{a} = {rem} % (2 * n{a});")
        if a:
            unflat.append(f"const long long r{a} = {rem} / (2 * n{a});")
            rem = f"r{a}"
    unflat += [f"const long long c{a} = I{a} / 2, s{a} = (I{a} % 2) ? 1 : -1;" for a in range(ndim)]
    params = ", ".join(["double* __restrict__ out", "const double* __restrict__ c"]
                       + [f"const long long n{a}" for a in range(ndim)]
                       + ["const long long g", "const long long og", "const long long total"])
    # output: fine array with ``og`` ghost layers (0: bare interior array)
    ostride = [" * ".join(f"(2 * n{b} + 2 * og)" for b in range(a + 1, ndim)) or "1" for a in range(ndim)]
    oidx = " + ".join(f"(I{a} + og) * ({ostride[a]})" for a in range(ndim))
    nl = "\n    "
    return f"""
extern "C" __global__ void prolong({params}) {{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= total) return;
    {nl.join(unflat)}
    out[{oidx}] = {value(ndim - 1, [""] * ndim)};
}}
"""


def _restrict_source(ndim: int) -> str:
    unflat, rem = [], "t"
    for a in range(ndim - 1, -1, -1):
        unflat.append(f"const long long i{a} = {rem} % n{a};")
        if a:
            unflat.append(f"const long long r{a} = {rem} / n{a};")
            rem = f"r{a}"
    fstride = [" * ".join(f"(2 * n{b})" for b in range(a + 1, ndim)) or "1" for a in range(ndim)]
    terms = []
    for child in itertools.product((0, 1), repeat=ndim):
        terms.append("f[" + " + ".join(f"(2 * i{a} + {child[a]}) * ({fstride[a]})" for a in range(ndim)) + "]")
    acc = [f"double s = {terms[0]};"] + [f"s = s + {t};" for t in terms[1:]]
    params = ", ".join(["double* __restrict__ out", "const double* __restrict__ f"]
                       + [f"const long long n{a}" for a in range(ndim)] + ["const long long total"])
    nl = "\n    "
    return f"""
extern "C" __global__ void restrict_cell({params}) {{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= total) return;
    {nl.join(unflat)}
    {nl.join(acc)}
    out[t] = s / {float(2 ** ndim)!r};
}}
"""


@lru_cache(maxsize=None)
def _kernel(ndim: int):
    import cupy as cp  # noqa: PLC0415

    return cp.RawKernel(_source(ndim), "prolong", options=("--fmad=false",))


@lru_cache(maxsize=None)
def _restrict_kernel(ndim: int):
    import cupy as cp  # noqa: PLC0415

    return cp.RawKernel(_restrict_source(ndim), "restrict_cell", options=("--fmad=false",))


def prolong(coarse_data, n, ghosts: int, out=None, out_ghosts: int = 0):
    """Fine values from the padded coarse array.

    Without ``out``: a new interior array of shape ``2 n``. With ``out``: written
    into the interior of ``out``, a fine array padded with ``out_ghosts`` layers
    (no allocation, so it can be captured in a CUDA graph).
    """
    import cupy as cp  # noqa: PLC0415

    if coarse_data.dtype != cp.float64 or not coarse_data.flags.c_contiguous:
        raise TypeError("prolongation kernel needs a C-contiguous float64 array")
    fine = tuple(2 * v for v in n)
    if out is None:
        out, out_ghosts = cp.empty(fine, dtype=cp.float64), 0
    elif tuple(out.shape) != tuple(v + 2 * out_ghosts for v in fine) or not out.flags.c_contiguous:
        raise ValueError("prolongation output has the wrong shape or layout")
    total = 1
    for v in fine:
        total *= v
    _kernel(len(n))(((total + THREADS - 1) // THREADS,), (THREADS,),
                    (out, coarse_data, *[cp.int64(v) for v in n], cp.int64(ghosts), cp.int64(out_ghosts),
                     cp.int64(total)))
    return out


def restrict(fine, out=None):
    """Average of the 2^d children of a fine interior array (each axis even)."""
    import cupy as cp  # noqa: PLC0415

    fine = cp.ascontiguousarray(fine)
    n = tuple(v // 2 for v in fine.shape)
    if out is None:
        out = cp.empty(n, dtype=cp.float64)
    elif tuple(out.shape) != n or not out.flags.c_contiguous or out.dtype != cp.float64:
        raise ValueError("restriction output has the wrong shape or layout")
    total = out.size
    _restrict_kernel(fine.ndim)(((total + THREADS - 1) // THREADS,), (THREADS,),
                                (out, fine, *[cp.int64(v) for v in n], cp.int64(total)))
    return out
