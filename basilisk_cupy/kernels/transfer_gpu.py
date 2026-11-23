"""Multigrid prolongation as one cupy kernel.

Same arithmetic, in the same order, as ``multigrid.prolong_bilinear`` (array
path), which applies ``fine = 0.75 c[0] + 0.25 c[+-1]`` axis by axis (axis 0
first) over the padded extent of the axes not yet refined. Unrolling that
recursion gives 2^d coarse reads per fine cell, nested exactly as the array
path evaluates them.
"""

from __future__ import annotations

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
                       + ["const long long g", "const long long total"])
    nl = "\n    "
    return f"""
extern "C" __global__ void prolong({params}) {{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= total) return;
    {nl.join(unflat)}
    out[t] = {value(ndim - 1, [""] * ndim)};
}}
"""


@lru_cache(maxsize=None)
def _kernel(ndim: int):
    import cupy as cp  # noqa: PLC0415

    return cp.RawKernel(_source(ndim), "prolong", options=("--fmad=false",))


def prolong(coarse_data, n, ghosts: int):
    """Fine interior array (shape ``2 n``) from the padded coarse array."""
    import cupy as cp  # noqa: PLC0415

    if coarse_data.dtype != cp.float64 or not coarse_data.flags.c_contiguous:
        raise TypeError("prolongation kernel needs a C-contiguous float64 array")
    out = cp.empty(tuple(2 * v for v in n), dtype=cp.float64)
    total = out.size
    _kernel(len(n))(((total + THREADS - 1) // THREADS,), (THREADS,),
                    (out, coarse_data, *[cp.int64(v) for v in n], cp.int64(ghosts), cp.int64(total)))
    return out
