"""Red/black Gauss--Seidel sweep of ``div(alpha grad a) + lambda a = res`` as one cupy kernel per colour.

Same arithmetic, in the same order, as ``HelmholtzProblem.relax`` (array path):

    num = -res + sum_ax (a_hi u[+1] + a_lo u[-1]) / h_ax^2
    den = -lam + sum_ax (a_hi + a_lo) / h_ax^2
    u   = num / den   (0 where den == 0)

Cells of the other colour are only read, so updating in place equals the
masked update of the array path. Kernels are compiled with ``--fmad=false`` so
that no multiply-add is contracted and the result is bitwise the array path's.

Index conventions shared by the kernels: one thread per interior cell, flat
index ``t`` in C order over ``n``; ``p`` is the cell's index in the padded
array (``ghosts`` layers per side); the face array normal to axis ``a`` has
``n_a + 1`` entries along ``a``.
"""

from __future__ import annotations

from functools import lru_cache

THREADS = 256


def unflatten(ndim: int) -> list[str]:
    """C statements giving ``i0 .. i{ndim-1}`` from the flat index ``t``."""
    out, rem = [], "t"
    for a in range(ndim - 1, -1, -1):
        out.append(f"const long long i{a} = {rem} % n{a};")
        if a:
            out.append(f"const long long r{a} = {rem} / n{a};")
            rem = f"r{a}"
    return out


def padded_strides(ndim: int) -> list[str]:
    return [" * ".join(f"(n{b} + 2 * g)" for b in range(a + 1, ndim)) or "1" for a in range(ndim)]


def face_strides(ndim: int, axis: int) -> list[str]:
    return [" * ".join((f"(n{c} + 1)" if c == axis else f"n{c}") for c in range(b + 1, ndim)) or "1"
            for b in range(ndim)]


def launch_args(kernel, n, args) -> None:
    """Run ``kernel`` for colour 0 then colour 1; ``args`` end before ``colour, total``."""
    import cupy as cp  # noqa: PLC0415

    total = 1
    for v in n:
        total *= v
    blocks = (total + THREADS - 1) // THREADS
    for colour in (0, 1):
        kernel((blocks,), (THREADS,), tuple(args) + (cp.int32(colour), cp.int64(total)))


def _source(ndim: int, has_lambda: bool) -> str:
    ps = padded_strides(ndim)
    pidx = " + ".join(f"(i{a} + g) * ({ps[a]})" for a in range(ndim))
    params = (["double* __restrict__ u", "const double* __restrict__ res"]
              + (["const double* __restrict__ lam"] if has_lambda else [])
              + [f"const double* __restrict__ al{a}" for a in range(ndim)]
              + [f"const long long n{a}" for a in range(ndim)]
              + [f"const double h2_{a}" for a in range(ndim)]
              + ["const long long g", "const int colour", "const long long total"])
    body = ["double num = -res[t];", f"double den = {'-lam[t]' if has_lambda else '0.0'};"]
    for a in range(ndim):
        fs = face_strides(ndim, a)
        flo = " + ".join(f"i{b} * ({fs[b]})" for b in range(ndim))
        body += [
            "{",
            f"    const long long f = {flo};",
            f"    const double alo = al{a}[f], ahi = al{a}[f + ({fs[a]})];",
            f"    const long long s = {ps[a]};",
            f"    num = num + (ahi * u[p + s] + alo * u[p - s]) / h2_{a};",
            f"    den = den + (ahi + alo) / h2_{a};",
            "}",
        ]
    parity = " + ".join(f"i{a}" for a in range(ndim))
    nl = "\n    "
    return f"""
extern "C" __global__ void relax({", ".join(params)}) {{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= total) return;
    {nl.join(unflatten(ndim))}
    if (({parity}) % 2 != colour) return;
    const long long p = {pidx};
    {nl.join(body)}
    u[p] = den != 0.0 ? num / den : 0.0;
}}
"""


@lru_cache(maxsize=None)
def _kernel(ndim: int, has_lambda: bool):
    import cupy as cp  # noqa: PLC0415

    return cp.RawKernel(_source(ndim, has_lambda), "relax", options=("--fmad=false",))


def relax_redblack(u, res, alphas, lam, n, h, ghosts: int) -> None:
    """Both colours (even, then odd) on the padded array ``u`` in place."""
    import cupy as cp  # noqa: PLC0415

    if u.dtype != cp.float64 or not u.flags.c_contiguous:
        raise TypeError("relax kernel needs a C-contiguous float64 array")
    args = [u, cp.ascontiguousarray(res, dtype=cp.float64)]
    if lam is not None:
        args.append(cp.ascontiguousarray(lam, dtype=cp.float64))
    args += [cp.ascontiguousarray(a, dtype=cp.float64) for a in alphas]
    args += [cp.int64(v) for v in n] + [cp.float64(hv ** 2) for hv in h] + [cp.int64(ghosts)]
    launch_args(_kernel(len(n), lam is not None), n, args)
