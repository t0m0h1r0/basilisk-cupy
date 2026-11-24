"""Red/black Gauss--Seidel sweep of ``div(alpha grad a) + lambda a = res`` as one cupy kernel per colour.

Same arithmetic, in the same order, as ``HelmholtzProblem.relax`` (array path):

    num = -res + sum_ax (a_hi u[+1] + a_lo u[-1]) / h_ax^2
    den = -lam + sum_ax (a_hi + a_lo) / h_ax^2
    u   = num / den   (0 where den == 0)

Cells of the other colour are only read, so updating in place equals the
masked update of the array path. Kernels are compiled with ``--fmad=false`` so
that no multiply-add is contracted and the result is bitwise the array path's.

Index conventions shared by the kernels: ``t`` is the cell's flat index in C
order over the interior ``n``; ``p`` is its index in the padded array
(``ghosts`` layers per side); the face array normal to axis ``a`` has
``n_a + 1`` entries along ``a``. Indices are ``idx_t``: 32-bit when every array
is small enough, 64-bit otherwise (integer division is much cheaper in 32 bits).
A red/black sweep launches threads for the cells of one colour only: along the
last axis every other cell, offset by the parity of the other indices.
"""

from __future__ import annotations

from functools import lru_cache

THREADS = 256
_INT32_LIMIT = 2**31 - 2**20


def unflatten(ndim: int, flat: str = "t") -> list[str]:
    """C statements giving ``i0 .. i{ndim-1}`` from the flat interior index ``flat``."""
    out, rem = [], flat
    for a in range(ndim - 1, -1, -1):
        out.append(f"const long long i{a} = {rem} % n{a};")
        if a:
            out.append(f"const long long r{a} = {rem} / n{a};")
            rem = f"r{a}"
    return out


def unflatten_colour(ndim: int) -> list[str]:
    """``i0 .. i{ndim-1}`` and ``t`` for thread ``q`` of a colour-``colour`` sweep (returns if outside)."""
    last = ndim - 1
    out = [f"const long long m = (n{last} + 1) / 2;", "const long long k = q % m;"]
    rem = "(q / m)"
    for a in range(last - 1, -1, -1):
        out.append(f"const long long i{a} = {rem} % n{a};")
        if a:
            out.append(f"const long long r{a} = {rem} / n{a};")
            rem = f"r{a}"
    others = " + ".join([f"i{a}" for a in range(last)] + ["colour"])
    out.append(f"const long long i{last} = 2 * k + (({others}) & 1);")
    out.append(f"if (i{last} >= n{last}) return;")
    strides = [" * ".join(f"n{b}" for b in range(a + 1, ndim)) or "1" for a in range(ndim)]
    out.append("const long long t = " + " + ".join(f"i{a} * ({strides[a]})" for a in range(ndim)) + ";")
    return out


def padded_strides(ndim: int) -> list[str]:
    return [" * ".join(f"(n{b} + 2 * g)" for b in range(a + 1, ndim)) or "1" for a in range(ndim)]


def face_strides(ndim: int, axis: int) -> list[str]:
    return [" * ".join((f"(n{c} + 1)" if c == axis else f"n{c}") for c in range(b + 1, ndim)) or "1"
            for b in range(ndim)]


def wide_indices(*arrays) -> bool:
    """Whether 64-bit indices are needed for ``arrays``."""
    return any(a.size >= _INT32_LIMIT for a in arrays)


def compile_kernel(source: str, name: str, wide: bool):
    """Compile ``source`` with ``idx_t`` (written ``long long`` in the generators) 32- or 64-bit."""
    import cupy as cp  # noqa: PLC0415

    text = source.replace("unsigned long long", "UNSIGNED_LL").replace("long long", "idx_t")
    text = text.replace("UNSIGNED_LL", "unsigned long long")
    header = "typedef long long idx_t;\n" if wide else "typedef int idx_t;\n"
    return cp.RawKernel(header + text, name, options=("--fmad=false",))


def index(wide: bool):
    """Host scalar type matching ``idx_t``."""
    import cupy as cp  # noqa: PLC0415

    return cp.int64 if wide else cp.int32


def launch_colours(kernel, n, args, wide: bool) -> None:
    """Run ``kernel`` for colour 1 (odd index sum) then colour 0, as Basilisk's red/black
    relaxation (``(point.i + parity) % 2 != point.j % 2`` with parity 0 first); one thread per
    cell of the colour; ``args`` end before ``colour, count``."""
    import cupy as cp  # noqa: PLC0415

    count = (n[-1] + 1) // 2
    for v in n[:-1]:
        count *= v
    blocks = (count + THREADS - 1) // THREADS
    for colour in (1, 0):
        kernel((blocks,), (THREADS,), tuple(args) + (cp.int32(colour), index(wide)(count)))


def _source(ndim: int, has_lambda: bool) -> str:
    ps = padded_strides(ndim)
    pidx = " + ".join(f"(i{a} + g) * ({ps[a]})" for a in range(ndim))
    params = (["double* __restrict__ u", "const double* __restrict__ res"]
              + (["const double* __restrict__ lam"] if has_lambda else [])
              + [f"const double* __restrict__ al{a}" for a in range(ndim)]
              + [f"const long long n{a}" for a in range(ndim)]
              + [f"const double h2_{a}" for a in range(ndim)]
              + ["const long long g", "const int colour", "const long long count"])
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
    nl = "\n    "
    return f"""
extern "C" __global__ void relax({", ".join(params)}) {{
    const long long q = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (q >= count) return;
    {nl.join(unflatten_colour(ndim))}
    const long long p = {pidx};
    {nl.join(body)}
    u[p] = den != 0.0 ? num / den : 0.0;
}}
"""


@lru_cache(maxsize=None)
def _kernel(ndim: int, has_lambda: bool, wide: bool):
    return compile_kernel(_source(ndim, has_lambda), "relax", wide)


def relax_redblack(u, res, alphas, lam, n, h, ghosts: int) -> None:
    """Both colours (even, then odd) on the padded array ``u`` in place."""
    import cupy as cp  # noqa: PLC0415

    if u.dtype != cp.float64 or not u.flags.c_contiguous:
        raise TypeError("relax kernel needs a C-contiguous float64 array")
    args = [u, cp.ascontiguousarray(res, dtype=cp.float64)]
    if lam is not None:
        args.append(cp.ascontiguousarray(lam, dtype=cp.float64))
    alphas = [cp.ascontiguousarray(a, dtype=cp.float64) for a in alphas]
    wide = wide_indices(u, *alphas)
    ix = index(wide)
    args += alphas + [ix(v) for v in n] + [cp.float64(hv ** 2) for hv in h] + [ix(ghosts)]
    launch_colours(_kernel(len(n), lam is not None, wide), n, args, wide)
