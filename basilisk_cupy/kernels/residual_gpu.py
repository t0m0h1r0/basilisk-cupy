"""Multigrid residuals with their maximum norm as one cupy kernel each (fd2 operators).

Same arithmetic, in the same order, as the array paths
``HelmholtzProblem.residual`` with :class:`FD2FaceComplex` and
``ViscosityProblem.residual``. The kernel also reduces ``max |res|`` into a
device scalar: each warp takes the maximum of the bit patterns of ``|res|``
(order independent for non-negative doubles, NaN included, so the norm equals
``xp.max(xp.abs(res))`` exactly) and one lane per warp does an ``atomicMax``.
Nothing is allocated, so the residual can be recorded in a CUDA graph together
with the cycle; reading the norm back is then the only host synchronisation of
a cycle.
"""

from __future__ import annotations

from functools import lru_cache

from .relax_gpu import THREADS, compile_kernel, face_strides, index, padded_strides, unflatten, wide_indices

_ATOMIC_MAX = """
__device__ __forceinline__ unsigned long long abs_bits(double r) {
    return (unsigned long long)__double_as_longlong(fabs(r));
}
__device__ __forceinline__ void warp_max_to(unsigned long long* out, unsigned long long best) {
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long other = __shfl_down_sync(0xffffffffu, best, off);
        best = other > best ? other : best;
    }
    if ((threadIdx.x & 31) == 0 && best != 0ull) atomicMax(out, best);
}
"""


def _helmholtz_source(ndim: int, has_lambda: bool) -> str:
    ps = padded_strides(ndim)
    pidx = " + ".join(f"(i{a} + g) * ({ps[a]})" for a in range(ndim))
    params = (["const double* __restrict__ u", "const double* __restrict__ b", "double* __restrict__ res"]
              + (["const double* __restrict__ lam"] if has_lambda else [])
              + [f"const double* __restrict__ al{a}" for a in range(ndim)]
              + [f"const long long n{a}" for a in range(ndim)]
              + [f"const double h{a}" for a in range(ndim)]
              + ["const long long g", "unsigned long long* __restrict__ norm", "const long long total"])
    body = ["double div;"]
    for a in range(ndim):
        fs = face_strides(ndim, a)
        flo = " + ".join(f"i{b} * ({fs[b]})" for b in range(ndim))
        body += [
            "{",
            f"    const long long f = {flo};",
            f"    const long long s = {ps[a]};",
            f"    const double flo = al{a}[f] * ((u[p] - u[p - s]) / h{a});",
            f"    const double fhi = al{a}[f + ({fs[a]})] * ((u[p + s] - u[p]) / h{a});",
            f"    const double term = (fhi - flo) / h{a};",
            f"    div = {'term' if a == 0 else 'div + term'};",
            "}",
        ]
    body.append("double r = b[t] - div;")
    if has_lambda:
        body.append("r = r - lam[t] * u[p];")
    nl = "\n    "
    return _ATOMIC_MAX + f"""
extern "C" __global__ void helmholtz_residual({", ".join(params)}) {{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long best = 0ull;
    if (t < total) {{
    {nl.join(unflatten(ndim))}
    const long long p = {pidx};
    {nl.join(body)}
    res[t] = r;
    best = abs_bits(r);
    }}
    warp_max_to(norm, best);
}}
"""


def _viscosity_source(ndim: int, rho_array: bool) -> str:
    ps = padded_strides(ndim)
    pidx = " + ".join(f"(i{a} + g) * ({ps[a]})" for a in range(ndim))
    params = ([f"const double* __restrict__ u{i}" for i in range(ndim)]
              + [f"const double* __restrict__ b{i}" for i in range(ndim)]
              + [f"double* __restrict__ res{i}" for i in range(ndim)]
              + [f"const double* __restrict__ mu{j}" for j in range(ndim)]
              + (["const double* __restrict__ rho"] if rho_array else [])
              + ["const double* __restrict__ prm"]  # prm[0] = dt, prm[1] = constant rho
              + [f"const long long n{a}" for a in range(ndim)]
              + ["const double h2", "const long long g", "unsigned long long* __restrict__ norm",
                 "const long long total"])
    body = []
    for j in range(ndim):
        fs = face_strides(ndim, j)
        flo = " + ".join(f"i{b} * ({fs[b]})" for b in range(ndim))
        body.append(f"const double mlo{j} = mu{j}[{flo}], mhi{j} = mu{j}[{flo} + ({fs[j]})];")
    body.append(f"const double dtr = prm[0] / {'rho[t]' if rho_array else 'prm[1]'};")
    for i in range(ndim):
        si = f"({ps[i]})"
        body.append(f"{{ const double uc = u{i}[p]; double acc = 0.0;")
        for j in range(ndim):
            sj = f"({ps[j]})"
            if j == i:
                body.append(f"acc = acc + 2.0 * mhi{j} * (u{i}[p + {si}] - uc) - 2.0 * mlo{j} * (uc - u{i}[p - {si}]);")
            else:
                body.append(
                    f"acc = acc + mhi{j} * (u{i}[p + {sj}] - uc + (u{j}[p + {si}] + u{j}[p + {si} + {sj}]) / 4.0"
                    f" - (u{j}[p - {si}] + u{j}[p - {si} + {sj}]) / 4.0);")
                body.append(
                    f"acc = acc - mlo{j} * (uc - u{i}[p - {sj}] + (u{j}[p + {si} - {sj}] + u{j}[p + {si}]) / 4.0"
                    f" - (u{j}[p - {si} - {sj}] + u{j}[p - {si}]) / 4.0);")
        body.append(f"const double r = b{i}[t] - uc + dtr * acc / h2; res{i}[t] = r;")
        body.append("const unsigned long long bits = abs_bits(r); best = bits > best ? bits : best; }")
    nl = "\n    "
    return _ATOMIC_MAX + f"""
extern "C" __global__ void viscosity_residual({", ".join(params)}) {{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long best = 0ull;
    if (t < total) {{
    {nl.join(unflatten(ndim))}
    const long long p = {pidx};
    {nl.join(body)}
    }}
    warp_max_to(norm, best);
}}
"""


@lru_cache(maxsize=None)
def _helmholtz_kernel(ndim: int, has_lambda: bool, wide: bool):
    return compile_kernel(_helmholtz_source(ndim, has_lambda), "helmholtz_residual", wide)


@lru_cache(maxsize=None)
def _viscosity_kernel(ndim: int, rho_array: bool, wide: bool):
    return compile_kernel(_viscosity_source(ndim, rho_array), "viscosity_residual", wide)


def _launch(kernel, n, args, wide: bool):
    total = 1
    for v in n:
        total *= v
    kernel(((total + THREADS - 1) // THREADS,), (THREADS,), tuple(args) + (index(wide)(total),))


def _check(*arrays):
    import cupy as cp  # noqa: PLC0415

    for arr in arrays:
        if arr.dtype != cp.float64 or not arr.flags.c_contiguous:
            raise TypeError("residual kernels need C-contiguous float64 arrays")


def helmholtz_residual(u, b, res, alphas, lam, n, h, ghosts: int, norm) -> None:
    """``res = b - div(alpha grad u) - lam u`` (fd2) and ``norm = max(norm, max|res|)``.

    ``norm``: device ``uint64`` scalar holding the bits of a non-negative double
    (zero it first).
    """
    import cupy as cp  # noqa: PLC0415

    _check(u, b, res, *alphas, *([lam] if lam is not None else []))
    wide = wide_indices(u, *alphas)
    ix = index(wide)
    args = [u, b, res] + ([lam] if lam is not None else []) + list(alphas)
    args += [ix(v) for v in n] + [cp.float64(v) for v in h] + [ix(ghosts), norm]
    _launch(_helmholtz_kernel(len(n), lam is not None, wide), n, args, wide)


def viscosity_residual(us, bs, res, mu, rho, params, n, h2: float, ghosts: int, norm) -> None:
    """Basilisk's viscous residual for every component and ``norm = max(norm, max|res|)``."""
    import cupy as cp  # noqa: PLC0415

    rho_array = hasattr(rho, "shape")
    _check(*us, *bs, *res, *mu, *([rho] if rho_array else []))
    wide = wide_indices(*us, *mu)
    ix = index(wide)
    args = list(us) + list(bs) + list(res) + list(mu) + ([rho] if rho_array else []) + [params]
    args += [ix(v) for v in n] + [cp.float64(h2), ix(ghosts), norm]
    _launch(_viscosity_kernel(len(n), rho_array, wide), n, args, wide)
