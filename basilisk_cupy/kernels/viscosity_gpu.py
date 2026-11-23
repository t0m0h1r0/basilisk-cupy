"""Red/black relaxation of Basilisk's implicit viscosity as one cupy kernel per colour.

Same arithmetic, in the same order, as ``ViscosityProblem.relax`` (array path).
A cell of the current colour reads only cells of the other colour and the
frozen copy ``ua`` (corner terms), and all components of one cell are computed
before any is written, so the in-place update equals the masked update.
"""

from __future__ import annotations

from functools import lru_cache

from .relax_gpu import compile_kernel, face_strides, index, launch_colours, padded_strides, unflatten_colour, wide_indices


def _source(ndim: int, rho_array: bool) -> str:
    ps = padded_strides(ndim)
    pidx = " + ".join(f"(i{a} + g) * ({ps[a]})" for a in range(ndim))
    params = ([f"double* __restrict__ u{i}" for i in range(ndim)]
              + [f"const double* __restrict__ v{i}" for i in range(ndim)]
              + [f"const double* __restrict__ res{i}" for i in range(ndim)]
              + [f"const double* __restrict__ mu{j}" for j in range(ndim)]
              + (["const double* __restrict__ rho"] if rho_array else [])
              + ["const double* __restrict__ prm"]  # prm[0] = dt, prm[1] = constant rho
              + [f"const long long n{a}" for a in range(ndim)]
              + ["const double h2", "const long long g", "const int colour", "const long long count"])
    body = []
    for j in range(ndim):
        fs = face_strides(ndim, j)
        flo = " + ".join(f"i{b} * ({fs[b]})" for b in range(ndim))
        body.append(f"const double mlo{j} = mu{j}[{flo}], mhi{j} = mu{j}[{flo} + ({fs[j]})];")
    body.append(f"const double dtr = prm[0] / {'rho[t]' if rho_array else 'prm[1]'};")
    for i in range(ndim):
        si = f"({ps[i]})"
        body.append(f"double num{i} = 0.0, den{i} = 0.0;")
        for j in range(ndim):
            sj = f"({ps[j]})"
            if j == i:
                body.append(f"num{i} = num{i} + 2.0 * mhi{j} * u{i}[p + {si}] + 2.0 * mlo{j} * u{i}[p - {si}];")
                body.append(f"den{i} = den{i} + 2.0 * mhi{j} + 2.0 * mlo{j};")
            else:
                body.append(
                    f"num{i} = num{i} + mhi{j} * (u{i}[p + {sj}] + (u{j}[p + {si}] + v{j}[p + {si} + {sj}]) / 4.0"
                    f" - (u{j}[p - {si}] + v{j}[p - {si} + {sj}]) / 4.0);")
                body.append(
                    f"num{i} = num{i} - mlo{j} * (-u{i}[p - {sj}] + (v{j}[p + {si} - {sj}] + u{j}[p + {si}]) / 4.0"
                    f" - (v{j}[p - {si} - {sj}] + u{j}[p - {si}]) / 4.0);")
                body.append(f"den{i} = den{i} + mhi{j} + mlo{j};")
        body.append(f"const double new{i} = (res{i}[t] * h2 + dtr * num{i}) / (h2 + dtr * den{i});")
    body += [f"u{i}[p] = new{i};" for i in range(ndim)]
    nl = "\n    "
    return f"""
extern "C" __global__ void relax_viscosity({", ".join(params)}) {{
    const long long q = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (q >= count) return;
    {nl.join(unflatten_colour(ndim))}
    const long long p = {pidx};
    {nl.join(body)}
}}
"""


@lru_cache(maxsize=None)
def _kernel(ndim: int, rho_array: bool, wide: bool):
    return compile_kernel(_source(ndim, rho_array), "relax_viscosity", wide)


def relax_viscosity_redblack(us, res, mu, rho, n, h2: float, params, ghosts: int, frozen=None) -> None:
    """Both colours on the padded component arrays ``us`` in place.

    ``params``: device array ``[dt, rho]`` (``rho`` used when it is a constant),
    read by the kernel so that a recorded CUDA graph follows a changing dt.
    ``frozen``: preallocated arrays like ``us`` for Basilisk's frozen copy
    (allocated here when not given).
    """
    import cupy as cp  # noqa: PLC0415

    ndim = len(n)
    for u in us:
        if u.dtype != cp.float64 or not u.flags.c_contiguous:
            raise TypeError("viscosity kernel needs C-contiguous float64 arrays")
    if frozen is None:
        ua = [u.copy() for u in us]  # Basilisk's frozen copy for the corner terms
    else:
        ua = frozen
        for dst, u in zip(ua, us):
            cp.copyto(dst, u)
    rho_array = hasattr(rho, "shape")
    mu = [cp.ascontiguousarray(m, dtype=cp.float64) for m in mu]
    wide = wide_indices(*us, *mu)
    ix = index(wide)
    args = (list(us) + ua + [cp.ascontiguousarray(r, dtype=cp.float64) for r in res] + mu
            + ([cp.ascontiguousarray(rho, dtype=cp.float64)] if rho_array else []) + [params]
            + [ix(v) for v in n] + [cp.float64(h2), ix(ghosts)])
    launch_colours(_kernel(ndim, rho_array, wide), n, args, wide)
