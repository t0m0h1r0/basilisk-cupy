"""Run independent experiments on several CPU cores (one single-threaded process per run).

    from cases.parallel import pmap
    results = pmap(run, [dict(n=32, dt=...), dict(n=64, dt=...)], jobs=8)

``run`` must be a module-level function of an importable module (the workers are spawned); each item is a dict of keyword arguments.
Results come back in the order of the items. ``jobs=1`` runs in this process (for debugging).
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _init() -> None:
    for v in _THREAD_VARS:
        os.environ[v] = "1"


def _call(fn, kw):
    return fn(**kw)


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def pmap(fn, items: list[dict], jobs: int | None = None) -> list:
    jobs = default_jobs() if jobs is None else int(jobs)
    if jobs <= 1 or len(items) <= 1:
        return [fn(**kw) for kw in items]
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    for v in _THREAD_VARS:  # inherited by the spawned workers before numpy loads its BLAS
        os.environ[v] = "1"
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(jobs, len(items)), mp_context=ctx, initializer=_init) as ex:
            futs = [ex.submit(_call, fn, kw) for kw in items]
            return [f.result() for f in futs]
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
