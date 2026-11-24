import inspect
import os

# Parallel runs (pytest -n auto --dist loadgroup): one BLAS/OpenMP thread per worker, set before numpy loads.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

import pytest  # noqa: E402

from basilisk_cupy.backend import cupy_available  # noqa: E402

BACKENDS = ["numpy"] + (["cupy"] if cupy_available() else [])


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: uses the GPU without a cupy backend parameter (kept on one worker)")


def _uses_gpu(item) -> bool:
    if "cupy" in item.nodeid or item.get_closest_marker("gpu"):
        return True
    try:
        return "cupy" in inspect.getsource(item.function)
    except (OSError, TypeError):
        return False


def pytest_collection_modifyitems(config, items):
    """With pytest-xdist, the GPU tests share one worker (``--dist loadgroup``); numpy tests spread."""
    for item in items:
        if _uses_gpu(item):
            item.add_marker(pytest.mark.xdist_group("gpu"))


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def order(errors, ns):
    """Observed convergence orders between successive resolutions."""
    import math

    return [math.log(errors[i] / errors[i + 1]) / math.log(ns[i + 1] / ns[i]) for i in range(len(errors) - 1)]
