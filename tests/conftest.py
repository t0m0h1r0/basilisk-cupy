import pytest

from basilisk_cupy.backend import cupy_available

BACKENDS = ["numpy"] + (["cupy"] if cupy_available() else [])


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def order(errors, ns):
    """Observed convergence orders between successive resolutions."""
    import math

    return [math.log(errors[i] / errors[i + 1]) / math.log(ns[i + 1] / ns[i]) for i in range(len(errors) - 1)]
