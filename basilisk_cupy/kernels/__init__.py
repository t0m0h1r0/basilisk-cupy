"""Fused compute kernels for hot loops (GPU: cupy RawKernel).

The reference implementation of every kernel is the array-expression code in
the module that calls it (``linear/poisson.py`` etc.); a kernel here must give
the same result to rounding and is used only when the arrays live on the GPU.
Fusing matters because each array expression is one kernel launch (~10 us),
which dominates the small multigrid levels.
"""
