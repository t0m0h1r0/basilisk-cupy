#!/bin/bash
# 監修の計算の実行：BLAS のスレッドを 1 にする（プロセス並列と重ねると過負荷になる）
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
exec /root/work/basilisk-cupy/.venv/bin/python -u -m "$@"
