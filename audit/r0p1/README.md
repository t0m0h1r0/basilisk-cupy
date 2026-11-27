# 監修（C）：PR #143（ノート112、R0-P1）の独立の検証

main には入れない。実行は、リポジトリの直下で `PYTHONPATH=. python audit/r0p1/<file>`。

| ファイル | 内容 |
| --- | --- |
| `derive_local_error.py` | 1 ステップの局所誤差の主要項（奇成分：ユーザーの式、偶成分：源の項）の記号計算による導出。A のコードを使わない |
| `independent_transport.py` | 輸送の独立の再実装（numpy、周期は `np.roll`）。ノート112 の 2・3 節の数値との照合と、非対称な流れ（kx ≠ ky） |
| `run_coupled.py` | ノート112 の 4 節の主要な行の再現（A のコード） |
| `ghost_scope.py` | 連成のソルバーで、運動量のゴーストセルを q 自身の境界条件にした場合（周期。壁は圧力の求解が遅く未完了） |
| `ghost_scope_transport.py` | 輸送だけ（厳密な面の速度と g）で、周期と壁（対称の境界）のゴーストセルの影響 |
| `c/run.sh` | C 版の Basilisk（quadtree・multigrid）で、`conserving.h` の q1・q2 の境界条件の上書きの有無を比べる |
