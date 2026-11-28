# #195：C 版の conserving.h とライブラリの既定の食い違い（再現の手順）
```
B=<basilisk の src>; export PATH=$PATH:$B BASILISK=$B
cd <作業ディレクトリ> && cp <repo>/cases/c195/a.c . && qcc -O2 -disable-dimensions a.c -o a -lm && ./a   # vof_*.txt, end_*.txt（3 ステップ）
PYTHONPATH=<repo> python <repo>/cases/c195/cmp.py    # 段ごとの |lib − C|（lib は conserving_stale_ghosts=True・conserving_vof_half=True）
PYTHONPATH=<repo> python <repo>/cases/c195/cmp2.py   # ライブラリの既定と C、x 方向の一様性
PYTHONPATH=<repo> python <repo>/cases/c195/cmp3.py   # 仮説 H1（掃過の間でゴーストセルを更新しない）・H1p（隣の内部のセル）
```
作業ディレクトリで実行する（cmp*.py は C の出力を相対パスで読む）。監修の `audit/20261128-pr192/audit/pr192/c/one.c` と同じ配置で、conserving.h を改造せずに使う。
