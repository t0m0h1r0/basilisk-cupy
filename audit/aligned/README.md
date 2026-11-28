# 監修（C）：格子の面に揃った界面を一様流で運ぶと、源の項なしの運動量保存形でも誤差が指数的に育つ
- `aligned_base.py`・`aligned_debug.py`：`PYTHONPATH=. python audit/aligned/<file>`（main、または PR #160 の版）
- `col.c`：C 版の Basilisk（`qcc -O2 -disable-dimensions col.c -o col -lm; ./col <rho2> <tolerance> <CFL>`）。
- `../pr161/aligned.py`：PR #161（案 4）のブランチで実行する。`../pr158/check158.py`：PR #158 のブランチで実行する。
