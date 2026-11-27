# 監修（C）：PR #148（#126、固有値の走査の個数の検算）の独立の検証
PR のブランチ（`audit126-root-count`）の上で `PYTHONPATH=. python -m audit.pr148.check148`、`... break148`（scipy は不要、numpy のみ）。
`_stokes_reference.py` は、`audit/20261127-initial` ブランチの層ごとの Chebyshev 選点法（A のコードと独立）の写し。
