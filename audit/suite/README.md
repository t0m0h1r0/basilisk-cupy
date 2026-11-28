# 監修の独立の検査一式（#185）
`PYTHONPATH=<対象のツリー> python audit/suite/suite.py --config base|r0|<JSON> --out result.json`（CPU、14 並列で数分）。
合否は判定せず値を記録する。A の `cases/` は使わない。参照解：`oblique_shear_reference.py`（監修 C-3）、毛管波は厳密な線形の値（t ≤ 2）。

## 未統合（打ち切りの条件により、一覧に残す）
- 壁と粘性のある総運動量の収支（`audit/20261128-c6b`）、壁と周期の境界の 2 次の次数（鏡映の一致だけを入れた）
- 毛管波の 10 周期、静水圧の完全なソルバー（t = 1・20）、上昇気泡
