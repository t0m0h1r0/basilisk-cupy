# 12 二相流（M4）

## 計画（引き継ぎ文書の順）
1. **Basilisk 互換のベースライン**：VOF（vof.h）＋二相の物性（two-phase.h）＋表面張力 CSF（tension.h、高さ関数の曲率）。参照は Basilisk のテスト（reversed、curvature、capwave、oscillation、rising など）。
2. TwoPhaseFlow の手法へ順に差し替え：CLS（ψ）→ Ridge–Eikonal → ψ 曲率 → 分相 PPE＋HFE/GFM。
3. 変密度の圧力境界条件（壁で a ≠ 0）もここで扱う。

ベースラインは次の段に分ける。
- M4a：VOF 移流（PLIC、方向分割）← 済
- M4b：二相の物性と変密度の射影（two-phase.h、centered.h の α = 1/ρ）
- M4c：表面張力（tension.h＋curvature.h の高さ関数、balanced force）
- M4d：検証（静止液滴の寄生流れ、capwave、rising bubble）

## M4a：VOF 移流（PR #23）
- 移植：geometry.h（line_alpha、line_area、rectangle_fraction）、myc2d.h（MYC 法線）、fractions.h（頂点レベルセット → 体積率、再構成）、vof.h（方向分割、Weymouth & Yue の cc 項、反復ごとの方向交代、VOF 濃度）。2次元・一様格子。
- C と同じ演算順で書いた結果、reversed.c（一様格子版 reversed.cref）と表示桁まで一致：
  - 初期の総和 N = 32 / 64 / 128：0.124638306824 / 0.125405298502 / 0.125600659213（12桁）
  - T 後の誤差 avg / rms / max：N = 32 で 0.0605584 / 0.21841 / 1、N = 64 で 0.00992636 / 0.0733979 / 1、N = 128 で 0.00143235 / 0.017043 / 0.474599
  - 総和は保存、f は [−1.4e-16, 1 + 4e-16]、VOF 濃度 cf/f は 1 から 8e-13 以内
- イベント順：時刻 t の速度 → dt（timestep、最初の1回だけ CFL 0.8、以後 vof.h の上限 0.5）→ VOF。
- cupy で N = 128（3723 ステップ）28 秒。
- MYC 法線は直線を常に厳密には再現しない（30° の直線で最大 2.3e-3。中心差分の型が使えないセルでは Youngs 法になる）。

## FCCD と組み合わせるときの注意
- VOF は面速度の D_f が厳密に 0 であることを前提にしている。FCCD の射影後の u_f はそうならない（03）。`ops.wall_free_flux(uf, periodic)` を VOF に渡す。
