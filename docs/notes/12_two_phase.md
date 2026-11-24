# 12 二相流（M4）

## 計画（引き継ぎ文書の順）
1. **Basilisk 互換のベースライン**：VOF（vof.h）＋二相の物性（two-phase.h）＋表面張力 CSF（tension.h、高さ関数の曲率）。参照は Basilisk のテスト（reversed、curvature、capwave、oscillation、rising など）。
2. TwoPhaseFlow の手法へ順に差し替え：CLS（ψ）→ Ridge–Eikonal → ψ 曲率 → 分相 PPE＋HFE/GFM。
3. 変密度の圧力境界条件（壁で a ≠ 0）もここで扱う。

ベースラインは次の段に分ける。
- M4a：VOF 移流（PLIC、方向分割）← 済
- M4b：二相の物性、高さ関数と界面位置、界面力、換算重力 ← 済
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

## M4b：二相の物性・高さ関数・換算重力（PR #24）
- 移植：two-phase.h / two-phase-generic.h（α = 1/ρ(f_face)、ρ(f)、μ(f_face)、算術平均、FILTERED は任意）、heights.h（マルチグリッド版の高さ関数）、curvature.h の `position()`（高さ関数 → だめなら PLIC 線分の中点）、iforce.h（α φ ∇f の面加速度）、reduced.h（φ = [ρ] G·(x − Z)）。
- CenteredNS のフックに付ける。フックは後から登録したものが先に動く（Basilisk の同名イベントの順）。`on(..., after=True)` で後ろにも付けられる。
- 高さ関数の列の伝播（column_propagation）は逐次更新なので、GPU では1列1スレッドの RawKernel にした（numpy 版とビット一致）。
- 検証：Basilisk の gravity.c（重力波、Prosperetti の解との相対 RMS 誤差、gravity.ref）

| 1波長あたりの点数 | 本実装 | gravity.ref | 差 |
| --- | --- | --- | --- |
| 8 | 0.126721 | 0.126713 | +0.006% |
| 16 | 0.0317969 | 0.031758 | +0.12% |
| 32 | 0.00809545 | 0.00807759 | +0.22% |
| 64 | 0.00455989 | 0.00455815 | +0.04% |

- 738 ステップ。cupy で N = 128 が 37 秒。
- 平らな界面・換算重力のもとで、20 ステップ後の最大速度 1.4e-17（力と圧力ジャンプが厳密に釣り合う）。
- 未解決：参照との 0.04〜0.22% の差。N = 16 の差（8e-6）は Poisson の許容値による揺れ（1e-6 → 1e-8 → 1e-10 で 0.126721 / 0.1267168 / 0.1267182）と同程度だが、N = 32 の差（3.9e-5）は許容値の揺れ（2e-6）より大きく、系統的な違いがある。参照ファイルが現行ソースより古い可能性もある（ミラーの履歴では確認できない）。
- 確認済みの解釈：`event vof (i++, first)` は VOF 本体をステップの先頭へ移し、前のステップの dt で動かす（通常の位置に置くと 0.130 / 0.0379 と大きくずれる）。振幅は VOF の直後に測る（VOF の前に測ると 0.130 / 0.0379）。
