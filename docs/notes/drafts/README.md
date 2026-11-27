# 保留中の試作（ユーザーの方針転換、2026-11-27）

ユーザーの判断で、研究の中心を CCD（原始関数型 CCD、PCCD）に戻した。最小二乗 Galerkin の拡張は保留する。ここには、その時点の下書きと測定を残す（PR にはしない）。

**注意**：ここの測定は、すべて誤った diff の形（制約付き最小二乗、ノート90 で訂正）で計算した。

- `90_shear_deformation.md`：P1-3、非定常せん断（厳密な写像）。`cases/galerkin2d_shear.py`。構造の確認まで済み、本計算は途中で停止した。
- `91_polynomial_galerkin.md`：多項式近似 Galerkin。残差の多項式を q² にすればエネルギーは増えない。q(1) = 0 なら自由流と総運動量は厳密。G₁ ⪰ M なので spec ⊂ [1, b]。`cases/galerkin2d_circle.py` の `poly_transport`。
  - PCCD–Galerkin（λ ∈ [1, 1.53]）でも、そのまま使える。
- `92_thin_band.md`：近い 2 つの界面。両側の適合の混合（`pair`）と、両方の折れの適合（`two`）。連続性の測定は途中で停止した。
- 高さ関数のヒンジ（`hinge="height"`、円の並進、`docs/figures/galerkin2d_circle_height.log`）：t = 0.1 で N = 24 → 64 の誤差が 1.05e-2、3.57e-3、1.60e-3、5.40e-4（約 3 次）。接線（2.9 次）と厳密な距離（4.9 次）の間。
