# 検討メモ

設計判断の理由、採らなかった案、根拠となる実測値をトピックごとに記録します。数値は実測値で、出典の PR を併記しています。

| ファイル | 内容 |
| --- | --- |
| [01_compact_principle.md](01_compact_principle.md) | 原則：ステンシルを広げず高階微分で精度を上げる／壁クロージャ（#16） |
| [02_hyperviscosity_spectrum.md](02_hyperviscosity_spectrum.md) | UCCD6 ハイパー粘性の負の固有値（判断：このまま） |
| [03_face_divergence.md](03_face_divergence.md) | 面の発散 D_f の隠れた2次誤差と4次の保存形（#17）／壁で 0 になる VOF 用の流束（#22） |
| [04_sixth_order_face_ops.md](04_sixth_order_face_ops.md) | 6次の面演算子：D2 を2回かけず、セルのジェットのエルミート補間で作る |
| [05_time_stepping.md](05_time_stepping.md) | 時間刻み：EXT2 の限界、IMEX-RK、面状態、陰的ハイパー粘性 |
| [06_viscous_defect_correction.md](06_viscous_defect_correction.md) | 粘性の欠陥補正：layered 形が停滞し split 形にした理由 |
| [07_mapped_grids.md](07_mapped_grids.md) | 写像格子：計量の作り方、Neumann データ、線緩和 |
| [08_reconstruction.md](08_reconstruction.md) | R_h（面 → セル）のコンパクト版 |
| [09_gpu.md](09_gpu.md) | GPU 高速化：融合カーネル、CUDA グラフ、問題の再利用 |
| [10_paper_errata.md](10_paper_errata.md) | TwoPhaseFlow 論文・実装で見つかった誤りと抜け |
| [11_verification_practice.md](11_verification_practice.md) | 検証のときの注意 |
| [12_two_phase.md](12_two_phase.md) | 二相流（M4）：計画と VOF 移流（#23） |
| [13_cls.md](13_cls.md) | CLS（保存型レベルセット）への差し替えと VOF との比較（#27）、Ridge–Eikonal 再初期化（#28） |
| [14_psi_curvature.md](14_psi_curvature.md) | ψ からの曲率：正確なプロファイルで O(h⁻¹)、輸送後の収束は再初期化しだい（#29） |
