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
| [12_two_phase.md](12_two_phase.md) | 二相流（M4）：計画、VOF 移流（#23）、物性・高さ関数・換算重力（#24）、表面張力（#26） |
| [13_cls.md](13_cls.md) | CLS（保存型レベルセット）への差し替えと VOF との比較（#27）、Ridge–Eikonal 再初期化（#28） |
| [14_psi_curvature.md](14_psi_curvature.md) | ψ からの曲率：正確なプロファイルで O(h⁻¹)、輸送後の収束は再初期化しだい（#29） |
| [15_cls_ns.md](15_cls_ns.md) | CLS の二相 NS への組み込み：静止液滴、毛管波（CSF の周波数ずれ）（#30） |
| [16_split_ppe.md](16_split_ppe.md) | 分相 PPE と GFM の圧力ジャンプ：静止液滴、毛管波、高密度比の時間刻み（#31） |
| [17_curvature_transport.md](17_curvature_transport.md) | 移流後の ψ 曲率の誤差は移流した界面から：毎ステップの圧縮・拡散で収束、毛管波は曲率で決まらない（#32） |
| [18_cls_transport.md](18_cls_transport.md) | ψ 移流の面差分は2次で ψ = ½ が 0.93% 遅れる：補正した発散と6次の面値で毛管波が VOF と同等に（#33） |
| [19_capillary_dt.md](19_capillary_dt.md) | 高密度比の発散は CCD 曲率の硬さ：表面張力の時間刻みに √(5/12)（#34） |
| [20_capillary_dt_check.md](20_capillary_dt_check.md) | √(5/12) の検証：曲率の硬さの比は測定と合うが、CLS では境目が N・粘性・密度比で下がる（#35） |
| [22_static_drop_figures.md](22_static_drop_figures.md) | 静止液滴の可視化：Basilisk（C）・VOF・CLS＋CSF・CLS＋GFM の密度・速度・圧力（#37） |
