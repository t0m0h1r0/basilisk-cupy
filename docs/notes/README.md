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
| [21_capillary_mode.md](21_capillary_mode.md) | 境目が N とともに下がる原因：格子規模の毛管波に対して CLS の移流が散逸を持たない。毎ステップの圧縮・拡散で解消（#36） |
| [22_static_drop_figures.md](22_static_drop_figures.md) | 静止液滴の可視化：Basilisk（C）・VOF・CLS＋CSF・CLS＋GFM の密度・速度・圧力（#37） |
| [24_acls.md](24_acls.md) | ACLS（2008、Chiodi–Desjardins 2017）の再初期化：平らな界面で厳密に釣り合うが、FCCD の圧縮・拡散を上回らない（#38） |
| [25_height_curvature.md](25_height_curvature.md) | ψ の高さ関数による曲率：交差の高さ＋CCD は正確なプロファイルで最良、流れでは2次差分の高さが静止液滴に強い（#39） |
| [26_acls_fccd.md](26_acls_fccd.md) | CCD＋ACLS：2008 年の形の FCCD 化（静止界面で最良）、陰的擬似時間は刻み 4 倍まで。格子規模の毛管波を抑えるのは等方拡散だけ（#40） |
| [27_implicit_tension.md](27_implicit_tension.md) | 表面張力の半陰的化：ψ を運ぶ速度を射影後に平滑化して射影し直すと、格子規模の毛管波を抑えられる（#41） |
| [28_eps_width.md](28_eps_width.md) | 界面幅 ε/h を 2〜4 に広げる：曲率・移流・ACLS の精度が1〜3桁良くなるが、R/ε が小さいと逆効果。毛管波と不安定は変わらない（#42） |
| [29_capwave_damping.md](29_capwave_damping.md) | 毛管波の誤差の原因：粗い格子では界面の渦層が解像できず減衰不足。VOF は移流の数値散逸で打ち消している（#43） |
| [30_vorticity_layer.md](30_vorticity_layer.md) | 毛管波の減衰誤差は渦層の解像度 δ/h で決まり、周期の誤差は点数で決まる（法線モードと比較、#44） |
| [31_ccd_viscous_centered.md](31_ccd_viscous_centered.md) | CenteredNS の粘性項の CCD 化：単相は FCCD の面演算子と合わせて4次。毛管波の減衰は射影が2次のままでは改善しない（#45） |
| [32_fccd_projection.md](32_fccd_projection.md) | FCCD の射影とジャンプ補正付き面勾配（滑らかな圧力 p − H j_ext を解く）：寄生流は2次と同水準、毛管波は周期が改善（#46） |
| [33_coupled_stability.md](33_coupled_stability.md) | 毛管ループの連成安定性：粘性で安定限界が下がるのは時間分割の順序のため。粘性演算子は原因でない（layered 形は格子規模で増大）（#47） |
| [34_coupled_viscous.md](34_coupled_viscous.md) | 表面張力・粘性・射影の連成反復：粘性が安定化に働くようになる（N = 64、μ = 1 で境目 1.0 → 2.6）。密度比 100 では未改善（#48） |
| [35_step_jacobian.md](35_step_jacobian.md) | 1ステップの Jacobian の固有値解析：実測の境目と8ケースで一致。密度比 100 で K = 1 が効かないのは、遅れた圧力勾配（気相の α で密度比倍）が粘性を通って射影で消えず振動モード（arg ≈ 0.21π）になるため。残差で反復を止めると境目 1.0 → 1.8〜2.0（N = 64）（#49） |
| [36_coupled_solver.md](36_coupled_solver.md) | 連成反復の収束判定（運動量残差は変化量の4〜28倍）・失敗時の扱い・遅れた圧力の残留の分解（近似射影の項が96 %）・格子依存（収束させれば境目は N によらず 1.7〜1.9）・Anderson 加速（−35〜50 %）・一定時間あたりのコスト（N = 32 では K = 0 が速い）（#50） |
| [37_layered_energy.md](37_layered_energy.md) | layered 形の CCD 粘性でエネルギーが増える原因：壁で層 A（Dirichlet）と層 C（外挿）の閉じ方が違い、D1∘D1 が非対称（対称部 +12.7/h²、‖(I − κL)⁻¹‖ = 8.7）で零固有値が欠陥的。周期境界では増えない。split 形（既定）は問題なし（#51） |
| [38_capwave_layer_diag.md](38_capwave_layer_diag.md) | 毛管波の減衰不足の原因：界面の渦度の尖り（[∂²ₙu_τ] = ∂_τ[p]/μ）がセル中心で表せず散逸が欠ける。演算子の次数・Δt・ε・ジャンプの形によらず δ/h だけで決まり、減衰率の比は格子で採った厳密解のエンストロフィーの比と 1% で一致（#52） |
