# 53 方程式の整理：二相の標準経路と時間積分（実装どおりの離散式）

実装した式を一か所にまとめる。式はコードから確かめた（括弧内はファイル）。論文の式との対応と違いは 9 節にまとめる。測定値と経緯は各ノートを参照する。

## 定式化と理論的根拠（このノートの式の導出の要点）
このノートは実装の離散式の一覧なので、各式の根拠をここにまとめる。詳しい導出は、括弧のノートにある。

- **換算重力**：相 k の中で `p = p_true − ρ_k G·(x − Z)` と置くと、`−∇p_true + ρ_k G = −∇p`。重力は相の内部で消える。界面では `p_true` が `[p_true] = σκ` を満たすので、`[p] = σκ + (ρ₂ − ρ₁)G·(x − Z)`。界面の位置の誤差は、この跳びを通じてだけ重力に効く。
- **CLS の流束の補正項**（ノート18）：点の値 ψ に対して、面の点の流束 F の単純な差は `∂_x F + h²/24 ∂_x³F + O(h⁴)`。そのため、面の流束から `h²/24 ∂_x²(ψu)` を引けば 4 次になる。
  - `∂²(ψu) = ψ_xx u + 2ψ_x u_x + ψ u_xx` のうち、最後の項は外す。射影後の u_f は単純な差で発散 0 なので、この項は ψ ≡ 1 の領域に O(h²) の湧き出しを作るため。
  - 外した項の誤差は u の滑らかさだけで決まり、ψ の幅 ε に比例して大きくならない。
- **SSP-RK3**：各段が前進 Euler の段の凸結合（係数 3/4、1/4、1/3、2/3 は非負）。前進 Euler が CFL 条件の下で有界性を保てば、その凸結合も保つ（Shu–Osher）。各段の [0, 1] への切り詰めは、この性質が空間の離散化で崩れる分の補償で、質量を厳密には保たない（そのため次の質量の閉鎖がある）。
- **質量の閉鎖**：w = 4ψ(1 − ψ) は界面の帯でだけ正（ψ = 0、1 で 0）なので、補正は界面に集まる。切り詰めがかからなければ、`Σψ dV` は M₀ に厳密に戻る（`Σ(ψ + c w)dV = Σψ dV + c Σw dV`、c を合わせる）。切り詰めがかかると厳密ではなくなる。
- **滑らかな形のジャンプ**（ノート32）：`p = p̃ + H j_ext` と分けると、`G_f p − B(j) = G_f p̃ + G_f(H j_ext) − B(j)`。j_ext が界面の近くで j_Γ に等しければ、`G_f(H j_ext) − B(j)` は切断面で消え、残りは j_ext の勾配の小さな項。等密度なら p̃ は値も法線微分も連続になり、高次の面勾配を界面越しに使える。
- **近似射影**（5 節）：面の速度は `D_f u_f = 0` を満たす。セルの速度は面の加速度の平均 R_h で補正されるので、`D_f P_f u ≠ 0`（P_f と R_h の往復は恒等ではない）。7 節の誤差の床はこれから来る（ノート51・52）。
- **SBDF2 の射影の刻み** `σ = 2Δt/3`：BDF2 の `α₀ = 3/2` で割った刻み。圧力は、陰的な時刻の Lagrange 乗数としてこの刻みで入る。
- **表面張力の刻み** `f = √(5/12)`：CCD の曲率の格子規模の応答（3 点差分の 12/5 倍）から（ノート19・66）。

## 0. 記号
| 記号 | 意味 |
| --- | --- |
| $h$ | 格子幅（一様格子）。$\Delta$ は面をはさむセル中心の距離 |
| $\mathbf u$、$u_f$ | セル中心の速度、面の法線速度（状態は両方を持つ） |
| $p$、$\tilde p$ | 圧力、ジャンプを除いた滑らかな圧力（滑らかな形） |
| $\psi$ | CLS の変数（相 1 で 1、相 2 で 0） |
| $\rho_k, \mu_k$ | 相 $k = 1, 2$ の密度・粘性 |
| $\alpha_f$ | 面の $1/\rho$ |
| $a_f$ | 面の加速度（ジャンプ・重力） |
| $g$ | セル中心の $a - \alpha\,G p$（Basilisk の `g`） |
| $G_f, D_f, P_f$ | 面勾配、面からの発散、面への補間 |
| $R_h$ | 面からセルへの平均（`face_to_cell`） |
| $\sigma$ | 表面張力係数（6 節以降の $\sigma_{\Delta t}$ は射影の刻みで、別のもの） |

## 1. 支配方程式
一流体の形（相ごとに定数の $\rho_k, \mu_k$）：

$$\rho\left(\partial_t \mathbf u + \mathbf u\cdot\nabla\mathbf u\right) = -\nabla p + \nabla\cdot\left(2\mu \mathsf D(\mathbf u)\right) + \rho\,\mathbf G,\qquad \nabla\cdot\mathbf u = 0,$$

$$[p]_\Gamma = p_1 - p_2 = \sigma\kappa,\qquad [\mathbf u]_\Gamma = 0,\qquad \partial_t\psi + \nabla\cdot(\psi\mathbf u) = 0.$$

**換算重力**（`TwoPhaseCLS.add_reduced_gravity`、Basilisk `reduced.h`）：各相で $p = p_{\rm true} - \rho_k\,\mathbf G\cdot(\mathbf x - \mathbf Z)$ と置くと、相の内部では重力が圧力に吸収される。重力は界面のジャンプに移る。

$$[p]_\Gamma = \sigma\kappa + (\rho_2 - \rho_1)\,\mathbf G\cdot(\mathbf x - \mathbf Z) \equiv j.$$

論文の分解 $\rho\mathbf g = -\nabla(\rho\Phi_g) + \Phi_g\nabla\rho'$（02 章）の残差 $\Phi_g\nabla\rho'$ を、拡散した帯ではなく鋭い界面に集めた形にあたる。

## 2. 空間の演算子
セル中心の格子。面の量は、その面に垂直な成分だけを持つ。

| 演算子 | fd2（Basilisk） | FCCD（`FCCDFaceComplex`） |
| --- | --- | --- |
| $G_f q$ | $(q_R - q_L)/\Delta$ | 面のコンパクト差分（4 次） |
| $P_f q$ | $(q_L + q_R)/2$ | 面のコンパクト補間（4 次、CLS では 6 次） |
| $D_f F$ | $\sum_{\rm ax}(F_{i+1/2} - F_{i-1/2})/h$ | 補正した流束の差（4 次、保存形、ノート03） |
| $R_h$ | 二つの面の平均 | 同左（`face_to_cell`） |

- **粘性**（`CCDViscosity`）：$\rho^{-1}\nabla\cdot(2\mu\mathsf D)$ を、CCD の全応力（高次の残差）と fd2 の解法（低次の補正）の欠陥補正で解く（split 形、ノート06）。
- **対流**：既定は散逸のない $E(\mathbf u) = -D_f\left(P_f(\mathbf u)\,u_f\right)$（`CentralFluxConvection`）。FUCCD₄ と UCCD6 は、$\sigma_r|a|h^7\partial^8$ の風上の散逸を加える（比較用）。

## 3. 界面：CLS
**プロファイル**（`psi_from_phi`）：$\psi = H_\varepsilon(\phi) = \tfrac12\left(1 + \tanh\frac{\phi}{2\varepsilon}\right)$、$\varepsilon = 1.5h$。$\phi$ は相 1 で正の符号付き距離。

**流束と右辺**（`CLSInterface._rhs`、divergence=`"fccd"`）：

$$R(\psi, u_f) = -D_f^{\rm plain} F,\qquad F = (P_f^{(6)}\psi)\,u_f - \frac{h^2}{24}\left(\psi_{xx}u + 2\psi_x u_x\right)_f .$$

- $D_f^{\rm plain}$ は単純な面の差。
- 補正項は、$\psi$ の微分を含む $h^2$ の誤差を除く（面に沿った CCD 微分、壁では 0、ノート18）。
- FCCD の射影と組むときは divergence=`"full"`（FCCD の補正した流束全体の差）にする。

**時間積分**（`advect`）：陽的 RK の各段の値を $[0, 1]$ に切り詰める。既定は SSP-RK3（Shu–Osher 形）：

$$\psi^{(1)} = \psi^n + \Delta t R(\psi^n),\quad \psi^{(2)} = \tfrac34\psi^n + \tfrac14\left(\psi^{(1)} + \Delta t R(\psi^{(1)})\right),\quad \psi^* = \tfrac13\psi^n + \tfrac23\left(\psi^{(2)} + \Delta t R(\psi^{(2)})\right).$$

速度 $u_f$ は段を通して凍結する（何を使うかは 6.3 節）。

**質量の閉鎖**（`mass_correction`、段階 B）：$w = 4\psi(1 - \psi)$ として

$$\psi \leftarrow \mathrm{clip}\left(\psi + \frac{M_0 - \sum\psi\,dV}{\sum w\,dV}\,w,\; 0, 1\right).$$

**再初期化**（任意、`reinit_every`）：ξ-SDF。$\phi = \varepsilon\ln\frac{\psi}{1-\psi}$ から交差点までの距離を作り、一様なずらしで $\sum\psi\,dV$ を閉じる。

## 4. 物性・曲率
**物性**（`TwoPhase._properties`）：$\rho(\psi) = \rho_2 + (\rho_1 - \rho_2)\,\mathrm{clip}(\psi)$、$\mu$ も同じ形。面では $\alpha_f = 1/\rho(\bar\psi_f)$、$\mu_f = \mu(\bar\psi_f)$（$\bar\psi_f$ は両隣の算術平均）。

**GFM**（`pressure_jump=True`）：$H = [\psi > \tfrac12]$。面で $H_L \ne H_R$ のとき、その面は切断面である。交差の位置は $\theta = (\psi_L - \tfrac12)/(\psi_L - \psi_R)$ で、

$$\alpha_f = \frac{1}{\theta\rho_L + (1-\theta)\rho_R}$$

とし、それ以外の面は $\alpha_f = 1/\rho_k$（相の内部は定数）。セルの $\rho$ と面の $\mu$ は $\psi$ の一次式のまま。

**曲率**（`psi_curvature`）：CCD の $D^{(1)}, D^{(2)}$ と $\psi_{xy} = D_y^{(1)}D_x^{(1)}\psi$ を使い、

$$\kappa = -\frac{\psi_y^2\psi_{xx} - 2\psi_x\psi_y\psi_{xy} + \psi_x^2\psi_{yy}}{\left(\psi_x^2 + \psi_y^2\right)^{3/2}},\qquad 0.01 < \psi < 0.99 \text{ の帯の外では } \kappa = 0.$$

## 5. 圧力ジャンプと射影
**交差点のジャンプ**：セルの $j = \sigma\kappa + (\rho_2-\rho_1)\mathbf G\cdot(\mathbf x - \mathbf Z)$ を、$j_\Gamma = (1-\theta)j_L + \theta j_R$ と補間する。

**鋭い形**（`jump_mode="sharp"`、ノート16）：ジャンプを補正した面勾配 $G_\Gamma(p; j) = G_f p - B(j)$、$B(j) = j_\Gamma (H_R - H_L)/\Delta$。面の加速度として

$$a_f = \alpha_f\, B(j).$$

**滑らかな形**（`jump_mode="smooth"`、ノート32）：$j_{\rm ext}$ を、最も近い交差点の $j_\Gamma$ で延長した場とする。$p = \tilde p + H j_{\rm ext}$ と分けて $\tilde p$ を解く。

$$a_f = \alpha_f\left[B(j) - G_f(H j_{\rm ext})\right] = \frac{\alpha_f}{\Delta}\left[j_\Gamma(H_R - H_L) - (H_R j_{{\rm ext},R} - H_L j_{{\rm ext},L})\right].$$

等密度なら、$\tilde p$ は値も法線微分も連続なので、FCCD の面勾配を界面越しに使える。

**射影**（`project`、Basilisk の近似射影）：面速度 $u_f^*$ に対して、

$$D_f\left(\alpha_f G_f\,\delta p\right) = \frac{D_f u_f^*}{\sigma_{\Delta t}},\qquad u_f = u_f^* - \sigma_{\Delta t}\,\alpha_f G_f\,\delta p .$$

- セル中心の圧力勾配：$g = R_h\left(a_f - \alpha_f G_f p\right)$（`centered_gradient`）。
- セルの補正：$\mathbf u \leftarrow \mathbf u + \sigma_{\Delta t}\,g$（`correction`）。
- **セルの速度は離散的に発散 0 ではない**（$R_h$ と $P_f$ の往復）。これが「近似射影」で、時間精度の床の原因になる（7 節）。

## 6. 時間積分
### 6.1 Basilisk の分割（`CenteredNS`、基準）

1. BCG で面速度を半ステップ予測し、$\Delta t/2$ で射影する。
2. 移流：$\mathbf u_{\rm adv}$。
3. 粘性：$(I - \tfrac{\Delta t}{\rho}L)\,\mathbf w = \mathbf u_{\rm adv} + \Delta t\, g^n$、$\mathbf u^* = \mathbf w - \Delta t\,g^n$。
4. $u_f^* = P_f\mathbf u^* + \Delta t\, a_f^{n+1}$。射影（$\sigma_{\Delta t} = \Delta t$、全圧力。Basilisk は $p^n$ から反復するので増分形と同値）。
5. $\mathbf u^{n+1} = \mathbf u^* + \Delta t\, g^{n+1}$。

### 6.2 多段法の IMEX（`IMEXMultistepNS`、ノート51）
一般形（$E$ は陽的な対流、$V = \rho^{-1}\nabla\cdot(2\mu\mathsf D)$ は陰的）：

$$\sum_{j\ge0}\alpha_j\,\mathbf u^{n+1-j} = \Delta t\left[\sum_{j\ge0}\beta_j E^{n-j} + \sum_{j\ge0}\gamma_j V^{n+1-j}\right] - \Delta t\,\nabla p\text{（陰的な時刻）}.$$

$\sigma_{\Delta t} = \Delta t/\alpha_0$、$\tau = \sigma_{\Delta t}\gamma_0$ として、増分圧力補正（IPC）で解く：

$$(I - \tau V)\,\mathbf u^* = \sum_{j\ge1}\left(-\frac{\alpha_j}{\alpha_0}\right)\mathbf u^{n+1-j} + \sigma_{\Delta t}\left(\sum_j\beta_j E^{n-j} + \sum_{j\ge1}\gamma_j V^{n+1-j} + g^n\right),$$

$$u_f^* = P_f\left(\mathbf u^* - \sigma_{\Delta t} g^n\right) + \sigma_{\Delta t}\left(a_f^{n+1} - \alpha_f G_f p^n\right),\quad \text{射影で }\delta p,\quad p^{n+1} = p^n + \delta p,\quad \mathbf u^{n+1} = \mathbf u^* + \sigma_{\Delta t}\left(g^{n+1} - g^n\right).$$

- 過去の $V^k$ は、解いた予測子から $V^k = (\mathbf u^*_k - {\rm rhs}_k)/\tau_k$ として保存する。
- **係数**（`schemes.py`）：実際の時刻の点 $t^{n+1}, t^n, \dots$ の Lagrange 多項式から、毎ステップ計算する（可変の刻みでも正確）。

| 族 | 係数 |
| --- | --- |
| BDF-$k$ | $\alpha_j = \Delta t\,\ell_j'(t^{n+1})$（点 $t^{n+1},\dots,t^{n+1-k}$）、$\gamma = (1)$ |
| EXT-$k$ | $\beta_j = \ell_j(t^{n+1})$（点 $t^n,\dots,t^{n+1-k}$） |
| AB-$k$ | $\beta_j = \frac{1}{\Delta t}\int_{t^n}^{t^{n+1}}\ell_j\,dt$（同じ点） |
| AM-$k$ | $\alpha = (1,-1)$、$\gamma_j = \frac{1}{\Delta t}\int_{t^n}^{t^{n+1}}\ell_j\,dt$（点 $t^{n+1},\dots,t^{n+2-k}$） |
| $\theta$ | $\alpha = (1,-1)$、$\gamma = (\theta, 1-\theta)$ |
| MCNAB2 | $\gamma = (\tfrac9{16}, \tfrac38, \tfrac1{16})$、$\beta = (\tfrac32, -\tfrac12)$（刻みは一定のみ） |
| CNLF | $\alpha = (\tfrac12, 0, -\tfrac12)$、$\beta = (1)$、$\gamma = (\tfrac12, 0, \tfrac12)$（刻みは一定のみ） |

例：SBDF2（一定の刻み）は $\alpha = (\tfrac32, -2, \tfrac12)$、$\beta = (2, -1)$ なので $\sigma_{\Delta t} = \tfrac23\Delta t$（論文 11b1 第5段の形）。

### 6.3 界面の輸送の速度（`TwoPhaseIMEX(transport=...)`、ノート50・51）
$\psi^{n+1} = \mathcal S_{\Delta t}(\psi^n; \hat u_f)$（3 節の RK、速度は凍結）。$\hat u_f$ の選び方：

| `transport` | $\hat u_f$ | 運動学の時間次数 |
| --- | --- | --- |
| `frozen`（論文） | $u_f^n$ | 1（陽的 Euler と同じ） |
| `ab2`、`ab3` | $\sum_j \beta^{\rm AB}_j u_f^{n-j}$ | 2、3 |
| `pc` | $\tfrac12\left(u_f^n + u_f^{n+1,*}\right)$：$u_f^n$ で 1 回進めて予測し、状態を戻して進め直す | 2（台形則） |

### 6.4 IMEX Runge–Kutta（`IMEXRKNS`、`TwoPhaseIMEXRK`、ノート52）
ARS 型の表 $(A^E, A^I, b^E, b^I, c)$、$A^I_{ii} = \gamma$。段 $i \ge 1$：

$$\psi_i = \mathrm{clip}\Big(\psi^n + \Delta t\sum_{j<i}A^E_{ij}R(\psi_j, u_{f,j})\Big),\qquad \rho, \alpha_f, \mu \leftarrow \psi_i,$$

$$(I - \gamma\Delta t\,V)\,\mathbf u_i^* = \mathbf u^n + c_i\Delta t\,g^n + \Delta t\sum_{j<i}\left(A^E_{ij}E_j + A^I_{ij}V_j\right),$$

$a_f$ は $\psi_i$ の界面のジャンプ、射影の刻みは $c_i\Delta t$（IPC は 6.2 節と同じ）。最後は

$$\psi^{n+1} = \psi^n + \Delta t\sum_j b^E_j R_j\ \text{（＋ 質量の閉鎖）},$$

$\mathbf u^{n+1}$ は最後の段の値とする。いずれかの部分が stiffly accurate でない場合は、$\mathbf u^n + \Delta t\sum_j(b^E_jE_j + b^I_jV_j)$ を $p^n$ から射影する。

### 6.5 刻みの制約
- **表面張力：** $\Delta t \le f\sqrt{\rho_m h^3/(\pi\sigma)}$、$\rho_m = \tfrac12(\rho_{\min} + \rho_{\max})$（面の $1/\alpha_f$ の両端）。既定は $f = \sqrt{5/12}$（CCD 曲率の硬さ、ノート19）。
- **CFL：** Basilisk の `timestep`（CLS では CFL ≤ 0.5）。
- **陽的な対流の安定限界：** UCCD6・FUCCD の実軸、RK の表の虚軸・実軸。虚軸上に安定域がない EXT2・ARS(1,1,1)・(1,2,1)・(2,2,2) では掛けない（粘性で安定に保つ）。

## 7. 誤差の構造（測定で確かめたもの）
| 誤差 | 大きさ | 根拠 |
| --- | --- | --- |
| 近似射影の交差項 | $O(\Delta t\,h^p)$。kx ≠ ky の場で、N = 16 では次数を 1 に落とし、N = 64 で消える。対流があると、3 次の RK も誤差が約 3e-6 で止まる | ノート51・52 |
| IPC の分割誤差 | $O(\Delta t^2)$。SBDF3・4 も 2 次止まり（SBDF3 の誤差は SBDF2 の約 6.5 倍）。各段で射影する IMEX-RK は Stokes で 3 次 | ノート51・52 |
| 運動学（`frozen`） | 時間 1 次。毛管波（16 点）で −18.8% → −13.0%（刻み 1/32 まで）。`ab2` で消える | ノート50 |
| CLS の運動学（空間） | 交点が ±h で平均した速度で動く。毛管波の半離散の減衰不足（16 点で約 −12.7%） | ノート41・49 |

## 8. 選択肢の一覧
| 層 | 選択肢 | 既定（論文） |
| --- | --- | --- |
| 面の演算子 | `faces="fd2"`、`"fccd"`（等密度のみ） | fd2 |
| 対流 | `central`、`fuccd`、`uccd6` | central（論文は UCCD6・FUCCD） |
| 運動量の時間積分 | 多段法：`sbdf1`〜`sbdf4`、`cnab2`、`mcnab2`、`cnlf`、任意の (`bdf<k>`/`am<k>`/`theta<x>`, `ext<k>`/`ab<k>`)。RK：`ars111`〜`ars443` | sbdf2 |
| ψ を運ぶ速度 | `frozen`、`ab2`、`ab3`、`pc`（RK では段ごと） | frozen |
| ψ の積分法 | `ssprk3`、`ssprk2`、`heun3`、`rk4`、`euler` | ssprk3 |
| ジャンプ | `sharp`、`smooth` | fd2 では sharp |
| 再初期化 | なし、`xi_sdf`、`compression`、ACLS | なし |

## 9. 論文（11b1）との対応と違い
| 段 | 論文 | 本実装 | 違い |
| --- | --- | --- | --- |
| 1 | $\psi$ を TVD-RK3、$u_f^n$ | 3 節、`frozen` | 同じ（1 次であることを確認） |
| 2 | Ridge–Eikonal、$\lambda_\Gamma$（鋭い体積）と $\varepsilon_{\rm eff}$（拡散した質量） | ξ-SDF と一様なずらしの一つ | 二つの閉鎖は未実装 |
| 3 | $\rho, \mu$ を $\psi$ の一次式で | 4 節 | 切断面の $\alpha_f$ は GFM |
| 4 | $\psi$ の CCD の曲率 | 4 節 | 壁に接する界面の閉包は未実装 |
| 5 | IMEX-BDF2、浮力の残差を面で | 6.2 節、換算重力をジャンプへ | 浮力は鋭い界面に集めた形 |
| 5a・6 | 分相 FCCD-PPE + 欠陥補正 + HFE、$\delta j$ | GFM の 1 回のマルチグリッド（fd2）、FCCD は滑らかな形（等密度） | 密度比のある FCCD は未実装 |
| 6 | 毛管の補正量 $\hat c_\sigma = s_f - B_h\mu$ | なし | |
| 7 | $G_\Gamma\delta p$ による補正 | 5 節・6.2 節 | 同じ面加速度で、射影・補正・セルの勾配が閉じる |
