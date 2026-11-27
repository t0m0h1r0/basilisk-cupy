"""監修（独立の検証）：ノート106 の 2 節「界面の移動で連続」を、検出力のある形で測り直す。

ノート106 の continuity() は、(1) 200 点の L で R(L + 1e-7) − R(L) を見るだけなので、標本の間にある跳びは見えない。
(2) データが滑らかな u の厳密な平均なので、形の違う再構築どうしの差が打切り誤差の大きさしかなく、演算子の跳びが隠れる。

ここでは、幾何によらない固定の無作為データ q を使い、L を細かく掃引して隣り合う再構築の差を測る。
差の大きい所は刻みを 1/10、1/100 にして、差が刻みに比例して縮む（連続）か、縮まない（跳び）かを判定する。

    （origin/main の作業ツリーで）python audit/short_run_continuity.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

N = 32
F1 = 0.4


def recon(args):
    L, ratio, kind = args
    from cases.short_runs import _recon
    h = 1.0 / N
    c0 = (N // 2 + 1 - F1) * h
    c1 = c0 + L * h
    q = np.random.default_rng(7).normal(size=N)          # 幾何によらない固定のデータ
    xs = (np.arange(N * 16) + 0.5) / (N * 16)
    # 動く界面のすぐ近くの点は、相が入れ替わるので除く（界面から 0.02 セル以内）
    keep = np.abs(((xs - c1 + 0.5) % 1.0) - 0.5) > 0.05 * h
    try:
        v = _recon(N, ratio, c1 % 1.0, c0 % 1.0, q, kind)(xs)
    except Exception as e:  # noqa: BLE001
        return L, None, repr(e)[:60]
    return L, np.where(keep, v, np.nan), None


def sweep(Ls, ratio, kind, ex):
    res = list(ex.map(recon, [(L, ratio, kind) for L in Ls], chunksize=16))
    out = []
    for (L0, v0, e0), (L1, v1, e1) in zip(res[:-1], res[1:]):
        if v0 is None or v1 is None:
            out.append((L0, L1, None))
        else:
            out.append((L0, L1, float(np.nanmax(np.abs(v1 - v0)))))
    return out


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "blend"
    report = {}
    with ProcessPoolExecutor(10) as ex:
        for ratio in (1.0, 1e3, 1e6):
            d = 2e-3
            Ls = np.arange(0.06, 6.7, d)
            sw = sweep(Ls, ratio, kind, ex)
            undefined = sum(1 for s in sw if s[2] is None)
            ok = [s for s in sw if s[2] is not None]
            ok.sort(key=lambda s: -s[2])
            med = float(np.median([s[2] for s in ok]))
            rows = []
            for L0, L1, diff in ok[:6]:
                # 細分：差の最大の小区間を 10 等分、さらに 10 等分
                fine = sweep(np.linspace(L0, L1, 11), ratio, kind, ex)
                f1 = max(fine, key=lambda s: -1 if s[2] is None else s[2])
                finer = sweep(np.linspace(f1[0], f1[1], 11), ratio, kind, ex)
                f2 = max(finer, key=lambda s: -1 if s[2] is None else s[2])
                rows.append({"L": round(L0, 4), "diff_d": diff, "diff_d/10": f1[2], "diff_d/100": f2[2]})
            report[ratio] = {"step": d, "undefined_intervals": undefined, "median_diff": med, "largest": rows}
            print(f"密度比 {ratio:g}（{kind}）：未定義 {undefined}、差の中央値 {med:.2e}（刻み {d}）")
            for r in rows:
                verdict = "跳び" if r["diff_d/100"] > 0.3 * r["diff_d"] else "連続"
                print(f"  L = {r['L']:.4f}: 差 {r['diff_d']:.3e} → 刻み 1/10 で {r['diff_d/10']:.3e} → 1/100 で {r['diff_d/100']:.3e}  [{verdict}]")
    json.dump(report, open(f"short_run_continuity_{kind}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
