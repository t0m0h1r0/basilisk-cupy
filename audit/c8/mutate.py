"""監修 #173：局所の収支の検証器の検出力。取り込んだ 1 掃過の配列を故意に壊して、どの検査が落ちるかを見る。
さらに、(a) conserving_fresh_dilation=True（正しい別の選択肢）で誤検出しないか、(b) L6 が実際に何を測るか。"""
import copy, warnings
import numpy as np
warnings.filterwarnings("ignore")
from cases import r0_local_budget as B
from cases.r0_aligned_free_stream import shape

cap = B._flow(lambda g: shape("drop", g), "base")
rec = cap[4]                      # ある掃過（1 回目）
base = B.from_library(rec, (1.0, 1e-3))
ax = base["axis"]
def tup(i, j):
    return (i, j) if ax == 0 else (j, i)
def run(sw):
    v = B.verify(sw, 2.5)
    out = {"L1": v["L1"]["max"], "L3": v["L3"]["max"]}
    for key in ("L2", "L4", "L5"):
        out[key] = max(r["max"] for r in v[key].values())
    return " ".join("%s=%.0e" % (k, out[k]) for k in ("L1", "L2", "L3", "L4", "L5"))
def copy_sw():
    s = dict(base)
    for k in ("F", "c1", "un"):
        s[k] = np.array(base[k], copy=True)
    for k in ("Fq", "ff", "q1", "q0", "Sq", "conc"):
        s[k] = [np.array(a, copy=True) for a in base[k]]
    return s
# 混合セルの面を選ぶ
c = base["c0"]
cells = np.argwhere((c > 0.3) & (c < 0.7))
i, j = (cells[0] if ax == 0 else cells[0][::-1])      # i: 掃過方向の添字
face = tup(i + 1, j); lo = tup(i, j); hi = tup(i + 1, j)
k = 0                                                 # q1（相 1）の x 運動量
print("unmodified        :", run(base))
s = copy_sw(); s["F"][face] += 1e-3; print("F one face        :", run(s))
s = copy_sw(); s["Fq"][k][face] += 1e-3; print("Fq one face       :", run(s))
s = copy_sw(); s["c1"][lo] += 1e-3; print("c1 one cell       :", run(s))
s = copy_sw(); s["q1"][k][lo] += 1e-3; print("q1 one cell       :", run(s))
# 面の値が誤っている（流束と更新はその値と整合）：D2 や再構成の誤りの型
s = copy_sw()
d = 0.5 * s["ff"][k][face]
s["ff"][k][face] += d
dF = d * s["F"][face]
s["Fq"][k][face] += dF
s["q1"][k][lo] -= dF; s["q1"][k][hi] += dF
print("wrong face value  :", run(s), " (face value x1.5; flux, update consistent; moved volume %.2e)" % abs(base["F"][face]))
# 相 2 の運動量が相 1 の体積流束を使う
s = copy_sw(); m = base["phase"].index(1)
new = s["ff"][m] * s["F"]; diff = new - s["Fq"][m]
s["Fq"][m] = new
s["q1"][m] = s["q1"][m] + B._faces(diff, ax, True) - B._faces(diff, ax, False)
print("phase-2 wrong vol :", run(s))

# (a) 正しい別の選択肢で誤検出しないか
cap2 = B._flow(lambda g: shape("drop", g), "base", conserving_fresh_dilation=True)
r = B.summarise(cap2, scale_q=2.5)
print("fresh_dilation=True worst:", {k: "%.1e" % v for k, v in r["worst"].items()})
# (b) L6 は完全版では呼び出し側の指定で 0 になる。指定を外すと何が出るか
for name in ("vortex", "free-stream"):
    cap3, scale = B._run(name, "full")
    print("L6 full/%s: lin=True %.1e   lin=False %.1e" % (name, B.summarise(cap3, scale_q=scale, lin=True)["worst"]["L6"], B.summarise(cap3, scale_q=scale)["worst"]["L6"]))
