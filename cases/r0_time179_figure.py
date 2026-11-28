"""Figure of docs/notes/125: python -m cases.r0_time179_figure (docs/figures/r0_time179.png)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cases.r0_time179 import OFFSETS, EXACT_DAMPING
R=json.load(open("docs/figures/r0_time179_t1.json"))
T2=json.load(open("docs/figures/r0_time179_t2.json"))
col={"c":"#257de0","d1":"#f47230","b2":"#00b894","d0":"#f2a600"}
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,3,figsize=(12,3.6))
for k,n in enumerate((32,64)):
    a=ax[k]
    for j,s in enumerate(("c","d1","b2","d0")):
        v=[]
        for o in OFFSETS:
            r=[r for r in R if r["scheme"]==s and r["n"]==n and r["dx"]==o[0] and r["dy"]==o[1] and r["tolerance"]==1e-10][0]
            v.append(r["damping"])
        a.plot(np.arange(6)+(j-1.5)*0.1, v, "o", ms=6, color=col[s], label=s, mec="white", mew=1)
    a.axhline(EXACT_DAMPING,color="#666666",lw=1,ls="--"); a.text(5.3,EXACT_DAMPING,"exact",va="bottom",ha="right",color="#666666")
    a.set_xticks(range(6)); a.set_xticklabels(["(%g,%g)"%o for o in OFFSETS],rotation=30)
    a.set_title("T1 capillary-wave damping, N = %d"%n); a.set_xlabel("interface offset (x phase, height) [cells]")
    a.grid(axis="y",color="#e5e5e5",lw=0.6)
ax[0].set_ylabel("damping rate"); ax[0].legend(frameon=False,ncol=4,loc="center")
a=ax[2]
for s in ("c","d1","b2"):
    r=[r for r in T2 if r["case"]=="static_drop" and r["scheme"]==s][0]
    u=np.array(r["umax_trace"]); a.semilogy(np.arange(len(u))*20,u,color=col[s],lw=2,label=s)
a.axvline(1000,color="#666666",lw=0.8,ls=":")
a.set_title("T2 static drop, mu/100, N = 32"); a.set_xlabel("step"); a.set_ylabel("max|u|"); a.legend(frameon=False)
a.grid(color="#e5e5e5",lw=0.6)
fig.tight_layout(); fig.savefig("docs/figures/r0_time179.png",dpi=150)
