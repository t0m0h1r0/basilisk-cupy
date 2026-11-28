import json, sys
R=[json.loads(l) for l in open(sys.argv[1])]
R.sort(key=lambda r:(r["case"],r["ratio"],r["mur"],r["curv"],r["n"],r["variant"]))
for r in R:
    h=r["hist"]
    u=[x["u"] for x in h]; ke=[x["ke"] for x in h]
    late=max(u[len(u)//2:]) if u else float("nan"); early=max(u[:len(u)//2]) if u else float("nan")
    extra=""
    if r["case"]=="wave" and h: extra=" amp(first,last,max) %.4f %.4f %.4f"%(h[0]["amp"],h[-1]["amp"],max(x["amp"] for x in h))
    print("%-6s %-10s N=%3d ratio=%-5g mur=%-3g %-7s t=%.2f steps=%6d umax=%.2e u(early/late)=%.2e/%.2e ke_end=%.2e df=%.2e%s %s"%(r["case"],r["variant"],r["n"],r["ratio"],r["mur"],r["curv"],r["t_end"],r["steps"],r["umax"],early,late,ke[-1] if ke else float("nan"),r["df"],extra,r["failed"]))
