"""監修（C）：ノート115 の表を JSON から数え直す。"""
import json, collections
D = json.load(open("docs/figures/r0p2_pressure_momentum.json"))
print("keys", list(D))
for r in sorted(D["momentum"], key=lambda r: (r["conserving"], r["kind"], r["n"])):
    print("mom %-6s cons=%-5s n=%3d change %.2e  history %s %s" % (r["kind"], r["conserving"], r["n"], r["momentum_change"], " ".join("%.0e" % h["momentum_change"] for h in r["history"]), r["failed"] or ""))
H = D["hydrostatic"]
sel = [r for r in H if r["gravity"] == "reduced" and r["ratio"] == 1000.0 and r["conserving"]]
bad = [r for r in sel if r["failed"] or max(max(x["cells"], x["faces"]) for x in r["history"]) > 1e-8]
print("reduced conditions", len(sel), "fail", len(bad))
for r in bad:
    print("  ", r["kind"], r["theta"], r.get("offset", 0.0137), r["n"], "%.1e" % max(x["cells"] for x in r["history"]))
sel = [r for r in H if r["gravity"] == "plain" and r["ratio"] == 1000.0 and r["conserving"]]
print("plain: t=1 cells by (kind, theta, offset): N=32/64/128")
g = collections.defaultdict(dict)
for r in sel:
    g[(r["kind"], r["theta"], r.get("offset", 0.0137))][r["n"]] = r["history"][-1]["cells"]
for k in sorted(g):
    print("  ", k, " / ".join("%.1e" % g[k][n] for n in sorted(g[k])))
