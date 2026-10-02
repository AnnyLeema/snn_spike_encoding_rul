"""STEP 9 - Tables 5, 6, 7, 8 (paper Section 5) as CSV files. Run after run.py (step 7) and analyze.py (step 8)."""
import pickle, collections, csv, numpy as np
from scipy.stats import rankdata, friedmanchisquare
from common_plot import ENCS, LAB
S = pickle.load(open("summary.pkl", "rb")); R = pickle.load(open("results.pkl", "rb"))
def fmt(fd, m, k): v = S[(fd, m)][k]; return "%.2f ± %.2f" % (v[0], v[1])
def w(name, hdr, rows):
    with open(f"tables/{name}.csv", "w", newline="") as f: c = csv.writer(f); c.writerow(hdr); c.writerows(rows)
    print(f"\n{name}"); print(" | ".join(hdr)); [print(" | ".join(map(str, r))) for r in rows]
M = ["lstm", "cnn"] + ENCS
w("Table5_accuracy", ["model", "FD001 RMSE", "FD001 score", "FD004 RMSE", "FD004 score", "FD004 RMSE uncapped"],
  [[LAB[m], fmt("FD001", m, "rmse"), "%.0f" % S[("FD001", m)]["nasa"][0], fmt("FD004", m, "rmse"), "%.0f" % S[("FD004", m)]["nasa"][0], "%.2f" % S[("FD004", m)]["rmse_uncapped"][0]] for m in M])
by = collections.defaultdict(dict)
for r in R:
    if r["model"] in ENCS: by[(r["fd"], r["seed"])][r["model"]] = r["rmse"]
cells = [[d[e] for e in ENCS] for d in by.values()]
rk = np.array([rankdata(c) for c in cells])
w("Table6_ranks", ["encoding", "overall average rank"], sorted([[LAB[e], "%.2f" % rk[:, i].mean()] for i, e in enumerate(ENCS)], key=lambda z: float(z[1])))
print("Friedman:", friedmanchisquare(*np.array(cells).T))
w("Table7_efficiency", ["model", "input spikes", "layer-1 spikes", "layer-2 spikes", "SynOps", "MACs", "energy FD001 nJ", "energy FD004 nJ"],
  [[LAB[m]] + ["%.0f" % S[("FD001", m)][k][0] if k in S[("FD001", m)] else "-" for k in ["in_spikes", "l1_spikes", "l2_spikes", "synops", "macs"]]
   + ["%.1f" % S[("FD001", m)]["energy_nJ"][0], "%.1f" % S[("FD004", m)]["energy_nJ"][0]] for m in ENCS + ["lstm", "cnn"]])
w("Table8_robustness", ["model", "FD001 clean", "FD001 noise", "FD001 stuck", "FD004 clean", "FD004 noise", "FD004 stuck"],
  [[LAB[m]] + ["%.2f" % S[(fd, m)][k][0] for fd in ["FD001", "FD004"] for k in ["rmse", "rmse_noise", "rmse_drop"]] for m in M])
