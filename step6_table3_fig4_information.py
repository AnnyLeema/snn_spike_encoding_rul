"""STEP 6 - Table 3 and Figure 4 (paper Sections 3.3 and 5.2): spikes per window and information loss
of each encoding on the test windows. Needs only the data (averaged over the three seeds, as in run.py)."""
import numpy as np, csv
from n1 import load, ENC, decode
from common_plot import *
rows = []
for fd in ["FD001", "FD004"]:
    Xte = load(fd)[4]
    for e in ENCS:
        sp, rc = [], []
        for seed in [0, 1, 2]:
            s = ENC[e](Xte, seed + 200)
            sp.append(0 if e == "direct" else s.reshape(len(s), -1).sum(1).mean())
            rc.append(np.sqrt(((decode(e, s, Xte) - Xte) ** 2).mean()))
        rows.append([fd, e, s.shape[2], round(np.mean(sp)), round(np.mean(rc), 3)])
with open("tables/Table3_encodings.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "encoding", "input_channels", "spikes_per_window", "reconstruction_rmse"]); w.writerows(rows)
print("saved tables/Table3_encodings.csv"); [print(r) for r in rows]
fig, ax = plt.subplots(figsize=(6, 4))
for fd, e, ch, sp, rc in rows:
    if fd != "FD001" or e == "direct": continue
    ax.scatter(sp, rc, s=70, color=COL[e], zorder=3)
    ax.annotate(LAB[e], (sp, rc), xytext={"sod-anchored": (-80, 6), "ttfs": (6, -10)}.get(e, (6, 4)), textcoords="offset points", fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("Input spikes per window"); ax.set_ylabel("Reconstruction RMSE (normalised units)")
for sp_ in ["top", "right"]: ax.spines[sp_].set_visible(False)
save(fig, "Fig4_information")
