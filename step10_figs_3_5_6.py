"""STEP 10 - Figures 3, 5 and 6 (paper Sections 5.1, 5.5, 5.6). Run after steps 7 and 8."""
import pickle, numpy as np
from common_plot import *
S = pickle.load(open("summary.pkl", "rb")); R = pickle.load(open("results.pkl", "rb"))
# Figure 3: predictions for FD004 test engines (seed 0)
yt = np.loadtxt("data/RUL_FD004.txt"); o = np.argsort(yt)
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.plot(np.minimum(yt[o], 125), "k-", lw=1.5, label="True RUL (capped at 125)")
for m in ["lstm", "sod-anchored"]:
    r = [r for r in R if r["fd"] == "FD004" and r["model"] == m and r["seed"] == 0][0]
    ax.plot(r["pred"][o], ".", ms=4, color=COL[m], label=LAB[m] if m == "lstm" else "SNN – SoD-anchored")
ax.set_xlabel("Test engines sorted by true RUL"); ax.set_ylabel("RUL (cycles)"); ax.legend(frameon=False, fontsize=9)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
save(fig, "Fig3_predictions")
# Figure 5: accuracy vs estimated energy
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, fd in zip(axs, ["FD001", "FD004"]):
    for m in ENCS + ["lstm", "cnn"]:
        v = S[(fd, m)]; ax.errorbar(v["energy_nJ"][0], v["rmse"][0], yerr=v["rmse"][1], fmt="o" if m in ENCS else "s", color=COL[m], ms=7, capsize=3)
        off = {("FD004", "level-crossing"): (-70, 6), ("FD004", "send-on-delta"): (6, -10), ("FD001", "level-crossing"): (6, -8), ("FD001", "sod-anchored"): (6, -9)}.get((fd, m), (6, 3))
        ax.annotate(LAB[m], (v["energy_nJ"][0], v["rmse"][0]), xytext=off, textcoords="offset points", fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlabel("Estimated energy per inference (nJ, log scale)"); ax.set_ylabel("Test RMSE (capped RUL)"); ax.set_title(fd)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
save(fig, "Fig5_pareto")
# Figure 6: robustness on FD001
ms = ["lstm", "cnn"] + ENCS; x0 = np.arange(len(ms)); w = 0.27
fig, ax = plt.subplots(figsize=(10, 3.6))
for k, (key, lab, c) in enumerate([("rmse", "Clean", "#455A64"), ("rmse_noise", "Gaussian noise (σ = 0.05)", "#FFA726"), ("rmse_drop", "One sensor stuck", "#EF5350")]):
    ax.bar(x0 + (k - 1) * w, [S[("FD001", m)][key][0] for m in ms], w, label=lab, color=c)
ax.set_xticks(x0); ax.set_xticklabels([LAB[m] for m in ms], rotation=20, ha="right"); ax.set_ylabel("Test RMSE (FD001)")
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
save(fig, "Fig6_robustness")
