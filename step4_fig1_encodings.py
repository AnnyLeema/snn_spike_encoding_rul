"""STEP 4 - Figure 1 (paper Section 3.3): spike trains of the seven encodings. Needs only the data."""
import numpy as np
from n1 import load, ENC
from common_plot import *
Xtr, ytr, Xva, yva, Xte, yte = load("FD001"); x = Xte[:1]; sensor = 3          # sensor index 3 = sensor 7
fig, axs = plt.subplots(len(ENCS), 1, figsize=(9, 8), sharex=True)
for ax, e in zip(axs, ENCS):
    s = ENC[e](x, 0)[0]; F = x.shape[2]
    if e == "direct":
        ax.plot(s[:, sensor], color=COL[e]); ax.set_ylim(0, 1)
    else:
        ch = {"population": [sensor * 5 + k for k in range(5)], "send-on-delta": [sensor, sensor + F],
              "sod-anchored": [sensor, sensor + F, sensor + 2 * F], "level-crossing": [sensor, sensor + F]}.get(e, [sensor])
        for j, c in enumerate(ch): ax.vlines(np.where(s[:, c] > 0)[0], j, j + 0.8, color=COL[e], lw=1.2)
        ax.set_ylim(0, len(ch)); ax.set_yticks([])
    ax.set_ylabel(LAB[e], rotation=0, ha="right", va="center", fontsize=9)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axs[-1].set_xlabel("Time step (4 steps per engine cycle, 30 cycles)")
save(fig, "Fig1_encodings")
