"""STEP 5 - Figure 2 (paper Section 3.4): architecture diagram. Needs nothing."""
from dl import *
import os; os.makedirs("figures", exist_ok=True)
f, ax = fig(14, 5.2)
for x, t in [(1.4, "Telemetry window"), (4.6, "Spike encoder"), (8.4, "Two-timescale SNN"), (12.3, "Readout and metrics")]:
    ax.text(x, 4.95, t, fontsize=11.5, fontweight="bold", ha="center")
box(ax, 1.4, 2.7, 2.3, 2.0, "14 sensors\n× 30 cycles\n(normalised\nto [0, 1])", "gray", fs=10)
box(ax, 4.6, 2.7, 2.9, 3.2, "One of seven encodings\n(T = 4 steps per cycle)\n\nrate · TTFS · population\nsend-on-delta\nSoD-anchored\nlevel-crossing · direct", "yel", fs=9.6)
box(ax, 8.4, 3.55, 3.2, 1.25, "Layer 1: 64 recurrent LIF\nfast learnable decay (β ≈ 0.95)", "sub", fs=9.6)
box(ax, 8.4, 1.8, 3.2, 1.25, "Layer 2: 64 LIF\nslow learnable decay (β ≈ 0.99)", "teal", fs=9.6)
box(ax, 12.3, 3.55, 2.9, 1.25, "Time-averaged linear\nreadout → RUL", "tr", fs=9.8)
box(ax, 12.3, 1.8, 2.9, 1.25, "RMSE · NASA score\nspikes · SynOps · energy", "gray", fs=9.6)
arr(ax, (2.55, 2.7), (3.15, 2.7)); arr(ax, (6.05, 2.7), (6.8, 3.4)); arr(ax, (8.4, 2.92), (8.4, 2.43))
arr(ax, (10.0, 1.8), (10.85, 3.2)); arr(ax, (12.3, 2.92), (12.3, 2.43))
ax.annotate("", xy=(10.0, 3.95), xytext=(10.0, 3.15), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=1.3", color="#2E5E8E"))
ax.text(10.55, 3.55, "recurrent\nR₁", fontsize=8.5, color="#2E5E8E", ha="left", va="center")
ax.set_ylim(0.6, 5.2); save(f, "figures/Fig2_architecture"); print("saved figures/Fig2_architecture.png (and .pdf)")
