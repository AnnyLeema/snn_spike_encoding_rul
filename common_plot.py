import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 10
os.makedirs("figures", exist_ok=True); os.makedirs("tables", exist_ok=True)
ENCS = ["rate", "ttfs", "population", "send-on-delta", "sod-anchored", "level-crossing", "direct"]
LAB = {"rate": "Rate", "ttfs": "TTFS", "population": "Population", "send-on-delta": "Send-on-delta", "sod-anchored": "SoD-anchored",
       "level-crossing": "Level-crossing", "direct": "Direct (analog)", "lstm": "LSTM", "cnn": "1D-CNN"}
COL = dict(zip(ENCS + ["lstm", "cnn"], ["#E57373", "#FFB74D", "#BA68C8", "#4FC3F7", "#1565C0", "#81C784", "#757575", "#212121", "#8D6E63"]))
def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"figures/{name}.pdf", bbox_inches="tight"); plt.close(fig)
    print("saved figures/%s.png (and .pdf)" % name)
