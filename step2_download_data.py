"""STEP 2 - download the NASA C-MAPSS dataset (12 small text files) into ./data.
Official source: NASA Prognostics Center of Excellence data repository. This script uses an identical public copy."""
import os, urllib.request
BASE = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"
os.makedirs("data", exist_ok=True)
for fd in ["FD001", "FD002", "FD003", "FD004"]:
    for part in ["train", "test", "RUL"]:
        name = f"{part}_{fd}.txt"; path = os.path.join("data", name)
        if not os.path.exists(path): print("downloading", name); urllib.request.urlretrieve(BASE + name, path)
print("Done:", sorted(os.listdir("data")))
